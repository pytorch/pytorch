# Owner(s): ["module: optimizer"]

import re
from copy import deepcopy

import torch
import torch.nn.utils.stateless as stateless
from torch.fx.experimental.proxy_tensor import make_fx
from torch.optim._stateless import swap_in_optimizer_params_and_state
from torch.testing._internal.common_utils import (
    load_tests,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


class ChainedLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(2, 2)
        self.l2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.l2(self.l1(x))


class TestSwapInOptimizerParamsAndState(TestCase):
    class _TestException(Exception):
        pass

    def _init_optimizer(self, freeze_l1: bool = False):
        # ChainedLinear + AdamW stepped once. With freeze_l1=True, l1's
        # params have requires_grad=False so no state is lazy-init'd for them
        # (their packed ids appear in param_groups but are absent from state).
        module = ChainedLinear()
        if freeze_l1:
            for p in module.l1.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [module.l1.weight, module.l1.bias],
                        "lr": 0.1,
                        "weight_decay": 0.01,
                    },
                    {
                        "params": [module.l2.weight, module.l2.bias],
                        "lr": 0.01,
                        "weight_decay": 0.0,
                    },
                ]
            )
        loss = module(torch.randn(4, 2)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return module, optimizer

    def _make_reparam_inputs(self):
        module, optimizer = self._init_optimizer()
        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }
        # Build a swap-in state with distinguishing values so tests can tell
        # the optimizer is reading the swap-in (not the live) state.
        swapin_optim_state = deepcopy(optimizer.state_dict())
        for per_param_state in swapin_optim_state["state"].values():
            for k, v in per_param_state.items():
                if isinstance(v, torch.Tensor):
                    per_param_state[k] = v + 0.1
        return module, optimizer, parameters, swapin_optim_state

    def _assert_optimizer_restored(
        self, optimizer, state_before, state_dict_before, params_before
    ):
        self.assertIs(optimizer.state, state_before)
        self.assertEqual(optimizer.state_dict(), state_dict_before)
        for group, params in zip(optimizer.param_groups, params_before, strict=True):
            self.assertIs(group["params"], params)

    def test_swap_in_optimizer_params_and_state_rejects_invalid_inputs(self):
        _, optimizer, parameters, base_osd = self._make_reparam_inputs()

        def expect(opt, params, bad_input, pattern):
            with self.assertRaisesRegex(RuntimeError, re.escape(pattern)):
                with swap_in_optimizer_params_and_state(opt, params, bad_input):
                    pass

        with self.subTest("uninitialized"):
            m = ChainedLinear()
            uninit_opt = torch.optim.AdamW(m.parameters(), lr=0.1)
            uninit_params = {
                n: torch.randn_like(p, requires_grad=p.requires_grad)
                for n, p in m.named_parameters()
            }
            expect(
                uninit_opt,
                uninit_params,
                deepcopy(uninit_opt.state_dict()),
                "swap_in_optimizer_params_and_state requires initialized optimizer state.",
            )

        with self.subTest("non_raw_state"):
            osd = deepcopy(base_osd)
            osd["state"] = {"l1.weight": {}}
            osd["param_groups"][0]["params"] = ["l1.weight", "l1.bias"]
            expect(
                optimizer,
                parameters,
                osd,
                "param_groups[*]['params'] entries keyed by packed parameter ids",
            )

        with self.subTest("bad_state_field"):
            expect(
                optimizer,
                parameters,
                {"state": [], "param_groups": []},
                "swapin_optim_state['state'] to be a dict",
            )

        with self.subTest("bad_param_groups_field"):
            expect(
                optimizer,
                parameters,
                {"state": {}, "param_groups": {}},
                "swapin_optim_state['param_groups'] to be a list",
            )

        with self.subTest("group_count_mismatch"):
            osd = deepcopy(base_osd)
            osd["param_groups"].pop()
            expect(optimizer, parameters, osd, "different number of parameter groups")

        with self.subTest("group_size_mismatch"):
            osd = deepcopy(base_osd)
            osd["param_groups"][0]["params"].pop()
            expect(
                optimizer,
                parameters,
                osd,
                "param group 0 has a different number of params",
            )

        with self.subTest("non_dict_param_state"):
            osd = deepcopy(base_osd)
            osd["state"][osd["param_groups"][0]["params"][0]] = torch.tensor(1.0)
            expect(
                optimizer,
                parameters,
                osd,
                "per-parameter optimizer state entries to be dictionaries",
            )

        with self.subTest("non_param_state"):
            osd = deepcopy(base_osd)
            osd["state"]["global_step"] = 42
            expect(
                optimizer,
                parameters,
                osd,
                "got extra keys ['global_step']",
            )

        with self.subTest("swapin_extra_key"):
            osd = deepcopy(base_osd)
            osd["param_groups"][0]["bogus"] = 1
            expect(
                optimizer,
                parameters,
                osd,
                "Keys only in swap-in: ['bogus']",
            )

        with self.subTest("swapin_missing_key"):
            osd = deepcopy(base_osd)
            del osd["param_groups"][0]["betas"]
            expect(
                optimizer,
                parameters,
                osd,
                "Keys only in live: ['betas']",
            )

        with self.subTest("too_few_parameters"):
            short = dict(list(parameters.items())[:-1])
            expect(
                optimizer,
                short,
                base_osd,
                "match optimizer.param_groups ordering",
            )

        with self.subTest("tensor_keyed_state"):
            osd = deepcopy(base_osd)
            osd["state"][torch.tensor(0)] = {}
            expect(
                optimizer,
                parameters,
                osd,
                "state keyed by packed parameter ids",
            )

    def test_swap_in_optimizer_params_and_state_handles_missing_param_state(self):
        # When some params have ``requires_grad=False`` their grads stay
        # ``None`` after ``loss.backward(); step()`` and the optimizer never
        # lazy-inits state for them. The resulting state_dict has packed ids
        # in param_groups that are absent from ``state`` —
        # swap_in_optimizer_params_and_state must accept this shape, and mutations made in the context
        # to such params must stay local (not pollute the input
        # ``swapin_optim_state``).
        module, optimizer = self._init_optimizer(freeze_l1=True)
        swapin_optim_state = deepcopy(optimizer.state_dict())
        # Sanity: state has fewer entries than param_groups (2 trainable l2
        # params have state; 2 frozen l1 params do not).
        self.assertEqual(len(swapin_optim_state["param_groups"][0]["params"]), 4)
        self.assertEqual(len(swapin_optim_state["state"]), 2)
        frozen_param_id = swapin_optim_state["param_groups"][0]["params"][0]
        self.assertNotIn(frozen_param_id, swapin_optim_state["state"])

        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }

        with swap_in_optimizer_params_and_state(
            optimizer, parameters, swapin_optim_state
        ):
            # First param is frozen (l1.weight); mutations to its swapped-in
            # state must not leak back into the input ``swapin_optim_state``.
            frozen_param = optimizer.param_groups[0]["params"][0]
            optimizer.state[frozen_param]["step"] = torch.tensor(7.0)

        self.assertNotIn(frozen_param_id, swapin_optim_state["state"])

    def test_swap_in_optimizer_params_and_state_supports_named_parameters_init(self):
        # Optimizer initialized with named_parameters() adds a ``param_names``
        # key to each param group; ``state`` is still keyed by packed ints.
        # Swap-in must round-trip param_names along with the other group fields.
        module = ChainedLinear()
        optimizer = torch.optim.AdamW(module.named_parameters(), lr=0.01)
        x = torch.randn(4, 2)
        loss = module(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        param_names_before = [g["param_names"] for g in optimizer.param_groups]

        parameters = {
            name: torch.randn_like(p, requires_grad=p.requires_grad)
            for name, p in module.named_parameters()
        }
        swapin_optim_state = deepcopy(optimizer.state_dict())

        with swap_in_optimizer_params_and_state(
            optimizer, parameters, swapin_optim_state
        ):
            # param_names is re-applied to the swapped-in group
            self.assertEqual(
                [g["param_names"] for g in optimizer.param_groups],
                param_names_before,
            )

        # And restored on exit
        self.assertEqual(
            [g["param_names"] for g in optimizer.param_groups],
            param_names_before,
        )

    def test_swap_in_optimizer_params_and_state_restores_after_exception(self):
        _, optimizer, parameters, swapin_optim_state = self._make_reparam_inputs()
        state_before = optimizer.state
        state_dict_before = deepcopy(optimizer.state_dict())
        params_before = [group["params"] for group in optimizer.param_groups]
        try:
            with swap_in_optimizer_params_and_state(
                optimizer, parameters, swapin_optim_state
            ):
                self.assertTrue(optimizer.state is not state_before)
                self.assertTrue(
                    optimizer.param_groups[0]["params"] is not params_before[0]
                )
                first_param = optimizer.param_groups[0]["params"][0]
                optimizer.state[first_param]["step"] = torch.tensor(123.0)
                optimizer.state[first_param]["exp_avg"].add_(1)
                optimizer.param_groups[0]["lr"] = 5.0
                raise self._TestException
        except self._TestException:
            self._assert_optimizer_restored(
                optimizer, state_before, state_dict_before, params_before
            )

    # Skipped under PYTORCH_TEST_WITH_DYNAMO=1: dynamo's OptimizerVariable
    # ._set_capturable writes capturable=True to the param_group VT, and at
    # any graph break SideEffects flushes the entire VT (including that write)
    # onto the live optimizer dict — silently flipping `capturable=True` on
    # the user's CPU optimizer. See https://github.com/pytorch/pytorch/issues/182706.
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/182706")
    def test_swap_in_optimizer_params_and_state_reflects_state_mutations(self):
        _, optimizer, parameters, swapin_optim_state = self._make_reparam_inputs()
        state_before = optimizer.state
        state_dict_before = deepcopy(optimizer.state_dict())
        params_before = [group["params"] for group in optimizer.param_groups]

        first_param_id = swapin_optim_state["param_groups"][0]["params"][0]
        exp_avg_before = swapin_optim_state["state"][first_param_id]["exp_avg"].clone()

        with swap_in_optimizer_params_and_state(
            optimizer, parameters, swapin_optim_state
        ):
            first_param = optimizer.param_groups[0]["params"][0]
            # In-place tensor op: tensor is shared with swapin_optim_state
            # (no clone), so the mutation is visible after exit.
            optimizer.state[first_param]["exp_avg"].add_(1)
            # Dict swap-in: per-param state dict is shallow-copied, so
            # structural changes do not leak back.
            optimizer.state[first_param]["step"] = torch.tensor(123.0)
            optimizer.param_groups[0]["lr"] = 5.0

        self._assert_optimizer_restored(
            optimizer, state_before, state_dict_before, params_before
        )
        self.assertEqual(
            swapin_optim_state["state"][first_param_id]["exp_avg"],
            exp_avg_before + 1,
        )
        self.assertNotEqual(
            swapin_optim_state["state"][first_param_id]["step"],
            torch.tensor(123.0),
        )

    def test_make_fx_swap_in_optimizer_params_and_state_tensor_reassignment_stays_local(
        self,
    ):
        module = ChainedLinear()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(4, 2)
        loss = module(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }
        swapin_optim_state = deepcopy(optimizer.state_dict())
        first_param_id = swapin_optim_state["param_groups"][0]["params"][0]
        original_momentum = swapin_optim_state["state"][first_param_id][
            "momentum_buffer"
        ]

        def f(state, x):
            params, optimizer_state = state
            with stateless._reparametrize_module(module, params):
                with swap_in_optimizer_params_and_state(
                    optimizer, params, optimizer_state
                ):
                    p = optimizer.param_groups[0]["params"][0]
                    # Intentional tensor reassignment inside make_fx. The new
                    # tensor is a graph-internal value (proxy for ``zeros_like``)
                    # and the assignment lives in the shallow-copied per-param
                    # state dict, so it must not propagate to ``optimizer_state``.
                    optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    for param in module.parameters():
                        param.grad = torch.ones_like(param)
                    optimizer.step()
                    return module(x)

        gm = make_fx(f)((parameters, swapin_optim_state), x)
        # Reassigned slot in the input dict still points at the original tensor.
        self.assertIs(
            swapin_optim_state["state"][first_param_id]["momentum_buffer"],
            original_momentum,
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, state, x):
    state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10, state_11, state_12, state_13, state_14, state_15, state_16, state_17, state_18, state_19, state_20, state_21, x_1, = fx_pytree.tree_flatten_spec([state, x], self._in_spec)
    zeros_like = torch.ops.aten.zeros_like.default(state_1, pin_memory = False)
    ones_like = torch.ops.aten.ones_like.default(state_1, pin_memory = False)
    ones_like_1 = torch.ops.aten.ones_like.default(state_2, pin_memory = False)
    ones_like_2 = torch.ops.aten.ones_like.default(state_3, pin_memory = False)
    ones_like_3 = torch.ops.aten.ones_like.default(state_4, pin_memory = False)
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new.default('Optimizer.step#SGD.step')
    mul_ = torch.ops.aten.mul_.Tensor(zeros_like, 0.9);  zeros_like = None
    add_ = torch.ops.aten.add_.Tensor(mul_, ones_like);  mul_ = ones_like = None
    add__1 = torch.ops.aten.add_.Tensor(state_1, add_, alpha = -0.1);  state_1 = add_ = None
    mul__1 = torch.ops.aten.mul_.Tensor(state_6, 0.9);  state_6 = None
    add__2 = torch.ops.aten.add_.Tensor(mul__1, ones_like_1);  mul__1 = ones_like_1 = None
    add__3 = torch.ops.aten.add_.Tensor(state_2, add__2, alpha = -0.1);  state_2 = add__2 = None
    mul__2 = torch.ops.aten.mul_.Tensor(state_7, 0.9);  state_7 = None
    add__4 = torch.ops.aten.add_.Tensor(mul__2, ones_like_2);  mul__2 = ones_like_2 = None
    add__5 = torch.ops.aten.add_.Tensor(state_3, add__4, alpha = -0.1);  state_3 = add__4 = None
    mul__3 = torch.ops.aten.mul_.Tensor(state_8, 0.9);  state_8 = None
    add__6 = torch.ops.aten.add_.Tensor(mul__3, ones_like_3);  mul__3 = ones_like_3 = None
    add__7 = torch.ops.aten.add_.Tensor(state_4, add__6, alpha = -0.1);  state_4 = add__6 = None
    _record_function_exit = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit = None
    t = torch.ops.aten.t.default(add__1);  add__1 = None
    addmm = torch.ops.aten.addmm.default(add__3, x_1, t);  add__3 = x_1 = t = None
    t_1 = torch.ops.aten.t.default(add__5);  add__5 = None
    addmm_1 = torch.ops.aten.addmm.default(add__7, addmm, t_1);  add__7 = addmm = t_1 = None
    return pytree.tree_unflatten([addmm_1], self._out_spec)""",
        )

    def test_make_fx_reparametrize_module_and_optimizer_records_aten_ops(self):
        module = ChainedLinear()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(4, 2)
        loss = module(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        parameters = {
            name: torch.randn_like(param, requires_grad=param.requires_grad)
            for name, param in module.named_parameters()
        }
        swapin_optim_state = deepcopy(optimizer.state_dict())

        def f(state, x):
            params, optimizer_state = state
            with stateless._reparametrize_module(module, params):
                with swap_in_optimizer_params_and_state(
                    optimizer, params, optimizer_state
                ):
                    y = module(x)
                    for param in module.parameters():
                        param.grad = torch.ones_like(param)
                    optimizer.step()
                    return y

        gm = make_fx(f)((parameters, swapin_optim_state), x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, state, x):
    state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10, state_11, state_12, state_13, state_14, state_15, state_16, state_17, state_18, state_19, state_20, state_21, x_1, = fx_pytree.tree_flatten_spec([state, x], self._in_spec)
    t = torch.ops.aten.t.default(state_1)
    addmm = torch.ops.aten.addmm.default(state_2, x_1, t);  x_1 = t = None
    t_1 = torch.ops.aten.t.default(state_3)
    addmm_1 = torch.ops.aten.addmm.default(state_4, addmm, t_1);  addmm = t_1 = None
    ones_like = torch.ops.aten.ones_like.default(state_1, pin_memory = False)
    ones_like_1 = torch.ops.aten.ones_like.default(state_2, pin_memory = False)
    ones_like_2 = torch.ops.aten.ones_like.default(state_3, pin_memory = False)
    ones_like_3 = torch.ops.aten.ones_like.default(state_4, pin_memory = False)
    _record_function_enter_new = torch.ops.profiler._record_function_enter_new.default('Optimizer.step#SGD.step')
    mul_ = torch.ops.aten.mul_.Tensor(state_5, 0.9);  state_5 = None
    add_ = torch.ops.aten.add_.Tensor(mul_, ones_like);  mul_ = ones_like = None
    add__1 = torch.ops.aten.add_.Tensor(state_1, add_, alpha = -0.1);  state_1 = add_ = add__1 = None
    mul__1 = torch.ops.aten.mul_.Tensor(state_6, 0.9);  state_6 = None
    add__2 = torch.ops.aten.add_.Tensor(mul__1, ones_like_1);  mul__1 = ones_like_1 = None
    add__3 = torch.ops.aten.add_.Tensor(state_2, add__2, alpha = -0.1);  state_2 = add__2 = add__3 = None
    mul__2 = torch.ops.aten.mul_.Tensor(state_7, 0.9);  state_7 = None
    add__4 = torch.ops.aten.add_.Tensor(mul__2, ones_like_2);  mul__2 = ones_like_2 = None
    add__5 = torch.ops.aten.add_.Tensor(state_3, add__4, alpha = -0.1);  state_3 = add__4 = add__5 = None
    mul__3 = torch.ops.aten.mul_.Tensor(state_8, 0.9);  state_8 = None
    add__6 = torch.ops.aten.add_.Tensor(mul__3, ones_like_3);  mul__3 = ones_like_3 = None
    add__7 = torch.ops.aten.add_.Tensor(state_4, add__6, alpha = -0.1);  state_4 = add__6 = add__7 = None
    _record_function_exit = torch.ops.profiler._record_function_exit._RecordFunction(_record_function_enter_new);  _record_function_enter_new = _record_function_exit = None
    return pytree.tree_unflatten([addmm_1], self._out_spec)""",
        )


if __name__ == "__main__":
    run_tests()
