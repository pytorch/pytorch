# Owner(s): ["oncall: distributed"]

import copy
import functools
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FlatParameter,
    MixedPrecision,
    OptimStateKeyType,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    HandleInitMode,
    ParamExecOrderPolicy,
    ParamExecOrderState,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
}
HANDLE_INIT_MAPPING = {
    "module_level": HandleInitMode.MODULE_LEVEL,
    "param_level": HandleInitMode.PARAM_LEVEL,
}

default_mp = MixedPrecision(
    param_dtype=torch.float16,
    buffer_dtype=torch.float16,
    reduce_dtype=torch.float16,
)

# Params and buffers are not cast, comm only happens
# in reduced precision.
mp_only_reduce = MixedPrecision(reduce_dtype=torch.float16)

# Only parameters are cast (thus comm should happen in the param_dtype precision)
mp_only_param_and_buf = MixedPrecision(param_dtype=torch.float16, buffer_dtype=torch.float16)

# Nothing is cast (thus param, comm, grad, and buffer should be in the full precision)
mp_no_mixed_precision = MixedPrecision()


class DoubleConv(nn.Module):
    def __init__(self, embed) -> None:
        super().__init__()
        self.embed = embed
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, 8)
        # Handles the edge case when this parameter is wrapped together with
        # parameters in the children modules
        self.weight = torch.nn.Parameter(
            torch.randn(16 * 5 * 5, 16 * 5 * 5)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = x @ self.weight
        x = F.relu(self.fc(x))
        return x


class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 3 * 32 * 32))

    def forward(self, x):
        x = x @ self.weight
        x = x.reshape([4, 3, 32, 32]) # (N, C, H, W)
        return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # test shared module
        self.shared = Embedding()
        self.conv1 = DoubleConv(self.shared)
        self.conv2 = DoubleConv(self.shared)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.root_weight = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x @ self.root_weight
        x = F.relu(self.fc3(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_inp_shape() -> torch.Size:
        return torch.Size([4, 8])


class TestFSDPExecOrderPolicy(FSDPTest):
    @property
    def device(self):
        return torch.device("cuda")

    def _init_fsdp_ddp(self, model_class, optim_class, handle_init_mode, *model_args, **model_kwargs):
        torch.manual_seed(42)
        model = model_class(*model_args, **model_kwargs).cuda()
        group = dist.distributed_c10d._get_default_group()

        fsdp_model = FSDP(
            copy.deepcopy(model), group, auto_wrap_policy=ParamExecOrderPolicy(handle_init_mode),
        )
        fsdp_optim = optim_class(fsdp_model.parameters(), lr=1e-3)
        ddp_model = DDP(model, device_ids=[self.rank], process_group=group)
        ddp_optim = optim_class(ddp_model.parameters(), lr=1e-3)
        return fsdp_model, fsdp_optim, ddp_model, ddp_optim

    def _warmup_fsdp(self, fsdp_model, optim_class, inp_shape, num_warmup_iters: int = 1):
        for _ in range(num_warmup_iters):
            inp = torch.randn(inp_shape).to(self.rank)
            # No optimizer step to keep the model parameters unchanged
            self._step_model(fsdp_model, inp)
        # Reset gradients and construct new optimizer
        return optim_class(fsdp_model.parameters(), lr=1e-3)

    def _check_fsdp_param_parity(self, fsdp_model, ref_model):
        with FSDP.summon_full_params(fsdp_model):
            fsdp_named_params = list(fsdp_model.named_parameters())
            ref_named_params = list(ref_model.named_parameters())
            # sort fsdp_named_params and ref_named_params based on the names.
            # This sort is needed since for some cases, parameters are generated
            # in different sequences.
            fsdp_named_params = sorted(fsdp_named_params, key=lambda x: x[0])
            ref_named_params = sorted(ref_named_params, key=lambda x: x[0])

            if len(fsdp_named_params) != len(ref_named_params):
                print(f"[Rank {self.rank}] expected len={len(ref_named_params)} got {len(fsdp_named_params)}")
                print(f"[Rank {self.rank}] fsdp names: {[n for n, _ in fsdp_model.named_parameters()]}")
            self.assertEqual(len(fsdp_named_params), len(ref_named_params))
            for (_, p1), (_, p2) in zip(fsdp_named_params, ref_named_params):
                torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-4)

    def _step_model(self, model, inp, optim=None):
        if optim is not None:
            optim.zero_grad()
        out = model(inp)
        loss = out.sum()
        loss.backward()
        if optim is not None:
            optim.step()
        return loss

    def _check_fsdp_train_parity(
        self,
        fsdp_model: FSDP,
        fsdp_optim: torch.optim.Optimizer,
        ref_model: nn.Module,
        ref_optim: torch.optim.Optimizer,
        inp_shape: torch.Size,
        num_iters: int,
    ):
        """Checks that FSDP model parameters match those of the reference,
        trains both for a few iterations, and checks that the losses and
        parameters match."""
        self._check_fsdp_param_parity(fsdp_model, ref_model)
        losses: List[List[torch.Tensor]] = []
        for _ in range(num_iters):
            inp = torch.randn(inp_shape).to(self.rank)
            iter_losses = [
                self._step_model(fsdp_model, inp, fsdp_optim),
                self._step_model(ref_model, inp, ref_optim),
            ]
            losses.append(iter_losses)
        for l1, l2 in losses:
            torch.testing.assert_close(l1, l2, rtol=1e-4, atol=1e-4)
        self._check_fsdp_param_parity(fsdp_model, ref_model)

    def _strip_module_prefix(self, optim_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips the prefix "module." from the parameter name keys in
        ``optim_state_dict``. The prefix arises from wrapping a model with DDP.
        Args:
            optim_state_dict (Dict[str, Any]): Optimizer state dict from an
                optimizer used for a DDP model that has been rekeyed to
                parameter names.
        """
        new_osd = {"state": {}, "param_groups": []}
        prefix = "module."
        for param_name, param_state in optim_state_dict["state"].items():
            assert param_name.startswith(prefix)
            new_osd["state"][param_name.lstrip(prefix)] = param_state
        for param_group in optim_state_dict["param_groups"]:
            new_params = []
            for param_name in param_group["params"]:
                assert param_name.startswith(prefix)
                new_params.append(param_name.lstrip(prefix))
            new_param_group = copy.copy(param_group)
            new_param_group["params"] = new_params
            new_osd["param_groups"].append(new_param_group)
        return new_osd

    @skip_if_lt_x_gpu(2)
    def test_pre_warmup(self):
        """Tests that on the FSDP model is constructed with one original
        parameter per ``FlatParameter``."""
        model_class = CNN
        optim_class = torch.optim.Adam
        fsdp_model, _, ddp_model, _ = self._init_fsdp_ddp(
            model_class, optim_class, HandleInitMode.PARAM_LEVEL,
        )
        num_ddp_params = len(list(ddp_model.parameters()))
        num_fsdp_params = 0
        for param in fsdp_model.parameters():
            self.assertTrue(isinstance(param, FlatParameter))
            self.assertEqual(param._num_params, 1)
            num_fsdp_params += 1
        self.assertEqual(num_ddp_params, num_fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("handle_init_mode", ["module_level", "param_level"])
    def test_train_flow(self, handle_init_mode: HandleInitMode):
        """Tests training parity with DDP."""
        model_class = CNN
        optim_class = torch.optim.Adam
        num_iters = 5
        fsdp_model, fsdp_optim, ddp_model, ddp_optim = self._init_fsdp_ddp(
            model_class, optim_class, HANDLE_INIT_MAPPING[handle_init_mode],
        )
        inp_shape = model_class.get_inp_shape()
        # Use deterministic convolution algorithms for numerical stability
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            fsdp_optim = self._warmup_fsdp(fsdp_model, optim_class, inp_shape)
            self._check_fsdp_train_parity(
                fsdp_model, fsdp_optim, ddp_model, ddp_optim, inp_shape, num_iters,
            )

    # TODO (awgu): `_sharded_pre_load_state_dict_hook()` needs some work to
    # support non-recursive wrapping
    @skip_if_lt_x_gpu(2)
    @parametrize("model_state_dict_type", ["state_dict"])
    @parametrize("handle_init_mode", ["module_level", "param_level"])
    def test_train_with_full_state_dict(
        self,
        model_state_dict_type: str,
        handle_init_mode: HandleInitMode,
    ):
        """Tests training parity with DDP when loading FSDP from a full state
        dict (for both model and optimizer)."""
        num_iters = 5
        model_class = CNN
        optim_class = torch.optim.Adam
        fsdp_model, fsdp_optim, ddp_model, ddp_optim = self._init_fsdp_ddp(
            model_class, optim_class, HANDLE_INIT_MAPPING[handle_init_mode],
        )
        inp_shape = model_class.get_inp_shape()
        fsdp_optim = self._warmup_fsdp(fsdp_model, optim_class, inp_shape)
        # Run the DDP model for a few iterations
        for _ in range(num_iters):
            inp = torch.randn(inp_shape).to(self.rank)
            self._step_model(ddp_model, inp, ddp_optim)
        # Load model state dict from DDP to FSDP
        ddp_state_dict = ddp_model.module.state_dict()
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[model_state_dict_type]):
            fsdp_model.load_state_dict(ddp_state_dict)
        # Load optim state dict from DDP to FSDP
        ddp_optim_state_dict = self._strip_module_prefix(
            FSDP.rekey_optim_state_dict(
                ddp_optim.state_dict(), OptimStateKeyType.PARAM_NAME, ddp_model,
            )
        )
        fsdp_optim.load_state_dict(
            FSDP.shard_full_optim_state_dict(ddp_optim_state_dict, fsdp_model),
        )
        # Run both models and check parity
        self._check_fsdp_train_parity(
            fsdp_model, fsdp_optim, ddp_model, ddp_optim, inp_shape, num_iters,
        )

    @skip_if_lt_x_gpu(2)
    def test_wrap_structure(self):
        """
        Tests that ``nn.Parameter`` are correctly grouped as ``FlatParameter``.
        """
        model = CNN().cuda()
        group = dist.distributed_c10d._get_default_group()
        # Used to test non-default module_level_group_policy
        module_level_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DoubleConv, nn.Linear},
        )
        param_exec_order_policy = ParamExecOrderPolicy(
            handle_init_mode=HandleInitMode.MODULE_LEVEL,
            module_level_group_policy=module_level_policy,
        )
        fsdp_model = FSDP(model, group, auto_wrap_policy=param_exec_order_policy)
        correct_prefixed_param_names = [
            # here `shared.weight` should be in the same wrap as `root_weight`
            # because the root module `model` is the lowest common ancestor
            # of the modules `conv1` and `conv2` that share `shared.weight`.
            {'root_weight', 'shared.weight'},
            {'fc3.weight', 'fc3.bias'},
            {'fc2.weight', 'fc2.bias'},
            {'conv2.weight', 'conv2.conv1.weight', 'conv2.conv1.bias', 'conv2.conv2.weight', 'conv2.conv2.bias'},
            {'conv2.fc.weight', 'conv2.fc.bias'},
            {'conv1.weight', 'conv1.conv1.weight', 'conv1.conv1.bias', 'conv1.conv2.weight', 'conv1.conv2.bias'},
            {'conv1.fc.weight', 'conv1.fc.bias'},
        ]
        for handle in fsdp_model._handles:
            self.assertIn(
                set(handle.flat_param._prefixed_param_names),
                correct_prefixed_param_names,
            )
        self.assertEqual(len(fsdp_model._handles), len(correct_prefixed_param_names))

    @skip_if_lt_x_gpu(2)
    @parametrize("iters", [1, 3])
    def test_reconstruct_reverse_gradient_ready_order(self, iters: int):
        """
        Tests that ``FlatParameter`` s are reconstructed following reverse gradient
        ready order after the first iteration.
        """
        model = CNN().cuda()
        group = dist.distributed_c10d._get_default_group()
        # Used to test non-default module_level_group_policy
        module_level_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DoubleConv, nn.Linear},
        )
        param_exec_order_policy = ParamExecOrderPolicy(
            handle_init_mode=HandleInitMode.MODULE_LEVEL,
            module_level_group_policy=module_level_policy,
        )
        fsdp_model = FSDP(model, group, auto_wrap_policy=param_exec_order_policy)
        self.assertTrue(fsdp_model._use_param_exec_order_policy)
        self.assertTrue(fsdp_model._param_exec_order_state, ParamExecOrderState.UNINITIALIZED)
        params_list = copy.deepcopy(list(fsdp_model.parameters()))
        (
            root_shared_weight,
            fc3_weight,
            fc2_weight,
            conv2_weight,
            conv2_fc_weight,
            conv1_weight,
            conv1_fc_weight,
        ) = params_list
        for _ in range(iters):
            inp_shape = CNN.get_inp_shape()
            input = torch.randn(inp_shape).to(self.rank)
            output = fsdp_model(input)
            loss = output.sum()
            loss.backward()
        params_exec_order_list = list(fsdp_model.parameters())
        self.assertEqual(
            params_exec_order_list,
            [
                root_shared_weight,
                conv1_weight,
                conv1_fc_weight,
                conv2_weight,
                conv2_fc_weight,
                fc3_weight,
                fc2_weight,
            ]
        )
        self.assertTrue(fsdp_model._param_exec_order_state, ParamExecOrderState.INITIALIZED)

    @skip_if_lt_x_gpu(2)
    @parametrize("mp_config", [
        default_mp, mp_only_reduce, mp_only_param_and_buf, mp_no_mixed_precision
    ])
    def test_fsdp_mixed_precision(self, mp_config: MixedPrecision):
        """
        Tests that ``FlatParameter`` dtypes are expected when using mixed precision.
        """
        model = CNN().cuda()
        group = dist.distributed_c10d._get_default_group()
        # Used to test non-default module_level_group_policy
        module_level_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DoubleConv, nn.Linear},
        )
        param_exec_order_policy = ParamExecOrderPolicy(
            handle_init_mode=HandleInitMode.MODULE_LEVEL,
            module_level_group_policy=module_level_policy,
        )
        fsdp_model = FSDP(
            model,
            group,
            auto_wrap_policy=param_exec_order_policy,
            mixed_precision=mp_config,
        )
        inp_shape = CNN.get_inp_shape()
        input = torch.randn(inp_shape).to(self.rank)
        output = fsdp_model(input)
        loss = output.sum()
        loss.backward()
        expected_param_type = (
            mp_config.param_dtype if mp_config.param_dtype is not None
            else torch.float32
        )
        for p in fsdp_model.parameters():
            self.assertEqual(p.dtype, expected_param_type)

instantiate_parametrized_tests(TestFSDPExecOrderPolicy)

if __name__ == "__main__":
    run_tests()
