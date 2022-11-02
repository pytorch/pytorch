# Owner(s): ["oncall: distributed"]

import bisect
import sys
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._shard_utils import _gather_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    OptimStateKeyType,
    StateDictType,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

STATE_DICT_TYPE = [StateDictType.FULL_STATE_DICT, StateDictType.SHARDED_STATE_DICT]

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class _OSDCommMethod(Enum):
    """Method for communicating the optimizer state dict for internal tests."""

    BROADCAST_OBJECT_LIST = auto()
    SCATTER_FULL_OSD = auto()
    FLATTEN_SHARDED_OSD = auto()


class _ModelClass(Enum):
    """Different model type to test."""

    NESTED = auto()
    TRANSFORMER = auto()


class Bias(torch.nn.Module):
    """This module applies a 1D additive bias with dimension ``dim``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim > 0
        torch.manual_seed(0)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))

    def forward(self, x):
        return x + self.bias


class BlockA(torch.nn.Module):
    """
    Used to define interesting nested structure for FSDP wrapping.
    BlockA
        Bias0
            bias
        weight
        Bias1
            bias
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        assert all(v > 0 for v in (in_dim, out_dim))
        torch.manual_seed(0)
        self.bias_module0 = Bias(out_dim)
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias_module1 = Bias(out_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x @ self.weight
        x = self.bias_module0(x)
        x = self.relu(x)  # ensure biases have different gradients
        x = self.bias_module1(x)
        return x


class BlockB(torch.nn.Module):
    """
    Used to define interesting nested structure for FSDP wrapping.
    BlockB
        weight
        Bias
            bias
        Bias
            bias
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        assert all(v > 0 for v in (in_dim, out_dim))
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias_module0 = Bias(out_dim)
        self.bias_module1 = Bias(out_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x @ self.weight
        x = self.bias_module0(x)
        x = self.relu(x)  # ensure biases have different gradients
        x = self.bias_module1(x)
        return x


class NestedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = BlockB(5, 7)
        self.block1 = BlockB(7, 7)
        self.bias = torch.nn.Parameter(torch.randn((5,)))
        self.block2 = torch.nn.Sequential(
            BlockA(7, 9),
            BlockA(9, 9),
            BlockB(9, 5),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.block0(x))
        x = self.relu(self.block1(x))
        x = self.relu(self.block2(x))
        x = x + self.bias
        return x

    def get_input(self, device):
        BATCH_SIZE = 8
        return (torch.randn((BATCH_SIZE, 5)).to(device),)

    def get_loss(self, inp, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def wrap(
        model: torch.nn.Module,
        group: Optional[dist.ProcessGroup] = None,
        ignore_modules: bool = False,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        # Flatten Bias0; then flatten weight and Bias1 together into `block1`
        model.block1.bias_module0 = FSDP(
            model.block1.bias_module0,
            process_group=group,
            **fsdp_kwargs,
        )
        model.block1 = FSDP(model.block1, process_group=group, **fsdp_kwargs)
        # Flatten Bias0; flatten Bias1; then flatten weight into `block2[1]`
        model.block2[1].bias_module0 = FSDP(
            model.block2[1].bias_module0,
            process_group=group,
            **fsdp_kwargs,
        )
        model.block2[1].bias_module1 = FSDP(
            model.block2[1].bias_module1,
            process_group=group,
            **fsdp_kwargs,
        )
        model.block2[1] = FSDP(model.block2[1], process_group=group, **fsdp_kwargs)
        # Flatten weight, Bias, bias into `block2[2]`
        ignored_modules = [model.block2[2].bias_module0] if ignore_modules else None
        model.block2[2] = FSDP(
            model.block2[2],
            process_group=group,
            ignored_modules=ignored_modules,
            **fsdp_kwargs,
        )
        return model

    @staticmethod
    def wrap_alt(
        model: torch.nn.Module,
        group: Optional[dist.ProcessGroup] = None,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        model.block0.bias_module0 = FSDP(
            model.block0.bias_module0,
            process_group=group,
            **fsdp_kwargs,
        )
        model.block0 = FSDP(model.block0, process_group=group, **fsdp_kwargs)
        return model

    @staticmethod
    def wrap_with_unmanaged_params(
        model,
        add_to_fsdp_module: bool,
        group=None,
    ) -> Tuple[torch.nn.Module, List[torch.nn.Parameter]]:
        """Registers unmanaged parameters before wrapping with :meth:`wrap`."""
        device = next(model.parameters()).device
        unmanaged_param = torch.nn.Parameter(torch.randn(5, 5, device=device))
        # Either register the parameter to a module to be wrapped with FSDP
        # (`model.block2[2]`) or a module not to be wrapped with FSDP (`model`)
        register_module = model.block2[2] if add_to_fsdp_module else model
        register_module.register_parameter(
            "unmanaged_param",
            unmanaged_param,
        )
        # For simplicity, we only add a single unmanaged parameter, but should
        # be easy to generalize if needed
        return NestedModel.wrap(model, group), [unmanaged_param]

    @staticmethod
    def add_unmanaged_param_entry(osd, unmanaged_param, step) -> None:
        """Adds an entry for the unmanaged parameter ``unmanaged_param``
        assuming Adam optimizer and a single parameter group."""
        # The unmanaged parameters should be passed to this method in
        # `model.parameters()` order since their parameter IDs will be assigned
        # in order of the skipped IDs
        # Assign a parameter ID to the unmanaged parameter
        unmanaged_param_id = -1
        param_ids = osd["param_groups"][0]["params"]
        for i in range(1, len(param_ids)):
            diff = param_ids[i] - param_ids[i - 1]
            if diff != 1:
                assert diff > 1, f"Invalid IDs: {param_ids[i - 1]} {param_ids[i]}"
                unmanaged_param_id = param_ids[i - 1] + 1
                break
        if unmanaged_param_id == -1:
            unmanaged_param_id = len(param_ids)  # last ID skipped
        assert unmanaged_param_id >= 0, "One parameter ID should be skipped"
        # Add a state entry for the unmanaged parameter
        state_device = next(iter(next(iter(osd["state"].values())).values())).device
        osd["state"][unmanaged_param_id] = {
            "step": torch.tensor(float(step), device=state_device),
            "exp_avg": torch.randn(unmanaged_param.shape, device=state_device),
            "exp_avg_sq": torch.randn(unmanaged_param.shape, device=state_device),
        }
        # Insert the ID into the parameter group in order
        bisect.insort(osd["param_groups"][0]["params"], unmanaged_param_id)

    # NOTE: We exclude `self.bias` from either parameter group to test the
    # case where the optimizer input does not include all model parameters
    def param_group0(self) -> List[torch.nn.Parameter]:
        # Use `block1`'s parameters for the first parameter group to deviate
        # from the `model.parameters()` order
        return list(self.block1.parameters())

    def param_group1(self) -> List[torch.nn.Parameter]:
        # Deviate from the `model.parameters()` order further by rearranging
        # `block2`'s parameters to be before `block0`'s parameters
        return list(self.block2.parameters()) + list(self.block0.parameters())


class TestFSDPOptimState(FSDPTest):
    def __init__(self, *args, **kwargs):
        super(TestFSDPOptimState, self).__init__(*args, **kwargs)
        self._model_class = {
            _ModelClass.NESTED: self._init_nested_model,
            _ModelClass.TRANSFORMER: self._init_transformer_model,
        }

    def _init_nested_model(
        self,
        wrap: bool,
        wrap_alt: bool = False,  # ignored if `wrap=False`
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        use_diff_optim_inputs: bool = False,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        model = NestedModel().to(device)
        if wrap:
            model = (
                NestedModel.wrap_alt(model, group, fsdp_kwargs)
                if wrap_alt
                else NestedModel.wrap(model, group, fsdp_kwargs=fsdp_kwargs)
            )
        if not use_multiple_param_groups:
            optim_input = list(model.parameters())
        else:
            optim_input = [
                {"params": model.param_group0()},
                {"params": model.param_group1(), "weight_decay": 0.9},
            ]
        # Use a reversed parameter order for the optimizer input on odd ranks
        if use_diff_optim_inputs and self.rank % 2 == 1:
            if isinstance(optim_input[0], dict):
                for param_group in optim_input:
                    param_group["params"] = list(reversed(param_group["params"]))
            else:
                optim_input = list(reversed(optim_input))
        optim = optim_class(optim_input, lr=0.01)
        return model, optim, optim_input

    def _init_transformer_model(
        self,
        wrap: bool,
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        use_diff_optim_inputs: bool = False,
    ):
        if use_multiple_param_groups or use_diff_optim_inputs:
            # Keep these as arguments for parity with `_init_nested_model()`;
            # these settings are not implemented since the transformer is
            # wrapped with FSDP at the top-level, which means that there is
            # only a single flattened parameter, making these booleans vacuous
            raise NotImplementedError()
        if group is None:
            group = dist.distributed_c10d._get_default_group()
        model = TransformerWithSharedParams.init(
            group,
            FSDPInitMode.RECURSIVE if wrap else FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        optim = optim_class(model.parameters(), lr=0.01)
        return model, optim, None

    def _step_model(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device = torch.device("cuda"),
        num_iters: int = 1,
    ) -> List[float]:
        """Performs a forward pass, backward pass, and optimizer step
        ``num_iters``-many times, and returns the per-iteration losses."""
        torch.manual_seed(0)  # set seed for determinism
        losses = []
        module = model.module if hasattr(model, "module") else model
        for _ in range(num_iters):
            optim.zero_grad()
            inp = module.get_input(device)
            output = model(*inp)
            loss = module.get_loss(inp, output).to(device)
            losses.append(loss.item())
            module.run_backward(loss)
            optim.step()
        return losses

    def _broadcast_full_osd(self, full_osd: Dict[str, Any], group=None):
        """Broadcasts the full optimizer state dict in place of using
        ``torch.save()`` and ``torch.load()`` so that all ranks can have it."""
        obj_list = [full_osd]
        dist.broadcast_object_list(
            obj_list,
            src=0,
            group=group,
        )
        full_osd = obj_list[0]
        return full_osd

    def _are_equal_states(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> bool:
        """Checks if ``state1`` and ``state2`` contain the same mappings."""
        if set(state1.keys()) != set(state2.keys()):
            return False
        for state_name, value1 in state1.items():
            value2 = state2[state_name]
            if type(value1) != type(value2):
                return False
            if torch.is_tensor(value1):  # tensor state
                assert torch.is_tensor(value2)
                # Check the values on CPU to be device-agnostic
                value1 = value1.cpu()
                value2 = value2.cpu()
                if value1.shape != value2.shape or not torch.all(
                    torch.isclose(value1, value2)
                ):
                    return False
            else:  # non-tensor state
                if value1 != value2:
                    return False
        return True

    def _check_same_state(
        self,
        fsdp_osd,
        ref_osd,
        check_same_param_keys: bool,
    ):
        """Checks that ``full_osd`` and ``ref_osd`` have the same "state" part.
        If ``check_same_param_keys=True``, then checks that the parameter keys
        match (e.g. when both should be parameter names), and does not check
        the parameter keys otherwise."""
        assert "state" in ref_osd
        self.assertTrue("state" in fsdp_osd)
        ref_osd_state = ref_osd["state"]
        fsdp_osd_state = {
            k: _gather_state_dict(v) for k, v in fsdp_osd["state"].items()
        }

        if check_same_param_keys:
            # Check parameter keys are the same first for earlier erroring
            ref_osd_param_ids = set(ref_osd_state.keys())
            fsdp_osd_param_ids = set(fsdp_osd_state.keys())
            self.assertTrue(ref_osd_param_ids == fsdp_osd_param_ids)
            # Check state values are the same
            for param_id, param_state in fsdp_osd_state.items():
                for state_name, value in param_state.items():
                    ref_value = ref_osd_state[param_id][state_name]
                    self.assertEqual(value, ref_value)
            return
        # Otherwise, only require the parameter keys to be isomorphic (e.g.
        # between IDs and names)
        ref_osd_states = list(ref_osd_state.values())
        fsdp_osd_states = list(fsdp_osd_state.values())
        self.assertEqual(len(ref_osd_states), len(fsdp_osd_states))
        # Use brute-force quadratic-time comparison since it is hard to
        # hash a tensor by value instead of by object
        for fsdp_osd_state in fsdp_osd_states:
            # Check for at least one match (may be > 1 in toy edge cases, e.g.
            # multiple biases); nonetheless, each having >= 1 match and the two
            # lists having equal length imply that the list contents are equal
            self.assertTrue(
                any(
                    self._are_equal_states(fsdp_osd_state, ref_osd_state)
                    for ref_osd_state in ref_osd_states
                )
            )

    def _check_same_param_groups(
        self,
        full_osd,
        ref_osd,
        check_same_param_keys: bool,
    ):
        """Checks that ``full_osd`` and ``ref_osd`` have the same
        "param_groups" part. If ``check_same_param_keys=True`, then checks that
        the parameter keys match (e.g. when both should be parameter names),
        and does not check the parameter keys otherwise."""
        assert "param_groups" in ref_osd
        self.assertTrue("param_groups" in full_osd)
        ref_osd_param_groups = ref_osd["param_groups"]
        full_osd_param_groups = full_osd["param_groups"]
        self.assertTrue(len(full_osd_param_groups), len(ref_osd_param_groups))
        for full_osd_pg, ref_osd_pg in zip(
            full_osd_param_groups,
            ref_osd_param_groups,
        ):
            self.assertEqual(
                set(full_osd_pg.keys()),
                set(ref_osd_pg.keys()),
            )
            for name, full_osd_value in full_osd_pg.items():
                if name == "params" and not check_same_param_keys:
                    continue
                self.assertEqual(full_osd_value, ref_osd_pg[name])

    def _check_state_device(self, osd: Dict[str, Any], on_gpu: bool):
        """Checks that all tensors in ``osd["state"]`` are on GPU if
        ``on_gpu=True`` and on CPU if ``on_gpu=False``."""
        for param_state in osd["state"].values():
            for value in param_state.values():
                if torch.is_tensor(value) and value.dim() > 0:
                    if on_gpu:
                        self.assertTrue(value.is_cuda)
                    else:
                        self.assertFalse(value.is_cuda)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", STATE_DICT_TYPE)
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("rank0_only", [False, True])
    @parametrize("use_diff_optim_inputs", [False, True])
    def test_optim_state_dict_nested(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
        rank0_only: bool,
        use_diff_optim_inputs: bool,
    ) -> None:
        """
        Tests :meth:`full_optim_state_dict` and `sharded_optim_state_dict`
        by comparing the returned dict for an FSDP-wrapped model with that of
        an equivalent non-wrapped model.

        The test checks the equivalence excluding the parameter keys since the
        FSDP and normal optimizer state dicts key by names and IDs,
        respectively. This means that the test can pass even if parameter keys
        are incorrectly mapped to values. Their correct mapping is tested in
        other tests that exercise the save/load workflow.
        """
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_optim_state_dict_nested,
            state_dict_type=state_dict_type,
            use_multiple_param_groups=use_multiple_param_groups,
            rank0_only=rank0_only,
            use_diff_optim_inputs=use_diff_optim_inputs,
        )

    def _test_optim_state_dict_nested(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
        rank0_only: bool,
        use_diff_optim_inputs: bool,
        use_optim_input: bool,
    ) -> None:
        if rank0_only and state_dict_type == StateDictType.SHARDED_STATE_DICT:
            return  # not supported
        NUM_ITERS = 3
        model1, optim1, optim_input = self._init_nested_model(
            wrap=True,
            use_multiple_param_groups=use_multiple_param_groups,
            use_diff_optim_inputs=use_diff_optim_inputs,
        )
        losses1 = self._step_model(model1, optim1, num_iters=NUM_ITERS)
        if state_dict_type == StateDictType.FULL_STATE_DICT:
            if use_optim_input:
                fsdp_osd = FSDP.full_optim_state_dict(
                    model1,
                    optim1,
                    optim_input,
                    rank0_only=rank0_only,
                )
            else:
                fsdp_osd = FSDP.full_optim_state_dict(
                    model1,
                    optim1,
                    rank0_only=rank0_only,
                )
        else:
            if use_optim_input:
                fsdp_osd = FSDP.sharded_optim_state_dict(model1, optim1, optim_input)
            else:
                fsdp_osd = FSDP.sharded_optim_state_dict(model1, optim1)
        # Non-target ranks get an empty state dict
        if rank0_only and self.rank != 0:
            self.assertEqual(len(fsdp_osd), 0)
            return
        model2, optim2, _ = self._init_nested_model(
            wrap=False,
            use_multiple_param_groups=use_multiple_param_groups,
            use_diff_optim_inputs=use_diff_optim_inputs,
        )
        losses2 = self._step_model(model2, optim2, num_iters=NUM_ITERS)
        ref_osd = optim2.state_dict()
        # Check the losses to eliminate model drift as a source of error
        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert l1 == l2, f"Losses differ on iter {i}: {l1:.5f} {l2:.5f}"
        # Do not check the parameter keys since the full/sharded optimizer state
        # dict uses parameter names, while the non-wrapped equivalent uses
        # parameter IDs
        check_same_param_keys = False
        self._check_same_param_groups(
            fsdp_osd,
            ref_osd,
            check_same_param_keys=check_same_param_keys,
        )
        self._check_same_state(
            fsdp_osd,
            ref_osd,
            check_same_param_keys=check_same_param_keys,
        )

    @skip_if_lt_x_gpu(2)
    def test_full_optim_state_dict_keys(self):
        """Tests that the parameter keys returned by
        :meth:`full_optim_state_dict` match those of :meth:`state_dict` with
        full ``state_dict_type`` for a non-FSDP-root model with nested FSDP
        instances and ignored modules."""
        device = torch.device("cuda")
        model = NestedModel().to(device)
        wrapped_model = NestedModel.wrap(model, ignore_modules=True)
        # Add checkpointing to ensure optim_state_dict and state_dict strip out
        # checkpointing prefixes.
        apply_activation_checkpointing(
            model, check_fn=lambda module: isinstance(module, torch.nn.Sequential)
        )
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._step_model(model, optim, device)
        optim_state_dict = FSDP.full_optim_state_dict(
            wrapped_model, optim, rank0_only=False
        )
        with FSDP.state_dict_type(wrapped_model, StateDictType.FULL_STATE_DICT):
            state_dict = wrapped_model.state_dict()
        self.assertEqual(optim_state_dict["state"].keys(), state_dict.keys())
        # Check that checkpointing prefix was indeed stripped.
        for key in optim_state_dict["state"]:
            self.assertNotIn(_CHECKPOINT_WRAPPED_MODULE, key)

    @skip_if_lt_x_gpu(2)
    def test_full_optim_state_dict_nested_invalid(self):
        """Tests that :meth:`full_optim_state_dict` raises an error when
        nonzero ranks are missing the optimizer state for parameters on rank
        0."""
        device = torch.device("cuda")
        model = NestedModel.wrap(NestedModel().to(device), None)
        optim_input = list(model.parameters())
        if self.rank != 0:
            # Exclude a parameter so that nonzero ranks are missing state
            optim_input = optim_input[:-1]
        optim = torch.optim.Adam(optim_input, lr=1e-3)
        self._step_model(model, optim, num_iters=3)
        error_regex = (
            "FSDP currently requires each rank to have at least the "
            "optimizer states needed by rank 0's optimizer but some ranks "
            "are missing some of those states"
        )
        with self.assertRaisesRegex(RuntimeError, error_regex):
            FSDP.full_optim_state_dict(model, optim)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("wrap_alt", [False, True])
    @parametrize("use_diff_optim_inputs", [False, True])
    def test_shard_full_optim_state_dict_nested(
        self,
        use_multiple_param_groups: bool,
        wrap_alt: bool,
        use_diff_optim_inputs: bool,
    ):
        """Tests :meth:`shard_full_optim_state_dict` for a non-FSDP-root model
        with nested FSDP instances."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.NESTED,
            use_multiple_param_groups=use_multiple_param_groups,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.BROADCAST_OBJECT_LIST,
            use_diff_optim_inputs=use_diff_optim_inputs,
            wrap_alt=wrap_alt,
        )

    @skip_if_lt_x_gpu(2)
    def test_shard_full_optim_state_dict_nested_halve_world_size(self):
        """Tests :meth:`shard_full_optim_state_dict` for a non-FSDP-root model
        with nested FSDP instances when loading into a new process group with
        halved world size."""
        # To save CI costs, we test with the "harder" settings:
        use_multiple_param_groups = True
        use_diff_optim_inputs = True
        wrap_alt = True
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.NESTED,
            use_multiple_param_groups=use_multiple_param_groups,
            halve_world_size=True,
            osd_comm_method=_OSDCommMethod.BROADCAST_OBJECT_LIST,
            use_diff_optim_inputs=use_diff_optim_inputs,
            wrap_alt=wrap_alt,
        )

    @skip_if_lt_x_gpu(2)
    def test_shard_full_optim_state_dict_transformer(self) -> None:
        """Tests :meth:`shard_full_optim_state_dict` for an FSDP-root
        transformer model with shared parameters."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.TRANSFORMER,
            use_multiple_param_groups=False,
            halve_world_size=True,
            osd_comm_method=_OSDCommMethod.BROADCAST_OBJECT_LIST,
            use_diff_optim_inputs=False,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("wrap_alt", [False, True])
    @parametrize("use_diff_optim_inputs", [False, True])
    def test_scatter_full_optim_state_dict_nested(
        self,
        use_multiple_param_groups: bool,
        wrap_alt: bool,
        use_diff_optim_inputs: bool,
    ):
        """Tests :meth:`scatter_full_optim_state_dict` for a non-FSDP-root
        model with nested FSDP instances."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.NESTED,
            use_multiple_param_groups=use_multiple_param_groups,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.SCATTER_FULL_OSD,
            use_diff_optim_inputs=use_diff_optim_inputs,
            wrap_alt=wrap_alt,
        )

    @skip_if_lt_x_gpu(2)
    def test_scatter_full_optim_state_dict_nested_halve_world_size(self):
        """Tests :meth:`scatter_full_optim_state_dict` for a non-FSDP-root
        model with nested FSDP instances when loading into a new process group
        with halved world size."""
        # To save CI costs, we test with the "harder" settings:
        use_multiple_param_groups = True
        use_diff_optim_inputs = True
        wrap_alt = True
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.NESTED,
            use_multiple_param_groups=use_multiple_param_groups,
            halve_world_size=True,
            osd_comm_method=_OSDCommMethod.SCATTER_FULL_OSD,
            use_diff_optim_inputs=use_diff_optim_inputs,
            wrap_alt=wrap_alt,
        )

    @skip_if_lt_x_gpu(2)
    def test_scatter_full_optim_state_dict_transformer(self) -> None:
        """Tests :meth:`scatter_full_optim_state_dict` for an FSDP-root
        transformer model with shared parameters."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.TRANSFORMER,
            use_multiple_param_groups=False,
            halve_world_size=True,
            osd_comm_method=_OSDCommMethod.SCATTER_FULL_OSD,
            use_diff_optim_inputs=False,
        )

    @skip_if_lt_x_gpu(2)
    def test_flatten_sharded_optim_state_dict_nested(self):
        """Tests :meth:`flatten_sharded_optim_state_dict` for an FSDP-root
        nested model."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            _ModelClass.NESTED,
            use_multiple_param_groups=False,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.FLATTEN_SHARDED_OSD,
            use_diff_optim_inputs=False,
            wrap_alt=True,
        )

    @skip_if_lt_x_gpu(2)
    def test_flatten_sharded_optim_state_dict_transformer(self) -> None:
        """Tests :meth:`flatten_sharded_optim_state_dict` for an FSDP-root
        transformer model."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            _ModelClass.TRANSFORMER,
            use_multiple_param_groups=False,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.FLATTEN_SHARDED_OSD,
            use_diff_optim_inputs=False,
        )

    def _test_load_optim_state(
        self,
        model_class: _ModelClass,
        use_multiple_param_groups: bool,
        halve_world_size: bool,
        osd_comm_method: _OSDCommMethod,
        use_diff_optim_inputs: bool,
        use_optim_input: bool,
        **new_model_kwargs,
    ):
        """
        (1) Runs a model with full world size for K iterations to generate a
        full/sharded optimizer state dict;
        (2) initializes a model with halved world size and possibly different
        FSDP wrapping scheme (based on ``new_model_kwargs``);
        (3) loads the full/sharded optimizer state dict from (1) according to the
        halved-world-size model;
        (4) runs the halved-world-size model for K iterations; and
        (5) checks that the sharded optimizer state dict from (3) matches the
        halved-world-size model's local optimizer state dict, meaning that the
        former could have equivalently been loaded into the local optimizer.
        """
        NUM_ITERS = 3
        initializer = self._model_class[model_class]
        osd_method = (
            FSDP.sharded_optim_state_dict
            if osd_comm_method == _OSDCommMethod.FLATTEN_SHARDED_OSD
            else FSDP.full_optim_state_dict
        )

        # First, run a wrapped model with full world size for a few iterations
        model1, optim1, optim_input1 = initializer(
            wrap=True,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model1, optim1, num_iters=NUM_ITERS)
        fsdp_osd1 = (
            osd_method(model1, optim1, optim_input1)
            if use_optim_input
            else osd_method(model1, optim1)
        )
        if halve_world_size:
            # Create a new process group with halved world size
            new_group_ranks = [r for r in range(self.world_size) if r % 2 == 0]
            new_group = dist.new_group(ranks=new_group_ranks)
            if self.rank not in new_group_ranks:
                return
        else:
            # Continue using the same group and hence world size
            new_group = dist.distributed_c10d._get_default_group()
        # Second, run a wrapped model with (possibly) halved world size and
        # (possibly) differing `optim_input` across ranks
        model2, optim2, optim_input2 = initializer(
            wrap=True,
            group=new_group,
            use_multiple_param_groups=use_multiple_param_groups,
            use_diff_optim_inputs=use_diff_optim_inputs,
            **new_model_kwargs,  # specify `wrap_alt` to change wrapping
        )
        self._step_model(model2, optim2, num_iters=NUM_ITERS)
        fsdp_osd2 = (
            osd_method(model2, optim2, optim_input2, group=new_group)
            if use_optim_input
            else osd_method(model2, optim2, group=new_group)
        )
        # Compute two sharded optim state dicts: (1) for the first model
        # according to the second model and (2) for the second model according
        # to the second model
        if osd_comm_method == _OSDCommMethod.BROADCAST_OBJECT_LIST:
            fsdp_osd1 = self._broadcast_full_osd(fsdp_osd1, group=new_group)
            sharded_osd1 = (
                FSDP.shard_full_optim_state_dict(
                    fsdp_osd1, model2, optim_input=optim_input2
                )
                if use_optim_input
                else FSDP.shard_full_optim_state_dict(fsdp_osd1, model2, optim=optim2)
            )
            fsdp_osd2 = self._broadcast_full_osd(fsdp_osd2, group=new_group)
            sharded_osd2 = (
                FSDP.shard_full_optim_state_dict(
                    fsdp_osd2, model2, optim_input=optim_input2
                )
                if use_optim_input
                else FSDP.shard_full_optim_state_dict(fsdp_osd2, model2, optim=optim2)
            )
        elif osd_comm_method == _OSDCommMethod.SCATTER_FULL_OSD:
            sharded_osd1 = (
                FSDP.scatter_full_optim_state_dict(
                    fsdp_osd1 if self.rank == 0 else None,
                    model2,
                    optim_input=optim_input2,
                    group=new_group,
                )
                if use_optim_input
                else FSDP.scatter_full_optim_state_dict(
                    fsdp_osd1 if self.rank == 0 else None,
                    model2,
                    optim=optim2,
                    group=new_group,
                )
            )
            sharded_osd2 = (
                FSDP.scatter_full_optim_state_dict(
                    fsdp_osd2 if self.rank == 0 else None,
                    model2,
                    optim_input=optim_input2,
                    group=new_group,
                )
                if use_optim_input
                else FSDP.scatter_full_optim_state_dict(
                    fsdp_osd2 if self.rank == 0 else None,
                    model2,
                    optim=optim2,
                    group=new_group,
                )
            )
            self._check_state_device(sharded_osd1, on_gpu=True)
            self._check_state_device(sharded_osd2, on_gpu=True)
        elif osd_comm_method == _OSDCommMethod.FLATTEN_SHARDED_OSD:
            sharded_osd1 = (
                FSDP.flatten_sharded_optim_state_dict(
                    fsdp_osd1,
                    model2,
                    optim_input=optim_input2,
                )
                if use_optim_input
                else FSDP.flatten_sharded_optim_state_dict(
                    fsdp_osd1,
                    model2,
                    optim=optim2,
                )
            )
            sharded_osd2 = (
                FSDP.flatten_sharded_optim_state_dict(
                    fsdp_osd2,
                    model2,
                    optim_input=optim_input2,
                )
                if use_optim_input
                else FSDP.flatten_sharded_optim_state_dict(
                    fsdp_osd2,
                    model2,
                    optim=optim2,
                )
            )

        # As a sanity check, check that sharding the second model's full/sharded
        # optimizer state dict according to itself is equivalent to its local
        # optimizer's state dict
        local_osd2 = optim2.state_dict()
        check_same_param_keys = True  # should all have matching parameter IDs
        self._check_same_param_groups(
            sharded_osd2,
            local_osd2,
            check_same_param_keys=check_same_param_keys,
        )
        self._check_same_state(
            sharded_osd2,
            local_osd2,
            check_same_param_keys=check_same_param_keys,
        )
        # Check that sharding the first model's full/sharded optimizer state dict
        # according to the second model is equivalent to the second model's
        # local optimizer state dict
        self._check_same_param_groups(
            sharded_osd1,
            local_osd2,
            check_same_param_keys=check_same_param_keys,
        )
        self._check_same_state(
            sharded_osd1,
            local_osd2,
            check_same_param_keys=check_same_param_keys,
        )
        # As a sanity check, check that we can load and run a few iterations
        if osd_comm_method != _OSDCommMethod.FLATTEN_SHARDED_OSD:
            optim2.load_state_dict(sharded_osd1)
            self._step_model(model2, optim2, num_iters=NUM_ITERS)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", STATE_DICT_TYPE)
    @parametrize("add_to_fsdp_module", [False, True])
    def test_shard_full_optim_state_dict_unmanaged_params(
        self,
        state_dict_type: StateDictType,
        add_to_fsdp_module: bool,
    ):
        """
        Tests :meth:`shard_full_optim_state_dict` when there are unmanaged
        parameters.
          - If ``add_to_fsdp_module=True``, then the unmanaged parameters are
          added to a module to be wrapped with FSDP, in which case there should
          be an error since we require that all unflattened parameter
          comprising a flattened parameter have the same scalar state (e.g.
          Adam "step") but the added parameter is missing its entry.
          - If ``add_to_fsdp_module=False``, then the unmanaged parameters are
          added to a module not to be wrapped with FSDP, in which case there
          should be no error (emulating model parallel use cases where some
          parameters may be managed externally to FSDP).
        We do not separately test unmanaged parameters for
        :meth:`scatter_full_optim_state_dict` and `flatten_sharded_optim_state_dict`
        to save CI cost since it call into the same subroutine
        :meth:`_flatten_optim_state_dict`.
        """
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_shard_full_optim_state_dict_unmanaged_params,
            state_dict_type=state_dict_type,
            add_to_fsdp_module=add_to_fsdp_module,
        )

    def _test_shard_full_optim_state_dict_unmanaged_params(
        self,
        state_dict_type: StateDictType,
        add_to_fsdp_module: bool,
        use_optim_input: bool,
    ):
        NUM_ITERS = 1
        # Create a normal wrapped model
        model, optim, optim_input = self._init_nested_model(wrap=True)
        self._step_model(model, optim, num_iters=NUM_ITERS)

        if state_dict_type == StateDictType.FULL_STATE_DICT:
            fsdp_osd = (
                FSDP.full_optim_state_dict(model, optim, optim_input, rank0_only=False)
                if use_optim_input
                else FSDP.full_optim_state_dict(model, optim, rank0_only=False)
            )  # save on all ranks to avoid having to broadcast from rank 0
        else:
            fsdp_osd = (
                FSDP.sharded_optim_state_dict(model, optim, optim_input)
                if use_optim_input
                else FSDP.sharded_optim_state_dict(model, optim)
            )
        # Create a new model with the same structure but additional unmanaged
        # parameters, representing the model for which we want to load
        device = torch.device("cuda")
        model = NestedModel().to(device)
        model, unmanaged_params = NestedModel.wrap_with_unmanaged_params(
            model,
            add_to_fsdp_module,
        )
        optim_input = list(model.parameters())
        optim = torch.optim.Adam(optim_input, lr=1e-3)
        if add_to_fsdp_module:
            # If we add the unmanaged parameters to a module wrapped with FSDP,
            # then the flattened parameter will be comprised of some
            # unflattened parameters with zero-dimensional tensor state (i.e.
            # Adam "step") and others without (i.e. the unmanaged parameters),
            # which triggers an error that we have to ensure correctness
            error_prefix = (
                "^(All unflattened parameters comprising a "
                "single flattened parameter must have scalar state with the "
                "same value and dtype)"
            )
            with self.assertRaisesRegex(ValueError, error_prefix):
                if state_dict_type == StateDictType.FULL_STATE_DICT:
                    (
                        FSDP.shard_full_optim_state_dict(
                            fsdp_osd, model, optim_input=optim_input
                        )
                        if use_optim_input
                        else FSDP.shard_full_optim_state_dict(
                            fsdp_osd, model, optim=optim
                        )
                    )
                else:
                    (
                        FSDP.flatten_sharded_optim_state_dict(
                            fsdp_osd, model, optim_input=optim_input
                        )
                        if use_optim_input
                        else FSDP.flatten_sharded_optim_state_dict(
                            fsdp_osd, model, optim=optim
                        )
                    )
        else:
            # If we add the unmanaged parameters to a module not wrapped with
            # FSDP, then we simply ignore them without erroring to enable
            # model parallelism use cases, where some parameters are managed
            # externally to FSDP
            if state_dict_type == StateDictType.FULL_STATE_DICT:
                flattened_osd = (
                    FSDP.shard_full_optim_state_dict(
                        fsdp_osd, model, optim_input=optim_input
                    )
                    if use_optim_input
                    else FSDP.shard_full_optim_state_dict(fsdp_osd, model, optim=optim)
                )
            else:
                flattened_osd = (
                    FSDP.flatten_sharded_optim_state_dict(
                        fsdp_osd, model, optim_input=optim_input
                    )
                    if use_optim_input
                    else FSDP.flatten_sharded_optim_state_dict(
                        fsdp_osd, model, optim=optim
                    )
                )
            # Add entries for the unmanaged parameters to be able to load
            for unmanaged_param in unmanaged_params:
                NestedModel.add_unmanaged_param_entry(
                    flattened_osd,
                    unmanaged_param,
                    NUM_ITERS,
                )
            # Check that we can load the optimizer state dict
            optim.load_state_dict(flattened_osd)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", STATE_DICT_TYPE)
    @parametrize("use_multiple_param_groups", [False, True])
    def test_rekey_optim_state_dict_to_ids(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
    ):
        """Tests :meth:`rekey_optim_state_dict` with the new keys being
        parameter IDs by checking that a wrapped model (i.e. with FSDP modules)
        can rekey its optimizer state dict to match that of an equivalent
        non-wrapped model (i.e. without FSDP modules)."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_rekey_optim_state_dict_to_ids,
            state_dict_type=state_dict_type,
            use_multiple_param_groups=use_multiple_param_groups,
        )

    @skip_if_lt_x_gpu(2)
    def _test_rekey_optim_state_dict_to_ids(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
        use_optim_input: bool,
    ):
        NUM_ITERS = 3
        # Run a wrapped model for a few iterations
        model1, optim1, optim_input1 = self._init_nested_model(
            wrap=True,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model1, optim1, num_iters=NUM_ITERS)
        if state_dict_type == StateDictType.FULL_STATE_DICT:
            fsdp_osd = (
                FSDP.full_optim_state_dict(model1, optim1, optim_input1)
                if use_optim_input
                else FSDP.full_optim_state_dict(model1, optim1)
            )
            # Broadcast instead of `torch.save()`/`torch.load()` so that all ranks
            # have the full state dict
            fsdp_osd = self._broadcast_full_osd(fsdp_osd)
        else:
            fsdp_osd = (
                FSDP.sharded_optim_state_dict(model1, optim1, optim_input1)
                if use_optim_input
                else FSDP.sharded_optim_state_dict(model1, optim1)
            )
        # Run a non-wrapped model for a few iterations
        model2, optim2, optim_input2 = self._init_nested_model(
            wrap=False,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model2, optim2, num_iters=NUM_ITERS)
        # Re-key the wrapped model's optimizer state dict using parameter IDs
        # according to the non-wrapped model
        rekeyed_osd = (
            FSDP.rekey_optim_state_dict(
                fsdp_osd,
                OptimStateKeyType.PARAM_ID,
                model2,
                optim_input=optim_input2,
            )
            if use_optim_input
            else FSDP.rekey_optim_state_dict(
                fsdp_osd,
                OptimStateKeyType.PARAM_ID,
                model2,
                optim=optim2,
            )
        )
        # Check that the re-keyed dict and actual dict are the same
        osd = optim2.state_dict()
        check_same_param_keys = True
        self._check_same_param_groups(
            rekeyed_osd,
            osd,
            check_same_param_keys=check_same_param_keys,
        )
        self._check_same_state(
            rekeyed_osd,
            osd,
            check_same_param_keys=check_same_param_keys,
        )
        # As a sanity check, check that we can load and run a few iterations
        if state_dict_type != StateDictType.SHARDED_STATE_DICT:
            optim2.load_state_dict(rekeyed_osd)
            self._step_model(model2, optim2, num_iters=NUM_ITERS)

    @skip_if_lt_x_gpu(2)
    def test_rekey_optim_state_dict_to_names(self):
        """Tests :meth:`rekey_optim_state_dict` with the new keys being
        parameter names by checking that a non-wrapped model (i.e. without FSDP
        modules) can rekey its optimizer state dict to match the expected
        output of :meth:`full_optim_state_dict`, hence be sharded using
        :meth:`shard_full_optim_state_dict`, and finally match the per-rank
        optimizer state dict of a wrapped model (i.e. with FSDP modules)."""
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_rekey_optim_state_dict_to_names,
            use_multiple_param_groups=False,
        )

    def _test_rekey_optim_state_dict_to_names(
        self,
        use_multiple_param_groups: bool,
        use_optim_input: bool,
    ):

        NUM_ITERS = 3
        # Run a wrapped model for a few iterations
        model1, optim1, optim_input1 = self._init_nested_model(
            wrap=True,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model1, optim1, num_iters=NUM_ITERS)
        # Run a non-wrapped model for a few iterations
        model2, optim2, optim_input2 = self._init_nested_model(
            wrap=False,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model2, optim2, num_iters=NUM_ITERS)
        # Re-key the non-wrapped model's optimizer state dict using parameter
        # names (still according to itself)
        osd2 = optim2.state_dict()
        rekeyed_osd = (
            FSDP.rekey_optim_state_dict(
                osd2,
                OptimStateKeyType.PARAM_NAME,
                model2,
                optim_input=optim_input2,
            )
            if use_optim_input
            else FSDP.rekey_optim_state_dict(
                osd2,
                OptimStateKeyType.PARAM_NAME,
                model2,
                optim=optim2,
            )
        )
        # Shard the non-wrapped model's re-keyed optimizer state dict, which
        # maps back to (flattened) parameter IDs
        sharded_osd = (
            FSDP.shard_full_optim_state_dict(
                rekeyed_osd,
                model1,
                optim_input=optim_input1,
            )
            if use_optim_input
            else FSDP.shard_full_optim_state_dict(
                rekeyed_osd,
                model1,
                optim=optim1,
            )
        )
        # Check that this sharded optimizer state dict matches the wrapped
        # model's per-rank optimizer state dict
        osd1 = optim1.state_dict()
        check_same_param_keys = True
        self._check_same_param_groups(
            sharded_osd,
            osd1,
            check_same_param_keys=check_same_param_keys,
        )
        self._check_same_state(
            sharded_osd,
            osd1,
            check_same_param_keys=check_same_param_keys,
        )
        # As a sanity check, check that we can load and run a few iterations
        optim1.load_state_dict(sharded_osd)
        self._step_model(model1, optim1, num_iters=NUM_ITERS)

    @skip_if_lt_x_gpu(2)
    def test_optim_input_warning(self):
        """Tests that passing the ``optim_input`` argument into optimizer state
        checkpointing APIs issues a warning."""

        def should_check_method(method_name: str):
            # Check every method since they all accept `optim_input`
            return True

        def get_warning_context():
            warning_regex = "`optim_input` argument is deprecated"
            return self.assertWarnsRegex(
                expected_warning=UserWarning, expected_regex=warning_regex
            )

        self._run_on_all_optim_state_apis(
            should_check_method, get_warning_context, fsdp_kwargs=None
        )

    @skip_if_lt_x_gpu(2)
    def test_use_orig_params_error(self):
        """Tests that the optimizer state checkpointing APIs raise an error
        when ``use_orig_params=True``."""

        def should_check_method(method_name: str):
            # Skip `rekey_optim_state_dict` since that does not depend on
            # `use_orig_params=True`
            return method_name != "rekey_optim_state_dict"

        def get_error_context():
            error_regex = "Optimizer state checkpointing is not supported yet for `use_orig_params=True`"
            return self.assertRaisesRegex(
                expected_exception=NotImplementedError, expected_regex=error_regex
            )

        fsdp_kwargs = {"use_orig_params": True}
        self._run_on_all_optim_state_apis(
            should_check_method, get_error_context, fsdp_kwargs
        )

    def _run_on_all_optim_state_apis(
        self,
        should_check_method_fn: Callable[[str], bool],
        context_fn: Callable,
        fsdp_kwargs: Optional[Dict[str, Any]],
    ):
        """
        Runs through all optimizer state checkpointing APIs with a context
        manager instantiated by ``context_fn``. Certain APIs can be skipped
        via ``should_check_method_fn``, which gets passed the string name of
        the method.
        """
        wrapped_model, wrapped_optim, wrapped_optim_input = self._init_nested_model(
            wrap=True,
            use_multiple_param_groups=False,
            fsdp_kwargs=fsdp_kwargs,
        )
        self._step_model(wrapped_model, wrapped_optim, num_iters=2)

        # Sharded optim state dict
        if should_check_method_fn("sharded_optim_state_dict"):
            with context_fn():
                fsdp_osd = FSDP.sharded_optim_state_dict(
                    wrapped_model,
                    wrapped_optim,
                    optim_input=wrapped_optim_input,
                )
        if "fsdp_osd" not in locals():
            fsdp_osd = {}  # may not be defined due to previous method erroring
        if should_check_method_fn("flatten_sharded_optim_state_dict"):
            with context_fn():
                FSDP.flatten_sharded_optim_state_dict(
                    fsdp_osd,
                    wrapped_model,
                    optim_input=wrapped_optim_input,
                )
        # Full optim state dict
        if should_check_method_fn("full_optim_state_dict"):
            with context_fn():
                fsdp_osd = FSDP.full_optim_state_dict(
                    wrapped_model,
                    wrapped_optim,
                    optim_input=wrapped_optim_input,
                    rank0_only=False,
                )
        if should_check_method_fn("shard_full_optim_state_dict"):
            with context_fn():
                FSDP.shard_full_optim_state_dict(
                    fsdp_osd,
                    wrapped_model,
                    optim_input=wrapped_optim_input,
                )
        if should_check_method_fn("scatter_full_optim_state_dict"):
            with context_fn():
                FSDP.scatter_full_optim_state_dict(
                    fsdp_osd,
                    wrapped_model,
                    optim_input=wrapped_optim_input,
                )
        # Rekey optim state dict
        (
            nonwrapped_model,
            nonwrapped_optim,
            nonwrapped_optim_input,
        ) = self._init_nested_model(wrap=False, use_multiple_param_groups=False)
        if should_check_method_fn("rekey_optim_state_dict"):
            with context_fn():
                rekeyed_osd = FSDP.rekey_optim_state_dict(
                    fsdp_osd,  # from `full_optim_state_dict()`
                    OptimStateKeyType.PARAM_ID,
                    nonwrapped_model,
                    optim_input=nonwrapped_optim_input,
                )
        self._step_model(nonwrapped_model, nonwrapped_optim, num_iters=2)
        osd = nonwrapped_optim.state_dict()
        if should_check_method_fn("rekey_optim_state_dict"):
            with context_fn():
                FSDP.rekey_optim_state_dict(
                    osd,
                    OptimStateKeyType.PARAM_NAME,
                    nonwrapped_model,
                    optim_input=nonwrapped_optim_input,
                )

    @skip_if_lt_x_gpu(2)
    def test_full_osd_without_first_param_state(self):
        """
        Tests saving and loading a full optim state dict for Adam optimizer
        (i.e. any optimizer with a "step" key in its state) when the first
        parameter does not have optimizer state (e.g. unused or frozen).
        """

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(5, 5)
                self.lin2 = nn.Linear(5, 5)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Do not use `lin1`, which is the parameter passed to the
                # optimizer and the one checked for "step" state to see if it
                # is tensor or float
                return self.relu(self.lin2(x))

        model = Model().cuda()
        model.lin1 = FSDP(model.lin1)
        model.lin2 = FSDP(model.lin2)
        fsdp_model = FSDP(model)
        optim = torch.optim.Adam(
            fsdp_model.parameters(), lr=1e-2
        )  # or any optimizer with "step"

        # Run an iteration to construct optimizer state
        device = torch.device("cuda")
        inp = torch.randn((2, 5), device=device)
        loss = fsdp_model(inp).sum()
        loss.backward()
        optim.step()

        # Check that save and load does not error
        full_osd = FSDP.full_optim_state_dict(fsdp_model, optim, rank0_only=False)
        sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, fsdp_model)
        optim.load_state_dict(sharded_osd)
        # `__setstate__()` will check the 0th parameter to see if "step" is
        # represented as a tensor or float, so it is imperative that its state
        # is non-empty.

        # Run an iteration as a sanity check
        inp = torch.randn((2, 5), device=device)
        loss = fsdp_model(inp).sum()
        loss.backward()
        optim.step()


instantiate_parametrized_tests(TestFSDPOptimState)

if __name__ == "__main__":
    run_tests()
