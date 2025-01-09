# Owner(s): ["oncall: distributed"]
import contextlib
import itertools
import math
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp._flat_param import FlatParameter
from torch.distributed.fsdp.fully_sharded_data_parallel import FLAT_PARAM
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    get_devtype,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestUnshardParamsBase(FSDPTest):
    """
    This contains any methods common to both the sharded and non-sharded cases.
    """

    def _test_unshard_params_writeback(
        self,
        writeback: bool,
        check_outer: bool,
        **fsdp_kwargs: Dict[str, Any],
    ):
        model = nn.Sequential(
            nn.Linear(5, 5, bias=False, device=device_type.type),
            nn.Linear(5, 3, bias=False, device=device_type.type),
        )
        model[0] = FSDP(model[0], **fsdp_kwargs)
        model = FSDP(model, **fsdp_kwargs)
        uses_sharded_strategy = model.sharding_strategy != ShardingStrategy.NO_SHARD
        offloading_params = model.cpu_offload.offload_params

        # Assumes depth-first `.parameters()`
        outer_param: Union[FlatParameter, nn.Parameter] = next(model.parameters())
        inner_param: Union[FlatParameter, nn.Parameter] = next(model[0].parameters())
        param_to_check = outer_param if check_outer else inner_param

        # Write a known value to all elements of the *sharded* parameter or
        # `FlatParameter` to check
        with torch.no_grad():
            param_to_check.zero_()
            param_to_check += self.rank + 2
        # Zero the *unsharded* parameters
        with FSDP.summon_full_params(model, writeback=writeback), torch.no_grad():
            for param in model.parameters():
                param.zero_()

        # Check the 0th singleton element of the sharded parameter to see if
        # the zeroing from inside the context persists
        param_elem_to_check = param_to_check[0]
        if param_elem_to_check.numel() > 1:
            # For `use_orig_params=True` and `NO_SHARD`, the parameter
            # preserves the original 2D shape, so we must access one more time
            param_elem_to_check = param_elem_to_check[0]
        if writeback or (not uses_sharded_strategy and not offloading_params):
            # When FSDP does not use a sharded strategy and is not offloading
            # parameters to CPU, it directly exposes the tensor storage that
            # serves as the unsharded source of truth, so the write is always
            # reflected regardless of `writeback`.
            self.assertEqual(param_elem_to_check, 0)
        else:
            self.assertEqual(param_elem_to_check, self.rank + 2)
        if offloading_params:
            cpu_device = torch.device("cpu")
            for param in model.parameters():
                self.assertEqual(param.device, cpu_device)

    def _get_test_unshard_params_writeback_config(self) -> Dict[str, List[Any]]:
        return {
            "writeback": [True, False],
            "check_outer": [True, False],
            "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
            "cpu_offload": [
                CPUOffload(offload_params=False),
                CPUOffload(offload_params=True),
            ],
            "use_orig_params": [True, False],
        }

    def _test_unshard_params_param_data(
        self,
        rank0_only: bool,
        offload_to_cpu: bool,
        cpu_offload: CPUOffload,
        mixed_precision: Optional[MixedPrecision],
        use_orig_params: bool,
    ):
        local_model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs={"device_id": device_type.type},
            deterministic=True,
        )
        # Apply FSDP such that the root module does not have FSDP applied,
        # while there are multiple FSDP root submodules (as proven later)
        fsdp_model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs={
                "cpu_offload": cpu_offload,
                "mixed_precision": mixed_precision,
                "use_orig_params": use_orig_params,
            },
            deterministic=True,
        )
        self.assertFalse(isinstance(fsdp_model, FSDP))
        # Hard code the following names because getting them is non-trivial
        non_fsdp_managed_param_names = {
            "module.0.weight",
            "module.0.bias",
            "module.3.weight",
            "module.3.bias",
        }

        with FSDP.summon_full_params(
            fsdp_model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            if not rank0_only or self.rank == 0:
                for p1, (n2, p2) in zip(
                    local_model.parameters(), fsdp_model.named_parameters()
                ):
                    self.assertEqual(p1.shape, p2.shape)
                    if (
                        offload_to_cpu
                        and clean_tensor_name(n2) not in non_fsdp_managed_param_names
                    ):
                        self.assertEqual(torch.device("cpu"), p2.device)
                    else:
                        self.assertEqual(p1.device, p2.device)
                    self.assertEqual(
                        p1.dtype, p2.dtype
                    )  # even if FSDP uses mixed precision
                    self.assertEqual(p1, p2)
                    self.assertTrue(isinstance(p2, nn.Parameter))
            else:
                # Check that each `FlatParameter` has the sharded size as a
                # proxy for it being resharded
                for handle in traversal_utils._get_fsdp_handles(fsdp_model):
                    if handle.uses_sharded_strategy:
                        self.assertEqual(
                            handle.flat_param.shape, handle.flat_param._sharded_size
                        )
                    else:
                        self.assertEqual(
                            handle.flat_param.shape,
                            handle.flat_param._unpadded_unsharded_size,
                        )

        # Prove the number of FSDP roots after lazy initialization
        num_fsdp_roots = 0
        for fsdp_state in traversal_utils._get_fsdp_states(fsdp_model):
            num_fsdp_roots += fsdp_state._is_root
        self.assertGreater(num_fsdp_roots, 1)

    def _get_test_unshard_params_param_data_config(self) -> Dict[str, List[Any]]:
        return {
            "rank0_only": [False, True],
            "offload_to_cpu": [False, True],
            "cpu_offload": [
                CPUOffload(offload_params=False),
                CPUOffload(offload_params=True),
            ],
            "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
            "use_orig_params": [True, False],
        }


class TestUnshardParams(TestUnshardParamsBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_writeback(self):
        """Tests the ``writeback`` argument (using default for all others)."""
        self.run_subtests(
            self._get_test_unshard_params_writeback_config(),
            self._test_unshard_params_writeback,
        )

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_param_data(self):
        """
        Tests that parameters are exposed correctly for ``recurse=True`` and
        all other argument configs for a non-FSDP root module.
        """
        self.run_subtests(
            self._get_test_unshard_params_param_data_config(),
            self._test_unshard_params_param_data,
        )

    @skip_if_lt_x_gpu(2)
    def test_unshard_singleton_param_writeback(self):
        """
        Tests ``writeback=True`` for a singleton parameter, which includes
        testing that writing to padding does not persist.
        NOTE: This method depends on FSDP internals.
        """
        model = FSDP(nn.Linear(1, 1, bias=False, device=device_type.type))
        flat_param = model._handle.flat_param
        self.assertEqual(1, flat_param.numel())
        # Write a known value to the *sharded* `FlatParameter`
        with torch.no_grad():
            # For nonzero ranks, this write is to padding
            flat_param[0] = self.rank + 2
        with FSDP.summon_full_params(model, writeback=True):
            self.assertEqual(1, flat_param.numel())
            with torch.no_grad():
                flat_param.zero_()
        # NOTE: This checks that writes to padding did not persist, which is
        # *not* strictly required for correctness.
        if self.rank == 0:  # did not write to padding
            self.assertEqual(0, flat_param[0])
        else:  # wrote to padding
            self.assertEqual(self.rank + 2, flat_param[0])

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_respects_reshard(self):
        """
        Tests that unsharding parameters respects the expected reshard behavior
        between forward and backward as well as after backward.

        For mixed precision, we should *not* respect the reshard behavior
        because the ``summon_full_params()`` forces full precision, which uses
        a different all-gather tensor than the one already in memory and will
        not persist any modifications correctly.
        """
        self.run_subtests(
            {
                "rank0_only": [False, True],
                "offload_to_cpu": [False, True],
                "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
                "use_orig_params": [False, True],
            },
            self._test_unshard_params_respects_reshard,
        )

    def _test_unshard_params_respects_reshard(
        self,
        rank0_only: bool,
        offload_to_cpu: bool,
        mixed_precision: Optional[MixedPrecision],
        use_orig_params: bool,
    ):
        """NOTE: This method depends on FSDP internals."""
        fsdp_kwargs = {
            "mixed_precision": mixed_precision,
            "use_orig_params": use_orig_params,
        }
        model = FSDP(
            nn.Sequential(
                FSDP(
                    nn.Linear(5, 5, bias=False, device=device_type.type), **fsdp_kwargs
                ),
                nn.Linear(5, 3, bias=False, device=device_type.type),
            ),
            **fsdp_kwargs,
        )
        outer_flat_param = model._handle.flat_param
        inner_flat_param = model.module[0]._handle.flat_param
        # NOTE: This assumes uniform sharding with padding across ranks.
        expected_outer_flat_param_unsharded_numel = (
            outer_flat_param.numel() * self.world_size
        )

        def _get_unsharded_storage_size(flat_param: FlatParameter):
            return flat_param._full_param_padded.storage().size()

        # Validate the expected behavior: the root does not reshard after
        # forward; the non-root reshards after forward; and both reshard after
        # backward
        output = model(torch.zeros(5, device=device_type.type))
        self.assertEqual(
            expected_outer_flat_param_unsharded_numel,
            _get_unsharded_storage_size(outer_flat_param),
        )
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))
        output.sum().backward()
        self.assertEqual(0, _get_unsharded_storage_size(outer_flat_param))
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))

        # Check that with parameter unsharding in between forward and backward
        # as well as after backward, the reshard behavior matches
        output = model(torch.zeros(5, device=device_type.type))
        with FSDP.summon_full_params(
            model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            pass
        if mixed_precision is not None:
            # After forcing full precision, we must invalidate the existing
            # unsharded low-precision flat parameter since it will not persist
            # changes from the `summon_full_params()` context, so we cannot
            # respect the reshard behavior
            expected_outer_flat_param_unsharded_numel = 0
        self.assertEqual(
            expected_outer_flat_param_unsharded_numel,
            _get_unsharded_storage_size(outer_flat_param),
        )
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))
        output.sum().backward()
        with FSDP.summon_full_params(
            model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            pass
        self.assertEqual(0, _get_unsharded_storage_size(outer_flat_param))
        self.assertEqual(0, _get_unsharded_storage_size(inner_flat_param))

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_recurse(self):
        """Tests the ``recurse`` argument (using default for all others)."""
        self.run_subtests(
            {
                "recurse": [False, True],
                "unshard_outer": [False, True],
                "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
                "use_orig_params": [False, True],
            },
            self._test_unshard_params_recurse,
        )

    def _test_unshard_params_recurse(
        self,
        recurse: bool,
        unshard_outer: bool,
        mixed_precision: Optional[MixedPrecision],
        use_orig_params: bool,
    ):
        """NOTE: This method depends on FSDP internals."""
        fsdp_kwargs = {
            "mixed_precision": mixed_precision,
            "use_orig_params": use_orig_params,
        }
        model = FSDP(
            nn.Sequential(
                FSDP(
                    nn.Linear(5, 5, bias=False, device=device_type.type), **fsdp_kwargs
                ),
                nn.Linear(5, 3, bias=False, device=device_type.type),
            ),
            **fsdp_kwargs,
        )
        # Hard code the numel values based on the model
        unsharded_inner_numel = 5 * 5
        unsharded_outer_numel = 5 * 3
        if use_orig_params:
            # Account for unsharded padding: since each `FlatParameter` only
            # has one original parameter, we only need to pad for divisibility
            # by world size and not address alignment
            if unsharded_inner_numel % self.world_size:
                unsharded_inner_numel += self.world_size - (
                    unsharded_inner_numel % self.world_size
                )
            if unsharded_outer_numel % self.world_size:
                unsharded_outer_numel += self.world_size - (
                    unsharded_outer_numel % self.world_size
                )
        # Round up the sharded numel to account for padding
        sharded_inner_numel = int(math.ceil(unsharded_inner_numel / self.world_size))
        sharded_outer_numel = int(math.ceil(unsharded_outer_numel / self.world_size))
        inner_flat_param = model.module[0]._handle.flat_param
        outer_flat_param = model._handle.flat_param
        self.assertEqual(sharded_inner_numel, inner_flat_param.numel())
        self.assertEqual(sharded_outer_numel, outer_flat_param.numel())
        expected_outer_numel = (
            unsharded_outer_numel if unshard_outer else sharded_outer_numel
        )
        expected_inner_numel = (
            unsharded_inner_numel
            if recurse or not unshard_outer
            else sharded_inner_numel
        )
        module_to_unshard = model if unshard_outer else model[0]
        with FSDP.summon_full_params(module_to_unshard, recurse=recurse):
            self.assertEqual(expected_outer_numel, outer_flat_param.numel())
            self.assertEqual(expected_inner_numel, inner_flat_param.numel())

    @skip_if_lt_x_gpu(2)
    def test_named_parameters_and_buffers(self):
        """
        Tests that ``named_parameters()`` and ``named_buffers()`` for a
        top-level FSDP-wrapped model matches their behavior for the equivalent
        non-wrapped module.
        """
        self.run_subtests(
            {"prefix": ["", "test_prefix"], "recurse": [False, True]},
            self._test_named_parameters_and_buffers,
        )

    def _test_named_parameters_and_buffers(self, prefix: str, recurse: bool):
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        model.buffer = nn.Buffer(torch.ones(1))
        # Wrap the top-level with FSDP since `named_parameters()` and
        # `named_buffers` will contain FSDP prefixes if called on a non-FSDP
        # root module
        fsdp_model = FSDP(
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_BEFORE,
                deterministic=True,
            ),
            self.process_group,
        )
        fsdp_model.buffer = nn.Buffer(torch.ones(1))
        with FSDP.summon_full_params(fsdp_model):
            for call in ["named_parameters", "named_buffers"]:
                for (n1, p1), (n2, p2) in itertools.zip_longest(
                    getattr(fsdp_model, call)(prefix=prefix, recurse=recurse),
                    getattr(model, call)(prefix=prefix, recurse=recurse),
                ):
                    self.assertEqual(n1, n2)
                    self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_with_grads_core(self):
        """
        Tests the core usage of``with_grads=True`` by comparing against DDP as
        the unsharded equivalent.
        """
        self.run_subtests(
            {
                "writeback": [False, True],
                "offload_to_cpu": [False, True],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [True],
            },
            self._test_with_grads_core,
        )

    def _test_with_grads_core(
        self,
        writeback: bool,
        offload_to_cpu: bool,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        def _check_grads(
            ddp_model: DDP,
            fsdp_model: FSDP,
            old_fsdp_grads: Optional[List[torch.Tensor]],
        ):
            """
            Checks that writes to the FSDP parameters' gradients persist or do
            not persist depending on ``writeback`` and the sharding strategy.
            The DDP model is used for checking gradient parity to ensure that
            FDSP all-gathers the correct gradient values.
            """
            WRITEBACK_FACTOR = 2
            with FSDP.summon_full_params(
                fsdp_model,
                writeback=writeback,
                offload_to_cpu=offload_to_cpu,
                with_grads=True,
            ):
                for (n1, p1), (n2, p2) in zip(
                    ddp_model.module.named_parameters(),
                    fsdp_model.named_parameters(),
                ):
                    self.assertEqual(n1, clean_tensor_name(n2))
                    assert p1.grad is not None
                    torch.testing.assert_close(p1.grad, p2.grad)
                    # Ensure that the tensor is not all zeros, which would
                    # mean that the multiplication is vacuous
                    assert torch.count_nonzero(p2.grad) > 0
                    p2.grad *= WRITEBACK_FACTOR
            new_fsdp_grads = [
                param.grad
                for param in fsdp_model.parameters()
                if param.grad is not None
            ]
            writeback_persists = writeback or (
                sharding_strategy == ShardingStrategy.NO_SHARD and not offload_to_cpu
            )
            for old_grad, new_grad in zip(old_fsdp_grads, new_fsdp_grads):
                if writeback_persists:
                    torch.testing.assert_close(old_grad * WRITEBACK_FACTOR, new_grad)
                else:
                    torch.testing.assert_close(old_grad, new_grad)
            if writeback_persists:
                # Modify the DDP gradients in the same way for parity
                for param in ddp_model.parameters():
                    param.grad *= WRITEBACK_FACTOR

        def _get_error_context(is_supported: bool):
            return (
                contextlib.nullcontext()
                if is_supported
                else self.assertRaises(NotImplementedError)
            )  # some configs are not implemented yet

        def _get_fsdp_grads(fsdp_model: FSDP, is_supported: bool):
            if is_supported:
                return [
                    param.grad.clone()
                    for param in fsdp_model.parameters()
                    if param.grad is not None
                ]
            return None  # unused

        is_supported = use_orig_params and not offload_to_cpu
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        ddp_model = DDP(model, device_ids=[device_type])
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
            fsdp_kwargs={
                "use_orig_params": use_orig_params,
                "sharding_strategy": sharding_strategy,
                "device_id": device_type.type,
            },
        )
        with FSDP.summon_full_params(fsdp_model):
            for p1, p2 in zip(ddp_model.module.parameters(), fsdp_model.parameters()):
                assert torch.all(torch.isclose(p1, p2))

        # Check calling after backward
        inp = fsdp_model.get_input(torch.device(device_type))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()
        old_fsdp_grads = _get_fsdp_grads(fsdp_model, is_supported)
        with _get_error_context(is_supported):
            _check_grads(ddp_model, fsdp_model, old_fsdp_grads)

        # Check calling between forward and backward
        inp = fsdp_model.get_input(torch.device(device_type))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        old_fsdp_grads = _get_fsdp_grads(fsdp_model, is_supported)
        with _get_error_context(is_supported):
            _check_grads(ddp_model, fsdp_model, old_fsdp_grads)

    @skip_if_lt_x_gpu(2)
    def test_with_grads_none_grads(self):
        """
        Tests that if all ranks' ``FlatParameter`` has ``None`` gradient, then
        each original parameter sees ``None`` gradient as well.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_with_grads_none_grads,
        )

    def _test_with_grads_none_grads(self, sharding_strategy: ShardingStrategy):
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
            fsdp_kwargs={
                "use_orig_params": True,
                "sharding_strategy": sharding_strategy,
                "device_id": device_type.type,
            },
        )
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            if fsdp_module._handle:
                assert fsdp_module._handle.flat_param.grad is None
        with FSDP.summon_full_params(fsdp_model, with_grads=True):
            for param in fsdp_model.parameters():
                self.assertTrue(param.grad is None)

    @skip_if_lt_x_gpu(2)
    def test_unshard_submodule(self):
        model = nn.Sequential(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)),
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)),
        ).to(device_type.type)
        model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy((nn.Sequential,)))
        with FSDP.summon_full_params(model[0]):
            # Check that the summoned module does not have its flat parameter
            for param_name, _ in model[0].named_parameters():
                self.assertFalse(FLAT_PARAM in param_name)
            self.assertGreater(len(list(model[0].parameters())), 1)


class TestUnshardParamsNoShard(TestUnshardParamsBase):
    @property
    def world_size(self) -> int:
        return 1

    @skip_if_lt_x_gpu(1)
    def test_unshard_params_writeback_no_shard(self):
        """Tests the ``writeback`` argument (using default for all others)."""
        self.run_subtests(
            self._get_test_unshard_params_writeback_config(),
            self._test_unshard_params_writeback,
        )

    @skip_if_lt_x_gpu(1)
    def test_unshard_params_param_data_no_shard(self):
        """
        Tests that parameters are exposed correctly for ``recurse=True`` and
        all other argument configs for a non-FSDP root module.
        """
        config = self._get_test_unshard_params_param_data_config()
        # TODO: `offload_to_cpu=True` with `NO_SHARD` is not supported yet. See
        # `test_offload_to_cpu_no_shard_raises()`.
        config["offload_to_cpu"] = [False]
        self.run_subtests(
            config,
            self._test_unshard_params_param_data,
        )


class TestUnshardParamsErrors(TestUnshardParamsBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_from_forward_raises(self):
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Parameter(torch.zeros(5))

            def forward(self, fsdp_module):
                with fsdp_module.summon_full_params(fsdp_module):
                    pass

        model = FSDP(MyModule()).to(device_type.type)
        with self.assertRaisesRegex(
            AssertionError, "Cannot manually unshard parameters during forward/backward"
        ):
            model(model)

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_from_backward_raises(self):
        model = FSDP(nn.Linear(2, 1, device=device_type.type))
        output = model(torch.ones(2, device=device_type.type))

        def invalid_backward_hook(*args, **kwargs):
            with FSDP.summon_full_params(model):
                pass

        self.assertTrue(output.requires_grad)
        output.register_hook(invalid_backward_hook)
        with self.assertRaisesRegex(
            AssertionError, "Cannot manually unshard parameters during forward/backward"
        ):
            output.backward()

    @skip_if_lt_x_gpu(2)
    def test_rank0_only_with_writeback_raises(self):
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
        )
        with self.assertRaisesRegex(NotImplementedError, "is not supported"):
            with FSDP.summon_full_params(
                nested_wrapped_module, rank0_only=True, writeback=True
            ):
                pass

    @skip_if_lt_x_gpu(2)
    def test_offload_to_cpu_no_shard_raises(self):
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            {"sharding_strategy": ShardingStrategy.NO_SHARD},
        )
        with self.assertRaisesRegex(NotImplementedError, "is not supported"):
            with FSDP.summon_full_params(
                nested_wrapped_module, rank0_only=True, writeback=True
            ):
                pass


if __name__ == "__main__":
    run_tests()
