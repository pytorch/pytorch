# Owner(s): ["oncall: distributed"]
import contextlib
import itertools
import math
import sys
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.flat_param import FlatParamHandle
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    DeterministicModel,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
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


def _run_test_summon_full_param_writeback(
    cls, writeback, modify_outer, *fsdp_args, **fsdp_kwargs
):
    with enable_wrap(wrapper_cls=FSDP, *fsdp_args, **fsdp_kwargs):
        lin1 = wrap(nn.Linear(5, 5, bias=False).cuda(cls.rank))
        lin2 = nn.Linear(5, 3, bias=False).cuda(cls.rank)
        model = wrap(nn.Sequential(lin1, lin2))

    # set the value
    outer_param = model._handles[0].flat_param
    inner_param = model.module[0]._handles[0].flat_param
    p = outer_param if modify_outer else inner_param

    with torch.no_grad():
        # This sets the local shard value
        p[0] = cls.rank + 2

    with model.summon_full_params(model, writeback=writeback):
        with torch.no_grad():
            p.copy_(torch.zeros_like(p))

    if writeback or cls.world_size == 1:
        # When world_size = 1, FSDP does not shard and parameter is not set to
        # a local shard, so write is always reflected.
        cls.assertEqual(p.cpu()[0], 0)
    else:
        cls.assertEqual(p.cpu()[0], cls.rank + 2)


class TestSummonFullParamsNoShard(FSDPTest):
    @property
    def world_size(self):
        return 1  # does not shard

    @skip_if_lt_x_gpu(2)
    # TODO: CPUOffload summon + writeback does not
    # work when param is not sharded
    # (currently when world_size == 1)
    def test_summon_full_param_writeback(self):
        subtest_config = {
            "writeback": [True, False],
            "modify_outer": [True, False],
            "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
            "use_orig_params": [True, False],
        }
        self.run_subtests(
            subtest_config,
            _run_test_summon_full_param_writeback,
            cls=self,
            cpu_offload=CPUOffload(offload_params=False),
        )


class TestSummonFullParams(FSDPTest):
    @property
    def world_size(self):
        return 2

    def get_model_param_count(self, m):
        return sum([p.numel() for p in m.parameters()])

    # padding ensures that all shards have the same size with the least amount of padding
    def get_expected_sharded_size(self, global_size):
        return int(math.ceil(global_size / self.world_size))

    @skip_if_lt_x_gpu(2)
    def test_summon_full_param_writeback(self):
        subtest_config = {
            "writeback": [True, False],
            "modify_outer": [True, False],
            "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
            "cpu_offload": [
                CPUOffload(offload_params=False),
                CPUOffload(offload_params=True),
            ],
            "use_orig_params": [True, False],
        }
        self.run_subtests(
            subtest_config,
            _run_test_summon_full_param_writeback,
            cls=self,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("mixed_precision", [True, False])
    def test_summon_full_param_shard_value(self, mixed_precision):
        mixed_precision = (
            MixedPrecision(param_dtype=torch.float16) if mixed_precision else None
        )
        raw_model = nn.Linear(10, 11)
        raw_model_size = self.get_model_param_count(raw_model)
        expected_shard_size = self.get_expected_sharded_size(raw_model_size)

        model = FSDP(raw_model.cuda(self.rank), mixed_precision=mixed_precision)
        self.assertEqual(expected_shard_size, self.get_model_param_count(model))

        # we're assuming a single flattened param
        self.assertEqual(1, len(list(model.parameters())))

        my_shard = torch.clone(next(model.parameters()))

        with model.summon_full_params(model):
            self.assertEqual(raw_model_size, self.get_model_param_count(model))
            parameters = list(model.parameters())
            all_shards = FlatParamHandle.flatten_params(parameters, requires_grad=False)
            my_slice = torch.chunk(all_shards, self.world_size)[self.rank]

            # shards are padded but the full_param tensor is not
            a, b = my_shard[0 : my_slice.numel()], my_slice
            self.assertTrue(
                torch.equal(my_shard[0 : my_slice.numel()].cpu(), my_slice.cpu())
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("recurse", [True, False])
    @parametrize("summon_outer", [True, False])
    @parametrize("mixed_precision", [True, False])
    def test_summon_full_param_recursive(self, recurse, summon_outer, mixed_precision):
        mixed_precision = (
            MixedPrecision(param_dtype=torch.float16) if mixed_precision else None
        )
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(5, 5, bias=False), mixed_precision=mixed_precision),
                nn.Linear(5, 3, bias=False),
            ),
            mixed_precision=mixed_precision,
        ).cuda(self.rank)

        global_inner_numel = self.get_model_param_count(nn.Linear(5, 5, bias=False))
        global_outer_numel = self.get_model_param_count(nn.Linear(5, 3, bias=False))

        shard_inner_numel = int(math.ceil(global_inner_numel / self.world_size))
        shard_outer_numel = int(math.ceil(global_outer_numel / self.world_size))

        outer_param = model._handles[0].flat_param
        inner_param = model.module[0]._handles[0].flat_param
        self.assertEqual(shard_outer_numel, outer_param.numel())
        self.assertEqual(shard_inner_numel, inner_param.numel())

        model_to_summon = model if summon_outer else model[0]
        # outer is summoned if _summon_full_param is called on the outer FSDP module
        expected_outer_numel = global_outer_numel if summon_outer else shard_outer_numel

        # inner is summoned if _summon_full_param is called with recursion or on the inner FSDP module
        expected_inner_numel = (
            global_inner_numel if recurse or not summon_outer else shard_inner_numel
        )

        with model_to_summon.summon_full_params(model_to_summon, recurse=recurse):
            self.assertEqual(expected_outer_numel, outer_param.numel())
            self.assertEqual(expected_inner_numel, inner_param.numel())

    @skip_if_lt_x_gpu(2)
    def test_cannot_summon_full_params_from_forward(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.zeros(5))

            def forward(self, fsdp_module):
                with fsdp_module.summon_full_params(fsdp_module):
                    pass

        model = FSDP(MyModule()).cuda(self.rank)
        with self.assertRaisesRegex(
            ValueError, "current state is TrainingState.FORWARD"
        ):
            model(model)

    @skip_if_lt_x_gpu(2)
    def test_cannot_summon_full_params_from_backward(self):
        model = FSDP(nn.Linear(2, 1)).cuda(self.rank)

        output = model(torch.ones(2).cuda(self.rank))

        def bad_backwards_hook(tensor):
            with model.summon_full_params(model):
                pass
            return None

        self.assertTrue(output.requires_grad)
        output.register_hook(bad_backwards_hook)

        with self.assertRaisesRegex(
            ValueError, "current state is TrainingState.FORWARD_BACKWARD"
        ):
            output.backward()

    @skip_if_lt_x_gpu(2)
    def test_summon_full_params_respects_reshard_after_forward(self):
        self.run_subtests(
            {
                "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
                "use_orig_params": [False, True],
            },
            self._test_summon_full_params_respects_reshard_after_forward,
        )

    def _test_summon_full_params_respects_reshard_after_forward(
        self, mixed_precision: Optional[MixedPrecision], use_orig_params: bool
    ):
        fsdp_kwargs = {
            "mixed_precision": mixed_precision,
            "use_orig_params": use_orig_params,
        }
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(5, 5, bias=False), **fsdp_kwargs),
                nn.Linear(5, 3, bias=False),
            ),
            **fsdp_kwargs,
        ).cuda(self.rank)

        outer_param = model._handles[0].flat_param
        inner_param = model.module[0]._handles[0].flat_param
        outer_full_param_size = outer_param.numel() * self.world_size

        # trigger lazy init
        model(torch.zeros(5).cuda(self.rank))
        # the root FSDP module keeps all params around
        self.assertEqual(
            outer_full_param_size, outer_param._full_param_padded.storage().size()
        )
        self.assertEqual(0, inner_param._full_param_padded.storage().size())

        # similarly summon_full_params should have the same behavior
        with model.summon_full_params(model):
            pass
        self.assertEqual(
            outer_full_param_size, outer_param._full_param_padded.storage().size()
        )
        self.assertEqual(0, inner_param._full_param_padded.storage().size())

    @skip_if_lt_x_gpu(2)
    def test_summon_single_param(self):
        model = FSDP(nn.Linear(1, 1, bias=False)).cuda(self.rank)

        p = model._handles[0].flat_param
        self.assertEqual(1, p.numel())

        with torch.no_grad():
            # This sets the local shard value
            p[0] = self.rank + 2

        with model.summon_full_params(model, writeback=True):
            self.assertEqual(1, p.numel())
            with torch.no_grad():
                p.copy_(torch.zeros_like(p))

        # most ranks hold no data and wrote to padding so only rank zero will observe the above write
        if self.rank == 0:
            self.assertEqual(0, p[0])
        else:
            self.assertEqual(self.rank + 2, p[0])

    @skip_if_lt_x_gpu(2)
    @parametrize("rank0_only", [True, False])
    @parametrize("offload_to_cpu", [True, False])
    def test_summon_full_params_equivalence(self, rank0_only, offload_to_cpu):
        offload = CPUOffload(offload_params=True)
        model = FSDP(
            DeterministicModel(wrap_fsdp=True, cpu_offload=offload), cpu_offload=offload
        )
        local_model = DeterministicModel(wrap_fsdp=False)

        params_to_compare = (
            [p.clone() for p in model.parameters()]
            if rank0_only and self.rank != 0
            else list(local_model.parameters())
        )

        writeback = not rank0_only

        with model.summon_full_params(
            model,
            recurse=True,
            rank0_only=rank0_only,
            writeback=writeback,
            offload_to_cpu=offload_to_cpu,
        ):
            if writeback:
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(1)
                    for p in params_to_compare:
                        p.add_(1)
            # Below sleep causes failures without stream synchronization in
            # summon_full_params fix.
            torch.cuda._sleep(1000000)
            # FSDP param deepcopy() of params has issues
            fsdp_params = [p.clone() for p in model.parameters()]

        self.assertEqual(fsdp_params, params_to_compare)

        # CPU offload is enabled for main API, so we should point back to CPU
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("cpu"))

    @skip_if_lt_x_gpu(2)
    def test_summon_from_non_fsdp(self):
        class FSDPContainer(nn.Module):
            def __init__(self, fsdp_1, fsdp_2, fsdp_3):
                super().__init__()
                self.fsdp_1 = fsdp_1
                self.fsdp_2 = fsdp_2
                self.fsdp_3 = fsdp_3

        model_fsdp = FSDPContainer(
            FSDP(DeterministicModel(wrap_fsdp=True)),
            FSDP(DeterministicModel(wrap_fsdp=True)),
            DeterministicModel(wrap_fsdp=False),
        )
        model_no_fsdp = FSDPContainer(
            DeterministicModel(wrap_fsdp=False),
            DeterministicModel(wrap_fsdp=False),
            DeterministicModel(wrap_fsdp=False),
        )

        params_to_compare = list(model_no_fsdp.parameters())
        with FSDP.summon_full_params(model_fsdp):
            fsdp_params = [p.clone() for p in model_fsdp.parameters()]

        self.assertEqual(params_to_compare, fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("rank0_only", [True, False])
    @parametrize("offload_to_cpu", [True, False])
    @parametrize("mixed_precision", [True, False])
    def test_reshard_outside_forward_backward_iteration(
        self, rank0_only, offload_to_cpu, mixed_precision
    ):
        mixed_precision = (
            MixedPrecision(param_dtype=torch.float16) if mixed_precision else None
        )
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(5, 5, bias=False), mixed_precision=mixed_precision),
                nn.Linear(5, 1, bias=False),
            ),
            mixed_precision=mixed_precision,
        ).cuda(self.rank)

        outer_param = model._handles[0].flat_param
        inner_param = model.module[0]._handles[0].flat_param
        outer_full_param_size = outer_param.numel() * self.world_size

        # First lets validate our assumption about resharding

        output = model(torch.zeros(5).cuda(self.rank))
        # the root FSDP module keeps all params around
        self.assertEqual(
            outer_full_param_size, outer_param._full_param_padded.storage().size()
        )
        self.assertEqual(0, inner_param._full_param_padded.storage().size())

        output.backward()
        # we reshard everything after backward() finishes
        self.assertEqual(0, outer_param._full_param_padded.storage().size())
        self.assertEqual(0, inner_param._full_param_padded.storage().size())

        # now lets repeat it with summon done in between

        output = model(torch.zeros(5).cuda(self.rank))
        self.assertEqual(
            outer_full_param_size, outer_param._full_param_padded.storage().size()
        )
        self.assertEqual(0, inner_param._full_param_padded.storage().size())
        with model.summon_full_params(
            model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            pass
        self.assertEqual(
            outer_full_param_size, outer_param._full_param_padded.storage().size()
        )
        self.assertEqual(0, inner_param._full_param_padded.storage().size())

        output.backward()
        with model.summon_full_params(
            model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            pass
        self.assertEqual(0, outer_param._full_param_padded.storage().size())
        self.assertEqual(0, inner_param._full_param_padded.storage().size())

    @skip_if_lt_x_gpu(2)
    @parametrize("rank0_only", [True, False])
    @parametrize("offload_to_cpu", [True, False])
    @parametrize("mixed_precision", [True, False])
    def test_params_are_unflattenned(self, rank0_only, offload_to_cpu, mixed_precision):
        layer_shape = (10, 12)
        model = nn.Linear(*layer_shape, bias=False).cuda(self.rank)
        mixed_precision = (
            MixedPrecision(param_dtype=torch.float16) if mixed_precision else None
        )
        fsdp_model = FSDP(deepcopy(model), mixed_precision=mixed_precision).cuda(
            self.rank
        )

        def _get_flat_param():
            return fsdp_model._handles[0].flat_param

        flattened_param = _get_flat_param()
        self.assertEqual(layer_shape[0] * layer_shape[1] / 2, flattened_param.numel())

        with fsdp_model.summon_full_params(
            fsdp_model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            if self.rank == 0 or not rank0_only:
                self.assertEqual(fsdp_model.weight.shape, model.weight.shape)
                expected_device = (
                    torch.device("cpu")
                    if offload_to_cpu
                    else torch.device("cuda", torch.cuda.current_device())
                )
                self.assertTrue(expected_device == fsdp_model.weight.device)
            else:
                # Nonzero rank with rank0_only maintains original params.
                flat_within_ctx = _get_flat_param()
                self.assertEqual(flat_within_ctx, flattened_param)
                self.assertEqual(
                    flat_within_ctx.device, torch.device(torch.cuda.current_device())
                )

        # CPU offload should restore the param device
        param = next(fsdp_model.parameters())
        self.assertTrue(
            param.device == torch.device("cuda", torch.cuda.current_device())
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("rank0_only", [True, False])
    @parametrize("offload_to_cpu", [True, False])
    @parametrize("mixed_precision", [True, False])
    def test_params_count_and_value(
        self,
        rank0_only: bool,
        offload_to_cpu: bool,
        mixed_precision: bool,
    ):
        mixed_precision = (
            MixedPrecision(param_dtype=torch.float16) if mixed_precision else None
        )
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        fsdp_model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        dev = (
            torch.device("cpu")
            if offload_to_cpu
            else torch.device("cuda", torch.cuda.current_device())
        )
        params_to_compare = (
            [p.to(dev) for p in model.module.parameters()]
            if not rank0_only or self.rank == 0
            else list(p.clone() for p in fsdp_model.parameters())
        )
        with FSDP.summon_full_params(
            fsdp_model, rank0_only=rank0_only, writeback=not rank0_only
        ):
            for p1, p2 in itertools.zip_longest(
                fsdp_model.parameters(), params_to_compare
            ):
                self.assertEqual(p1, p2)

        # CPU offload should restore the param device
        param = next(fsdp_model.parameters())
        self.assertTrue(
            param.device == torch.device("cuda", torch.cuda.current_device())
        )

    @skip_if_lt_x_gpu(2)
    def test_raises_rank0_with_writeback(self):
        """Tests that ``summon_full_params()`` with both ``rank0_only=True``
        and ``writeback=True`` raises an error."""
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
        )
        with self.assertRaisesRegex(ValueError, "is not supported"):
            with FSDP.summon_full_params(
                nested_wrapped_module, rank0_only=True, writeback=True
            ):
                pass

    @skip_if_lt_x_gpu(2)
    @parametrize("prefix", ["", "test_prefix"])
    @parametrize("recurse", [False, True])
    def test_named_parameters_buffers(self, prefix: str, recurse: bool):
        """Tests that ``named_parameters()`` and ``named_buffers()`` for a
        top-level FSDP-wrapped model matches their behavior for the equivalent
        non-wrapped model."""
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        model.register_buffer("buffer", torch.ones(1))
        # `named_parameters()` and `named_buffers` will contain FSDP prefixes
        # if called on a non-FSDP root module
        fsdp_model = FSDP(
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                deterministic=True,
            ),
            self.process_group,
        )
        fsdp_model.register_buffer("buffer", torch.ones(1))
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
        """Tests the core usage of ``summon_full_params(with_grads=True)``."""
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
                    # Parameter names are only expected to match because
                    # `fsdp_model` has top-level FSDP, so its
                    # `named_parameters()` cleans *all* of the names
                    self.assertEqual(n1, n2)
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
            writeback_persists = (
                writeback or sharding_strategy == ShardingStrategy.NO_SHARD
            )
            for old_grad, new_grad in zip(old_fsdp_grads, new_fsdp_grads):
                if writeback_persists:
                    torch.testing.assert_close(old_grad * WRITEBACK_FACTOR, new_grad)
                else:
                    torch.testing.assert_close(old_grad, new_grad)
            if writeback_persists:
                # Modify the DDP gradients for parity
                for param in ddp_model.parameters():
                    param.grad *= WRITEBACK_FACTOR

        def _get_error_context(is_supported: bool):
            return (
                contextlib.suppress()
                if is_supported
                else self.assertRaises(NotImplementedError)
            )  # some configs not implemented yet

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
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        ddp_model = DDP(model, device_ids=[self.rank])
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
            fsdp_kwargs={
                "use_orig_params": use_orig_params,
                "sharding_strategy": sharding_strategy,
            },
        )
        with FSDP.summon_full_params(fsdp_model):
            for p1, p2 in zip(ddp_model.module.parameters(), fsdp_model.parameters()):
                assert torch.all(torch.isclose(p1, p2))

        # Check `summon_full_params()` after backward
        inp = fsdp_model.get_input(torch.device("cuda"))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()
        old_fsdp_grads = _get_fsdp_grads(fsdp_model, is_supported)
        with _get_error_context(is_supported):
            _check_grads(ddp_model, fsdp_model, old_fsdp_grads)

        # Check `summon_full_params()` between forward and backward
        inp = fsdp_model.get_input(torch.device("cuda"))
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
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
            fsdp_kwargs={
                "use_orig_params": True,
                "sharding_strategy": sharding_strategy,
            },
        )
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            for handle in fsdp_module._handles:
                assert handle.flat_param.grad is None
        with FSDP.summon_full_params(fsdp_model, with_grads=True):
            for param in fsdp_model.parameters():
                self.assertTrue(param.grad is None)


instantiate_parametrized_tests(TestSummonFullParams)
instantiate_parametrized_tests(TestSummonFullParamsNoShard)


if __name__ == "__main__":
    run_tests()
