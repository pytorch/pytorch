# Owner(s): ["oncall: distributed"]

import sys
import contextlib
from functools import partial
from itertools import product

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    subtest_name,
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

# Various mixed precision configs to test under.
default_mp = MixedPrecision()
mp_diff_reduce = MixedPrecision(reduce_dtype=torch.bfloat16)
mp_diff_buffer = MixedPrecision(buffer_dtype=torch.bfloat16)
mp_diff_buffer_and_reduce = MixedPrecision(
    buffer_dtype=torch.bfloat16, reduce_dtype=torch.float32
)
# Buffer original dtype, which can differ from model params.
buffer_orig_dtype = torch.float64

mp_configs = [
    default_mp, mp_diff_reduce, mp_diff_buffer, mp_diff_buffer_and_reduce
]
params = "mp_config,cpu_offload,backward_prefetch,full_precision_param_dtype"
cpu_offload_config = [
    CPUOffload(offload_params=True), CPUOffload(offload_params=False)
]
backward_prefetch_config = [
    BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST
]
full_precision_param_dtype_config = [torch.float32, torch.float64]
configs = list(product(
    mp_configs,
    cpu_offload_config,
    backward_prefetch_config,
    full_precision_param_dtype_config,
))

test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(BackwardPrefetch.BACKWARD_PRE): "prefetch_pre",
    str(BackwardPrefetch.BACKWARD_POST): "prefetch_post",
    str(default_mp): "mp_fp16",
    str(mp_diff_reduce): "mp_diff_reduce",
    str(mp_diff_buffer): "mp_diff_buffer",
    str(mp_diff_buffer_and_reduce): "mp_diff_buffer_reduce",
    str(torch.float32): "fp32",
    str(torch.float64): "fp64",
}

subtest_name = partial(subtest_name, test_name_mapping)

@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter):
    """
    Patches dist._reduce_scatter_base with a new reduce_scatter_base and
    restores upon exiting. Used for validation of mixed precision
    """
    orig_reduce_scatter = dist._reduce_scatter_base
    dist._reduce_scatter_base = new_reduce_scatter
    try:
        yield
    finally:
        dist._reduce_scatter_base = orig_reduce_scatter

class LinearMixedPrecision(nn.Module):
    """
    A linear module with extra checks for mixed precision training.
    """
    def __init__(self, param_dtype):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False).to(param_dtype)
        self.register_buffer('buffer', torch.randn((1, 2), dtype=buffer_orig_dtype))

    def forward(self, tup):
        # Param and input should be the mixed precision type
        inp, cls, fsdp, mp_config, full_precision_param_dtype = tup
        expected_param_type = mp_config.param_dtype
        expected_buffer_type = mp_config.buffer_dtype
        cls.assertEqual(inp.dtype, expected_param_type)
        # Buffer should be in specified precision as well.
        cls.assertEqual(self.buffer.dtype, expected_buffer_type)

        # In FSDP, self.params should point to the right type.
        num_active_fsdp = 0
        for fsdp_module in FSDP.fsdp_modules(fsdp):
            fsdp_managed_params = fsdp_module.params
            # Single param assumption
            cls.assertEqual(1, len(fsdp_managed_params))
            for param in fsdp_managed_params:
                # FSDP unit is currently active if it is not using the param
                # local shard. This supports both FULL_SHARD and SHARD_GRAD_OP
                # cases. In FULL_SHARD, we have the additional property that
                # param._full_param_padded has not been freed.
                is_fsdp_unit_active = (
                    param._is_sharded and
                    (param.data.data_ptr() != param._local_shard.data_ptr())
                )
                if is_fsdp_unit_active:
                    num_active_fsdp += 1
                    # This FSDP unit is active, verify param points to mixed
                    cls.assertEqual(param.dtype, expected_param_type)
                    # _rebuild_full_param should have also freed the fp16 shard.
                    cls.assertEqual(0, param._mp_shard.storage().size())
                elif param._is_sharded:
                    # This FSDP unit is not active as full param has been
                    # freed or not yet allocated. Ensure param points to full
                    # precision param.
                    cls.assertEqual(param.dtype, full_precision_param_dtype)
        # We should have gotten at least one active FSDP unit for sharded
        # (world size > 1) cases. For cases where param is not sharded
        # (ie world_size == 1) it is a bit hard to check if FSDP unit is active
        # as we'd always point to the local shard, so we rely on the forward
        # pass self.lin(inp) working well and inp being reduced precision to
        # implicitly validate that the param is indeed in the reduced precision.
        if cls.world_size > 1:
            cls.assertGreater(num_active_fsdp, 0)

        return (self.lin(inp), cls, fsdp, mp_config, full_precision_param_dtype)


class TestFSDPMixedPrecision(FSDPTest):
    @property
    def world_size(self):
        raise ValueError("To be implemented by child classes")

    def _get_simple_nested_model(self, param_dtype, *fsdp_args, **fsdp_kwargs):
        model = FSDP(
            nn.Sequential(
                FSDP(LinearMixedPrecision(param_dtype).cuda(), *fsdp_args, **fsdp_kwargs),
                LinearMixedPrecision(param_dtype).cuda(),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, param_dtype, *fsdp_args, **fsdp_kwargs):
        model = FSDP(LinearMixedPrecision(param_dtype).cuda(), *fsdp_args, **fsdp_kwargs)
        return model

    def _validate_mp_shard_freed(self, fsdp_model):
        """
        Ensures that the mixed precision shard is greed for all FSDP units.
        """
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                self.assertEqual(0, param._mp_shard.storage().size())

    def _reduce_scatter_base_validate_mp(
        self,
        orig_reduce_scatter,
        mp_config,
        *args,
        **kwargs
    ):
        """
        Performs dist._reduce_scatter_base but verifies mixed precision settings
        before. This is to test mixed precision is working as expected during
        backward pass.
        """
        tensors = []
        for x in args:
            if isinstance(x, torch.Tensor):
                tensors.append(x)
        for _, x in kwargs.items():
            if isinstance(x, torch.Tensor):
                tensors.append(x)

        # reduce_dtype has higher priority than param_dtype, because mixed_precision
        # supports overriding param_dtype with reduce_dtype to control the
        # reduction precision. In the case where reduce_dtype == param_dtype
        # this tests that gradients are in the expected precision as well.
        expected_dtype = mp_config.reduce_dtype
        for t in tensors:
            self.assertEqual(expected_dtype, t.dtype)

        return orig_reduce_scatter(*args, **kwargs)

    def _run_test_mixed_precision_e2e(
        self,
        mp_config,
        cpu_offload,
        backward_prefetch,
        full_precision_param_dtype,
        sharding_strategy,
    ):
        torch.cuda.set_device(self.rank)
        fsdp_models = [
            self._get_simple_model(
                param_dtype=full_precision_param_dtype,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                mixed_precision=mp_config,
                backward_prefetch=backward_prefetch
            ),
            self._get_simple_nested_model(
                param_dtype=full_precision_param_dtype,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                mixed_precision=mp_config,
                backward_prefetch=backward_prefetch
            ),
        ]
        for model in fsdp_models:
            if not cpu_offload.offload_params:
                model.cuda()

            # Patch reduce_scatter to add validation for mixed precision types.
            orig_reduce_scatter = dist._reduce_scatter_base
            test_reduce_scatter = partial(
                self._reduce_scatter_base_validate_mp, orig_reduce_scatter, mp_config,
            )
            with patch_reduce_scatter(test_reduce_scatter):
                optim = torch.optim.Adam(model.parameters())

                for _ in range(3):
                    inp = torch.randn(3, 10).cuda()
                    # Forward pass of LinearMixedPrecision check casting of
                    # inputs, params, buffers.
                    act, *_ = model(
                        (inp, self, model, mp_config, full_precision_param_dtype)
                    )
                    # Buffers should be casted.
                    for buf in model.buffers():
                        self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                    # p._mp_shard should be freed.
                    if model.params[0]._is_sharded:  # i.e. world_size > 1
                        # TODO: free the mixed precision shard after forward
                        # when world_size == 1 as well, currently when
                        # world_size == 1 it is only freed after backward.
                        self._validate_mp_shard_freed(model)

                    loss = act.sum()
                    self.assertEqual(loss.dtype, mp_config.param_dtype)
                    # Will run patched reduce scatter that validates mixed_precision
                    # types in backward.
                    loss.backward()
                    # Buffers stay casted even after backwards.
                    for buf in model.buffers():
                        self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                    # p._mp_shard should be freed.
                    self._validate_mp_shard_freed(model)

                    # Ensure params and grads are in full precision
                    for param in model.parameters():
                        self.assertEqual(param.dtype, full_precision_param_dtype)
                        if param.grad is not None:
                            self.assertEqual(param.grad.dtype, full_precision_param_dtype)

                    optim.step()

                    # Summon full params should be in full precision
                    with model.summon_full_params():
                        # It is not expected for summon_full_params to allocate
                        # a mixed precision shard.
                        self._validate_mp_shard_freed(model)
                        params = list(model.parameters())
                        for p in params:
                            self.assertEqual(p.dtype, full_precision_param_dtype)

                        # Note that buffers are cast only once and only restored
                        # to the original buffer dtype in state_dict, so
                        # summon_full_params is not expected to restore buffer
                        # types to their original.
                        named_buffers = dict(model.named_buffers())
                        for k, v in named_buffers.items():
                            self.assertEqual(v.dtype, mp_config.buffer_dtype)

                    # state_dict should be in full precision
                    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    for name, tensor in state_dict.items():
                        # Parameters and buffers are checkpointed in their
                        # original dtypes, which may be different.
                        if name in named_buffers.keys():
                            self.assertEqual(tensor.dtype, buffer_orig_dtype)
                        else:
                            self.assertEqual(
                                tensor.dtype, full_precision_param_dtype,
                                f"{name}: {tensor.dtype} vs {full_precision_param_dtype}"
                            )

                    # After state_dict, buffer's dtype should have been restored
                    # to the mixed precision one.
                    for buf in model.buffers():
                        self.assertEqual(buf.dtype, mp_config.buffer_dtype)


class TestFSDPMixedPrecisionSharded(TestFSDPMixedPrecision):

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_no_reshard_after_forward(self):
        # Note that we don't exercise all possible different configs so as to
        # not increase test TTS too much.
        self._run_test_mixed_precision_e2e(
            mp_config=mp_diff_buffer_and_reduce,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixed_precision_e2e_full_shard(
        self,
        mp_config,
        cpu_offload,
        backward_prefetch,
        full_precision_param_dtype
    ):
        self._run_test_mixed_precision_e2e(
            mp_config,
            cpu_offload,
            backward_prefetch,
            full_precision_param_dtype,
            ShardingStrategy.FULL_SHARD,
        )

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_embedding_table(self):
        # Basic test to ensure int inputs are not casted which would break
        # modules such as embedding tables.
        mp_config = MixedPrecision()
        model = self._get_wrapped_model(
            group=torch.distributed.distributed_c10d._get_default_group(),
            config={"mixed_precision": mp_config}
        )
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(6):
            inp = model.module.get_input(torch.device("cuda"))
            # This would fail if we casted integer module inputs such as for
            # embedding tables.
            output = model(*inp)
            loss = model.module.get_loss(inp, output).cuda()
            self.assertEqual(loss.dtype, mp_config.param_dtype)
            model.module.run_backward(loss)
            optim.step()

class TestFSDPMixedPrecisionUnsharded(TestFSDPMixedPrecision):
    """
    Smaller test suite for unshared param (i.e. world_size == 1) case.
    """
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_no_reshard_after_forward(self):
        # Note that we don't exercise all possible different configs so as to
        # not increase test TTS too much.
        self._run_test_mixed_precision_e2e(
            mp_config=mp_diff_buffer_and_reduce,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_e2e_full_shard(self):
        self._run_test_mixed_precision_e2e(
            mp_config=mp_diff_buffer_and_reduce,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )

instantiate_parametrized_tests(TestFSDPMixedPrecisionSharded)

if __name__ == "__main__":
    run_tests()
