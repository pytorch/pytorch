# Owner(s): ["oncall: distributed"]

import contextlib
import sys
from functools import partial
from itertools import product
from typing import Any, Dict, List

import torch
import torch.cuda.nccl as nccl
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.modules.batchnorm import _BatchNorm
from torch.testing._internal.common_cuda import CUDA11OrLater
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    sandcastle_skip_if,
    TEST_WITH_DEV_DBG_ASAN,
)

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = sandcastle_skip_if(not HAS_TORCHVISION, "no torchvision")


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
default_mp = MixedPrecision(
    param_dtype=torch.float16,
    buffer_dtype=torch.float16,
    reduce_dtype=torch.float16,
)

# Params and buffers are not cast, comm only happens
# in reduced precision.
mp_only_reduce = MixedPrecision(reduce_dtype=torch.float16)

# Only parameters are cast (thus comm should happen in the param_dtype precision)
mp_only_param_and_buf = MixedPrecision(
    param_dtype=torch.float16, buffer_dtype=torch.float16
)

# Nothing is cast (thus param, comm, grad, and buffer should be in the full precision)
mp_no_mixed_precision = MixedPrecision()

nccl_supports_bf16 = (
    CUDA11OrLater and dist.is_nccl_available() and nccl.version() >= (2, 10)
)

mp_configs = [default_mp, mp_only_reduce, mp_only_param_and_buf, mp_no_mixed_precision]
if nccl_supports_bf16:
    mp_diff_buffer_and_reduce = MixedPrecision(
        param_dtype=torch.float16,
        buffer_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    mp_configs.extend([mp_diff_buffer_and_reduce])

# Buffer original dtype, which can differ from model params.
_BUFFER_ORIG_DTYPE = torch.float64

params = "mp_config,cpu_offload,full_precision_param_dtype,enable_sharded_grad_scaler"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
full_precision_param_dtype_config = [torch.float32, torch.float64]
enable_sharded_grad_scaler = ["enable_sharded_grad_scaler", None]

configs = list(
    product(
        mp_configs,
        cpu_offload_config,
        full_precision_param_dtype_config,
        enable_sharded_grad_scaler,
    )
)

test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(default_mp): "mp_fp16",
    str(mp_only_reduce): "mp_only_reduce",
    str(mp_only_param_and_buf): "mp_only_param_and_buf",
    str(mp_no_mixed_precision): "mp_no_mp",
    str(torch.float32): "fp32",
    str(torch.float64): "fp64",
    "enable_sharded_grad_scaler": "enable_sharded_grad_scaler",
}

if nccl_supports_bf16:
    test_name_mapping.update(
        {
            str(mp_diff_buffer_and_reduce): "mp_diff_buffer_reduce",
        }
    )

subtest_name = partial(subtest_name, test_name_mapping)

_CURRENT_FULL_PRECISION_PARAM_DTYPE = None


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter, full_precision_param_dtype):
    """
    Patches ``dist.reduce_scatter_tensor`` with ``new_reduce_scatter`` and
    restores upon exiting. Used for validation of mixed precision.
    """
    orig_reduce_scatter = dist.reduce_scatter_tensor
    dist.reduce_scatter_tensor = new_reduce_scatter
    global _CURRENT_FULL_PRECISION_PARAM_DTYPE
    _CURRENT_FULL_PRECISION_PARAM_DTYPE = full_precision_param_dtype
    try:
        yield
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter
        _CURRENT_FULL_PRECISION_PARAM_DTYPE = None


class LinearMixedPrecision(nn.Module):
    """
    A linear module with extra checks for mixed precision training.
    """

    def __init__(self, param_dtype, buffer_name="buffer"):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False).to(param_dtype)
        # Use a configurable buffer name to avoid all submodules sharing the
        # same buffer name, which may hide prefixed vs. unprefixed name bugs
        self.buffer_name = buffer_name
        self.register_buffer(buffer_name, torch.randn((1, 2), dtype=_BUFFER_ORIG_DTYPE))
        self._orig_param_type = param_dtype
        self._orig_buffer_dtype = _BUFFER_ORIG_DTYPE

    def forward(self, tup):
        # Param and input should be the mixed precision type
        inp, cls, fsdp, mp_config, full_precision_param_dtype = tup
        expected_param_type = (
            mp_config.param_dtype
            if mp_config.param_dtype is not None
            else self._orig_param_type
        )
        expected_buffer_type = (
            mp_config.buffer_dtype
            if mp_config.buffer_dtype is not None
            else self._orig_buffer_dtype
        )
        cls.assertEqual(inp.dtype, expected_param_type)
        # Buffer should be in specified precision as well.
        cls.assertEqual(getattr(self, self.buffer_name).dtype, expected_buffer_type)

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
                param_is_sharded = (
                    fsdp_module.sharding_strategy != ShardingStrategy.NO_SHARD
                    and fsdp_module.world_size > 1
                )
                is_fsdp_unit_active = (
                    param_is_sharded
                    and param.data.data_ptr() != param._local_shard.data_ptr()
                )
                if is_fsdp_unit_active:
                    num_active_fsdp += 1
                    # This FSDP unit is active, verify param points to mixed
                    cls.assertEqual(param.dtype, expected_param_type)
                    # _rebuild_full_param should have also freed the fp16 shard.
                    # Shard is never allocated if param_dtype mixed precision is not
                    # enabled.
                    if mp_config.param_dtype is not None:
                        cls.assertEqual(0, param._mp_shard.storage().size())
                    else:
                        cls.assertFalse(hasattr(param, "_mp_shard"))
                elif param_is_sharded:
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
                FSDP(
                    LinearMixedPrecision(param_dtype, buffer_name="buffer0").cuda(),
                    *fsdp_args,
                    **fsdp_kwargs,
                ),
                LinearMixedPrecision(param_dtype, buffer_name="buffer1").cuda(),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, param_dtype, *fsdp_args, **fsdp_kwargs):
        model = FSDP(
            LinearMixedPrecision(param_dtype).cuda(), *fsdp_args, **fsdp_kwargs
        )
        return model

    def _validate_no_mp_shard(self, fsdp_model):
        """
        Validates that there is no mixed precision _mp_shard allocated
        when it is not expected to be.
        """
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                self.assertFalse(hasattr(param, "_mp_shard"))

    def _validate_mp_shard_freed(self, fsdp_model):
        """
        Ensures that the mixed precision shard is greed for all FSDP units.
        """
        fsdp_units = FSDP.fsdp_modules(fsdp_model)
        for fsdp in fsdp_units:
            for param in fsdp.params:
                self.assertEqual(0, param._mp_shard.storage().size())

    def _reduce_scatter_validate_mp(
        self, orig_reduce_scatter, mp_config, *args, **kwargs
    ):
        """
        Runs reduce-scatter but verifies mixed precision settings before. This
        is to test mixed precision is working as expected during backward pass.
        In particular it ensures that the gradients were cast to the right type
        and comm. is going to happen in the right type.
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
        # If reduce_dtype is not specified (is None) we comm. in the param_dtype
        # if that is specified, otherwise full precision dtype.
        expected_dtype = (
            mp_config.reduce_dtype
            if mp_config.reduce_dtype is not None
            else (
                mp_config.param_dtype
                if mp_config.param_dtype is not None
                else _CURRENT_FULL_PRECISION_PARAM_DTYPE
            )
        )

        for t in tensors:
            self.assertEqual(expected_dtype, t.dtype)

        return orig_reduce_scatter(*args, **kwargs)

    def _test_grads_reduced_precision(self, offload_params: bool):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(10, 10)
                self.lin2 = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin2(self.lin1(x))

        m = MyModel().cuda()
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
            keep_low_precision_grads=True,
        )
        fsdp_kwargs = {
            "mixed_precision": mp,
            "cpu_offload": CPUOffload(offload_params=offload_params),
        }
        m.lin1 = FSDP(m.lin1, **fsdp_kwargs)
        m = FSDP(m, **fsdp_kwargs)
        for _ in range(6):
            inp = torch.ones(1, 10)
            m(inp).sum().backward()
            for param in m.parameters():
                self.assertEqual(torch.float16, param.grad.dtype)

        dist.barrier()

    def _run_test_mixed_precision_e2e(
        self,
        mp_config,
        cpu_offload,
        backward_prefetch,
        forward_prefetch,
        full_precision_param_dtype,
        sharding_strategy,
        enable_sharded_grad_scaler,
    ):
        torch.cuda.set_device(self.rank)
        fsdp_models = [
            self._get_simple_model(
                param_dtype=full_precision_param_dtype,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                mixed_precision=mp_config,
                backward_prefetch=backward_prefetch,
                forward_prefetch=forward_prefetch,
            ),
            self._get_simple_nested_model(
                param_dtype=full_precision_param_dtype,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                mixed_precision=mp_config,
                backward_prefetch=backward_prefetch,
                forward_prefetch=forward_prefetch,
            ),
        ]
        for model in fsdp_models:
            if not cpu_offload.offload_params:
                model.cuda()

            # Patch reduce_scatter to add validation for mixed precision types.
            orig_reduce_scatter = dist.reduce_scatter_tensor
            test_reduce_scatter = partial(
                self._reduce_scatter_validate_mp,
                orig_reduce_scatter,
                mp_config,
            )
            with patch_reduce_scatter(test_reduce_scatter, full_precision_param_dtype):
                scaler = ShardedGradScaler(enabled=enable_sharded_grad_scaler)
                optim = torch.optim.Adam(model.parameters())

                for _ in range(3):
                    inp = torch.randn(
                        3, 10, device="cuda", dtype=full_precision_param_dtype
                    )
                    # Forward pass of LinearMixedPrecision check casting of
                    # inputs, params, buffers.
                    act, *_ = model(
                        (inp, self, model, mp_config, full_precision_param_dtype)
                    )
                    # Buffers should be casted.
                    for buf in model.buffers():
                        if mp_config.buffer_dtype is not None:
                            self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                        else:
                            self.assertEqual(buf.dtype, _BUFFER_ORIG_DTYPE)
                    # p._mp_shard should be freed.
                    if mp_config.param_dtype is not None:
                        self._validate_mp_shard_freed(model)
                    else:
                        # We never should have allocated an _mp_shard.
                        self._validate_no_mp_shard(model)

                    loss = act.sum()
                    loss = scaler.scale(loss)
                    if mp_config.param_dtype is not None:
                        self.assertEqual(loss.dtype, mp_config.param_dtype)
                    else:
                        self.assertEqual(loss.dtype, full_precision_param_dtype)
                    # Will run patched reduce scatter that validates mixed_precision
                    # types in backward.
                    loss.backward()
                    # Buffers stay casted even after backwards.
                    for buf in model.buffers():
                        if mp_config.buffer_dtype is not None:
                            self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                        else:
                            self.assertEqual(buf.dtype, _BUFFER_ORIG_DTYPE)
                    # p._mp_shard should be freed.
                    if mp_config.param_dtype is not None:
                        self._validate_mp_shard_freed(model)
                    else:
                        self._validate_no_mp_shard(model)

                    # Ensure params and grads are in full precision,
                    # as after fwd/backward we maintain full precision shards.
                    for param in model.parameters():
                        self.assertEqual(param.dtype, full_precision_param_dtype)
                        if param.grad is not None:
                            self.assertEqual(
                                param.grad.dtype, full_precision_param_dtype
                            )

                    # Unscale the gradients and step
                    scaler.step(optim)
                    # Update the scale factor
                    scaler.update()

                    # Summon full params should be in full precision
                    with model.summon_full_params(model):
                        # It is not expected for summon_full_params to allocate
                        # a mixed precision shard.
                        if mp_config.param_dtype is not None:
                            self._validate_mp_shard_freed(model)
                        else:
                            self._validate_no_mp_shard(model)
                        params = list(model.parameters())
                        for p in params:
                            self.assertEqual(p.dtype, full_precision_param_dtype)

                        # Note that buffers are cast only once and only restored
                        # to the original buffer dtype in state_dict, so
                        # summon_full_params is not expected to restore buffer
                        # types to their original.
                        named_buffers = dict(model.named_buffers())
                        for v in named_buffers.values():
                            if mp_config.buffer_dtype is not None:
                                self.assertEqual(v.dtype, mp_config.buffer_dtype)
                            else:
                                self.assertEqual(v.dtype, _BUFFER_ORIG_DTYPE)

                    # state_dict should be in full precision
                    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    for name, tensor in state_dict.items():
                        # Parameters and buffers are checkpointed in their
                        # original dtypes, which may be different.
                        if name in named_buffers.keys():
                            self.assertEqual(tensor.dtype, _BUFFER_ORIG_DTYPE)
                        else:
                            self.assertEqual(
                                tensor.dtype,
                                full_precision_param_dtype,
                                f"{name}: {tensor.dtype} vs {full_precision_param_dtype}",
                            )

                    # After state_dict, buffer's dtype should have been restored
                    # to the mixed precision one.
                    for buf in model.buffers():
                        if mp_config.buffer_dtype is not None:
                            self.assertEqual(buf.dtype, mp_config.buffer_dtype)
                        else:
                            self.assertEqual(buf.dtype, _BUFFER_ORIG_DTYPE)


class TestFSDPMixedPrecisionSharded(TestFSDPMixedPrecision):
    @property
    def world_size(self):
        return 2

    def _get_subtest_config(self) -> Dict[str, List[Any]]:
        """Returns a subtest configuration that subtests prefetching settings
        together."""
        return {
            "forward_prefetch": [False, True],
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
        }

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_no_reshard_after_forward(self):
        # Note that we don't exercise all possible different configs so as to
        # not increase test TTS too much.
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            enable_sharded_grad_scaler=False,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_mixed_precision_e2e_full_shard(
        self,
        mp_config,
        cpu_offload,
        full_precision_param_dtype,
        enable_sharded_grad_scaler,
    ):
        self.run_subtests(
            self._get_subtest_config(),
            self._run_test_mixed_precision_e2e,
            mp_config=mp_config,
            cpu_offload=cpu_offload,
            full_precision_param_dtype=full_precision_param_dtype,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            enable_sharded_grad_scaler=enable_sharded_grad_scaler,
        )

    def _test_mixed_precision_embedding_table(self, mp_config):
        # Basic test to ensure int inputs are not casted which would break
        # modules such as embedding tables.
        param_dtype = mp_config.param_dtype or torch.float32
        orig_reduce_scatter = dist.reduce_scatter_tensor
        test_reduce_scatter = partial(
            self._reduce_scatter_validate_mp,
            orig_reduce_scatter,
            mp_config,
        )
        with patch_reduce_scatter(test_reduce_scatter, param_dtype):
            # TODO: `test_mp_embedding_reduce()` fails if we do not wrap the
            # entire `TransformerWithSharedParams` with a single top-level FSDP
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                {"mixed_precision": mp_config},
            )
            fsdp_model = FSDP(model, mixed_precision=mp_config)
            optim = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
            for _ in range(6):
                inp = fsdp_model.module.get_input(torch.device("cuda"))
                # This would fail if we casted integer module inputs such as for
                # embedding tables.
                output = fsdp_model(*inp)
                loss = fsdp_model.module.get_loss(inp, output).cuda()
                self.assertEqual(loss.dtype, param_dtype)
                fsdp_model.module.run_backward(loss)
                optim.step()

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_reduce(self):
        self._test_mixed_precision_embedding_table(
            mp_config=MixedPrecision(reduce_dtype=torch.float16)
        )

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_only_params_and_bufs(self):
        self._test_mixed_precision_embedding_table(
            mp_config=MixedPrecision(
                param_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        )

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_default(self):
        default_mp_config = MixedPrecision(
            param_dtype=torch.float16,
            buffer_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )
        self._test_mixed_precision_embedding_table(mp_config=default_mp_config)

    @skip_if_lt_x_gpu(2)
    def test_mp_embedding_params_and_reduce_diff(self):
        params_and_reduce_different = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
        self._test_mixed_precision_embedding_table(
            mp_config=params_and_reduce_different
        )

    @skip_if_lt_x_gpu(2)
    @skipIfNoTorchVision
    def test_mixed_precision_resnet(self):
        """
        End to end test to ensure mixed precision + auto_wrap works
        for ResNet model.
        """
        resnet_model = torchvision.models.resnet50().cuda()
        resnet_model = nn.SyncBatchNorm.convert_sync_batchnorm(
            resnet_model, process_group=dist.distributed_c10d._get_default_group()
        )
        n_bn = sum(
            1 if isinstance(x, _BatchNorm) else 0 for x in resnet_model.modules()
        )
        inp = torch.ones(1, 3, 1000, 1000, device="cuda")
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        fsdp = FSDP(
            resnet_model,
            auto_wrap_policy=size_based_auto_wrap_policy,
            mixed_precision=mp_config,
        )
        # Batchnorm units should be wrapped individually. Validate this by
        # ensuring there are equal no. of FSDP units that are BN as BN units
        # in original resnet model.
        fsdp_bn = 0
        for module in fsdp.fsdp_modules(fsdp):
            wrapped_module = module.module
            if isinstance(wrapped_module, _BatchNorm):
                fsdp_bn += 1

        self.assertEqual(fsdp_bn, n_bn)
        # Would throw type mismatch issue without mixed precision autowrapping.
        loss = fsdp(inp).sum()
        loss.backward()

    @skip_if_lt_x_gpu(2)
    def test_grads_reduced_precision(self):
        self.run_subtests(
            {"offload_params": [False, True]},
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("convert_sync_bn", [True, False])
    def test_mp_batchnorm(self, convert_sync_bn):
        class BatchNormNet(nn.Module):
            def __init__(self, affine=True):
                super(BatchNormNet, self).__init__()
                self.fc1 = nn.Linear(2, 40, bias=False)
                self.bn = nn.BatchNorm1d(4, affine=affine)
                self.fc2 = nn.Linear(40, 4, bias=False)

            def forward(self, x):
                x = torch.reshape(self.fc1(x), (-1, 4, 10))
                x = self.bn(x)
                x = torch.reshape(x, (-1, 40))
                x = self.fc2(x)
                return F.softmax(x, dim=1)

        def never_wrap_policy(*args, **kwargs):
            return False

        net = BatchNormNet().cuda()
        if convert_sync_bn:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        # FSDP detects that mixed precision + batchnorm will cause issues
        # and thus wrap batchnorm in a distinct FSDP unit that does not
        # use mixed precision.
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="batch norm submodules will be wrapped as separate",
        ):
            model = FSDP(
                net,
                mixed_precision=mp_config,
                auto_wrap_policy=never_wrap_policy,
            )

        bn = model.bn
        self.assertTrue(isinstance(bn, FSDP))
        # policy should not have wrapped any other submodules
        self.assertFalse(isinstance(model.fc1, FSDP))
        self.assertFalse(isinstance(model.fc2, FSDP))
        no_mixed_precision = MixedPrecision()
        self.assertEqual(no_mixed_precision, bn.mixed_precision)
        self.assertNotEqual(no_mixed_precision, model.mixed_precision)

        inp = torch.randn((1, 2), device="cuda")
        # Without FSDP BN mixed precision fix, this would result in
        # RuntimeError: Expected counts to have type Half but got Float
        # for syncBN
        model(inp).sum().backward()


class TestFSDPMixedPrecisionUnsharded(TestFSDPMixedPrecision):
    """
    Smaller test suite for unshared param (i.e. world_size == 1) case.
    """

    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    def test_grads_reduced_precision(self):
        self.run_subtests(
            {"offload_params": [False, True]},
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_no_reshard_after_forward(self):
        # Note that we don't exercise all possible different configs so as to
        # not increase test TTS too much.
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            enable_sharded_grad_scaler=False,
        )

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_e2e_full_shard(self):
        mp = default_mp if not nccl_supports_bf16 else mp_diff_buffer_and_reduce
        self._run_test_mixed_precision_e2e(
            mp_config=mp,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=None,
            forward_prefetch=False,
            full_precision_param_dtype=torch.float64,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            enable_sharded_grad_scaler=False,
        )


instantiate_parametrized_tests(TestFSDPMixedPrecisionSharded)

if __name__ == "__main__":
    run_tests()
