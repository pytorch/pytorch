# Owner(s): ["oncall: distributed"]

import contextlib
import itertools
import os
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
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.swa_utils import AveragedModel
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = skip_but_pass_in_sandcastle_if(
    not HAS_TORCHVISION, "no torchvision"
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

nccl_supports_bf16 = dist.is_nccl_available() and nccl.version() >= (2, 10)

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

    def __init__(self, param_dtype, buffer_name="buffer", run_checks=True):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False).to(param_dtype)
        # Use a configurable buffer name to avoid all submodules sharing the
        # same buffer name, which may hide prefixed vs. unprefixed name bugs
        self.buffer_name = buffer_name
        self.register_buffer(buffer_name, torch.randn((1, 2), dtype=_BUFFER_ORIG_DTYPE))
        self._orig_param_type = param_dtype
        self._orig_buffer_dtype = _BUFFER_ORIG_DTYPE
        self.run_checks = run_checks

    def forward(self, tup):
        inp, cls, fsdp, mp_config, full_precision_param_dtype = tup
        if self.run_checks:
            # Param and input should be the mixed precision type
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
                        # _unshard should have also freed the fp16 shard.
                        # Shard is never allocated if param_dtype mixed precision is not
                        # enabled.
                        if mp_config.param_dtype is not None:
                            cls.assertEqual(0, param._mp_shard.untyped_storage().size())
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

    def _get_simple_nested_model(
        self, param_dtype, run_checks, *fsdp_args, **fsdp_kwargs
    ):
        model = FSDP(
            nn.Sequential(
                FSDP(
                    LinearMixedPrecision(
                        param_dtype, buffer_name="buffer0", run_checks=run_checks
                    ).cuda(),
                    *fsdp_args,
                    **fsdp_kwargs,
                ),
                LinearMixedPrecision(
                    param_dtype, buffer_name="buffer1", run_checks=run_checks
                ).cuda(),
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
                self.assertEqual(0, param._mp_shard.untyped_storage().size())

    def _reduce_scatter_validate_mp(
        self, orig_reduce_scatter, mp_config, should_run_low_prec, *args, **kwargs
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
        for x in kwargs.values():
            if isinstance(x, torch.Tensor):
                tensors.append(x)

        # reduce_dtype has higher priority than param_dtype, because mixed_precision
        # supports overriding param_dtype with reduce_dtype to control the
        # reduction precision. In the case where reduce_dtype == param_dtype
        # this tests that gradients are in the expected precision as well.
        # If reduce_dtype is not specified (is None) we comm. in the param_dtype
        # if that is specified, otherwise full precision dtype.
        if should_run_low_prec:
            expected_dtype = (
                mp_config.reduce_dtype
                if mp_config.reduce_dtype is not None
                else (
                    mp_config.param_dtype
                    if mp_config.param_dtype is not None
                    else _CURRENT_FULL_PRECISION_PARAM_DTYPE
                )
            )
        else:
            expected_dtype = _CURRENT_FULL_PRECISION_PARAM_DTYPE

        for t in tensors:
            self.assertEqual(
                expected_dtype,
                t.dtype,
                f"Expected to reduce in {expected_dtype} but got tensors in {t.dtype}",
            )

        return orig_reduce_scatter(*args, **kwargs)

    def _test_grads_reduced_precision(
        self, offload_params: bool, use_orig_params: bool
    ):
        class MyModel(nn.Module):
            def __init__(self) -> None:
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
            "use_orig_params": use_orig_params,
        }
        m.lin1 = FSDP(m.lin1, **fsdp_kwargs)
        m = FSDP(m, **fsdp_kwargs)
        for _ in range(6):
            inp = torch.ones(1, 10)
            m(inp).sum().backward()
            for param in m.parameters():
                if param.grad is not None:
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
                run_checks=True,
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
                True,
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
            True,
        )
        with patch_reduce_scatter(test_reduce_scatter, param_dtype):
            # TODO: `test_mp_embedding_reduce()` fails if we do not wrap the
            # entire `TransformerWithSharedParams` with a single top-level FSDP
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_BEFORE,
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
            {
                "offload_params": [False, True],
                "use_orig_params": [False, True],
            },
            self._test_grads_reduced_precision,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("convert_sync_bn", [True, False])
    def test_mp_batchnorm(self, convert_sync_bn):
        class BatchNormNet(nn.Module):
            def __init__(self, affine=True):
                super().__init__()
                self.fc1 = nn.Linear(2, 40, bias=False)
                self.bn = nn.BatchNorm1d(4, affine=affine)
                self.fc2 = nn.Linear(40, 4, bias=False)
                self.ln = nn.LayerNorm(4)
                self.fc3 = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                x = torch.reshape(self.fc1(x), (-1, 4, 10))
                x = self.bn(x)
                x = torch.reshape(x, (-1, 40))
                x = self.fc2(x)
                x = self.ln(x)
                x = self.fc3(x)
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
            _module_classes_to_ignore=[_BatchNorm, nn.LayerNorm],
        )
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="These modules will be wrapped as separate FSDP",
        ):
            model = FSDP(
                net,
                mixed_precision=mp_config,
                auto_wrap_policy=never_wrap_policy,
            )

        no_mp = MixedPrecision()
        for mod in [model.ln, model.bn]:
            self.assertTrue(isinstance(mod, FSDP))
            self.assertEqual(no_mp, mod.mixed_precision)
        # policy should not have wrapped any other submodules
        for mod in [model.fc1, model.fc2, model.fc3]:
            self.assertFalse(isinstance(mod, FSDP))

        # Overall mixed precision is still enabled
        self.assertEqual(mp_config, model.mixed_precision)

        inp = torch.randn((1, 2), device="cuda")
        # Without FSDP BN mixed precision fix, this would result in
        # RuntimeError: Expected counts to have type Half but got Float
        # for syncBN
        model(inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_eval_root_cast_inputs(self):
        """
        In a case where root module does not manage FSDP parameters,
        ensure that we don't cast forward inputs which could potentially
        cause a dtype mismatch. Check that FSDP_USE_FULL_PREC_IN_EVAL controls
        this.
        """

        low_prec_dtype = torch.float16

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(5, 5)

            def forward(self, x, expect_use_full_prec_in_eval):
                if expect_use_full_prec_in_eval:
                    assert x.dtype == torch.float32, f"Expected fp32, got {x.dtype}"
                else:
                    assert (
                        x.dtype == low_prec_dtype
                    ), f"Expected {low_prec_dtype}, got {x.dtype}"
                return self.a(x)

        mp_config = MixedPrecision(
            param_dtype=low_prec_dtype,
            reduce_dtype=low_prec_dtype,
            buffer_dtype=low_prec_dtype,
        )

        for use_full_prec_in_eval in [True, False]:
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            m = MyModel().cuda()
            m.a = FSDP(m.a, mixed_precision=mp_config)
            model = FSDP(m, mixed_precision=mp_config)
            model.eval()
            inp = torch.randn(5, 5)
            model(inp, use_full_prec_in_eval).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval(self):
        """
        Tests that eval runs in full precision if FSDP_USE_FULL_PREC_IN_EVAL is set.
        """
        for (
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False]):
            mp_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
                cast_forward_inputs=cast_forward_inputs,
            )
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                {"mixed_precision": mp_config},
            )
            inp = model.get_input(torch.device("cuda"))
            output = model(*inp)
            loss = model.get_loss(inp, output).cuda()
            # Loss should be in fp16
            self.assertEqual(torch.float16, loss.dtype)
            model.run_backward(loss)
            # Grads should be in fp32 as we upcast them
            for p in model.parameters():
                if p.grad is not None:
                    self.assertEqual(torch.float32, p.grad.dtype)

            # Now in eval mode, loss should be fp32 if use_full_prec_in_eval is set.
            model.eval()
            inp = model.get_input(torch.device("cuda"))
            output = model(*inp)
            loss = model.get_loss(inp, output).cuda()
            expected_dtype = torch.float32 if use_full_prec_in_eval else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval_buffers(self):
        """
        Tests that when model.eval() and FSDP_USE_FULL_PREC_IN_EVAL is set,
        buffers are in the full precision.
        """
        for (
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False]):
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            mp_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
                cast_forward_inputs=cast_forward_inputs,
            )
            model_getter = self._get_simple_nested_model
            fsdp_model = model_getter(
                param_dtype=torch.float32,
                run_checks=False,
                mixed_precision=mp_config,
            )

            inp = torch.randn(3, 10, device="cuda")
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            for buf in fsdp_model.buffers():
                self.assertEqual(torch.float16, buf.dtype)

            # model.eval() + forward pass should make the buffers in full prec again
            # Add pre-forward hooks
            def verify_eval_buffer_dtype(module, input):
                expected_dtype = (
                    _BUFFER_ORIG_DTYPE if use_full_prec_in_eval else torch.float16
                )
                for buf in module.buffers():
                    self.assertEqual(expected_dtype, buf.dtype)

            def _get_underlying_module(m):
                return m.module if isinstance(m, FSDP) else m

            hook_handles = []
            hook_handles.append(
                _get_underlying_module(fsdp_model[0]).register_forward_pre_hook(
                    verify_eval_buffer_dtype
                )
            )
            hook_handles.append(
                _get_underlying_module(fsdp_model[1]).register_forward_pre_hook(
                    verify_eval_buffer_dtype
                )
            )

            fsdp_model.eval()
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            for hook_handle in hook_handles:
                hook_handle.remove()

            expected_dtype = (
                _BUFFER_ORIG_DTYPE if use_full_prec_in_eval else torch.float16
            )
            for buf in fsdp_model.buffers():
                self.assertEqual(expected_dtype, buf.dtype)

            # model.train() + forward again should make buffers in fp16
            fsdp_model.train()
            fsdp_model((inp, self, fsdp_model, mp_config, torch.float32))
            for buf in fsdp_model.buffers():
                self.assertEqual(torch.float16, buf.dtype)

    @skip_if_lt_x_gpu(2)
    def test_full_precision_in_eval_comm(self):
        for (
            cast_forward_inputs,
            use_full_prec_in_eval,
        ) in itertools.product([True, False], [True, False]):
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = (
                "1" if use_full_prec_in_eval else "0"
            )
            mp_config = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32,
                cast_forward_inputs=cast_forward_inputs,
                # cast reduction for batchnorm also just in this test, to make
                # validation easier.
                _module_classes_to_ignore=[],
            )
            model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                {"mixed_precision": mp_config},
            )
            # Patch reduce_scatter to add validation for mixed precision types.
            orig_reduce_scatter = dist.reduce_scatter_tensor
            test_reduce_scatter = partial(
                self._reduce_scatter_validate_mp,
                orig_reduce_scatter,
                mp_config,
                not use_full_prec_in_eval,
            )
            model.eval()
            with patch_reduce_scatter(test_reduce_scatter, torch.float32):
                inp = model.get_input(torch.device("cuda"))
                output = model(*inp)
                loss = model.get_loss(inp, output).cuda()
                model.run_backward(loss)

    @skip_if_lt_x_gpu(2)
    def test_input_grads_with_param_mixed_precision(self):
        """
        Tests that input tensors that require gradients do get their gradients
        even after being cast to a low precision (when parameter mixed
        precision is enabled).
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_input_grads_with_param_mixed_precision,
        )

    def _test_input_grads_with_param_mixed_precision(
        self,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        model = nn.Linear(1024, 1024, bias=False)
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            use_orig_params=use_orig_params,
        )
        # Use an input with dtype not equal to the mixed precision
        # `param_dtype` so that it gets cast
        x_float = torch.randn(
            (32, 1024),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        fsdp_model(x_float).sum().backward()
        self.assertTrue(x_float.grad is not None)
        # Check that `x_float` preserves its dtype, meaning that the gradient
        # propagated via `ToCopyBackward0`
        self.assertEqual(x_float.grad.dtype, torch.float32)


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
            {"offload_params": [False, True], "use_orig_params": [False, True]},
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


class IgnoredModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(100, 100)

    def forward(self, x):
        return self.l(x)


class ModelWithIgnoredModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.ignored = IgnoredModule()
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.ignored(self.l1(x)))


class TestFSDPMixedPrecisionIgnoredModules(FSDPTest):
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    def test_mixed_precision_with_ignored_module(self):
        model = ModelWithIgnoredModule().cuda()
        float16 = MixedPrecision(param_dtype=torch.float16)
        model = FSDP(
            model,
            ignored_modules=[model.ignored],
            mixed_precision=float16,
        )

        x = torch.ones(2, 100, device=torch.cuda.current_device())

        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            model(x).sum().backward()


class TestFSDPDifferentSubmodulePrecision(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule(self):
        forward_inputs: Dict[str, nn.Module] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)

        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device="cuda")

        # float16 on one submodule and float32 on everything else
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model)

        fsdp(x).sum().backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule_skip_inputs(self):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).cuda()
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device="cuda")

        # float16 on one submodule and float32 on everything else
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model)

        fsdp(x).sum().backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule_skip_inputs_error(self):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=False)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        x = torch.zeros(2, 100, device="cuda")

        # float16 on one submodule and float32 on everything else
        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model)

        with self.assertRaisesRegex(
            RuntimeError, "mat1 and mat2 must have the same dtype"
        ):
            fsdp(x).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_different_precisions_error(self):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        float32 = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        x = torch.zeros(2, 100, device="cuda")

        # For submodules with different precisions, right now current design
        # does not support the case when the root FSDP instance wraps a submodule
        # that is not the first one executed. Because for that submodule, its inputs
        # (or previous submodule's outputs) have no way to be casted, instead,
        # the root module's inputs are casted upfront before entering
        # root module's forward
        model.c1 = FSDP(model.c1, mixed_precision=float16)
        fsdp = FSDP(model, mixed_precision=float32)
        with self.assertRaisesRegex(
            RuntimeError, "mat1 and mat2 must have the same dtype"
        ):
            fsdp(x).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_different_precisions(self):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        float32 = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device="cuda")

        model.c2 = FSDP(model.c2, mixed_precision=float16)
        fsdp = FSDP(model, mixed_precision=float32)

        fsdp(x).sum().backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_external_inputs(self):
        class ToyModule(nn.Module):
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                self.l = nn.Linear(100, 100)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                return self.l(x)

        class ToyModel(nn.Module):
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                self.l1 = nn.Linear(100, 100)
                self.l2 = ToyModule(forward_inputs)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.forward_inputs["model_input_x"] = x
                y = torch.ones(2, 100, device="cuda", dtype=torch.float32)
                return self.l2(self.l1(x), y)

        forward_inputs: Dict[str, torch.Tensor] = {}

        float16 = MixedPrecision(param_dtype=torch.float16)
        model = ToyModel(forward_inputs).cuda()
        x = torch.zeros(2, 100, device="cuda", dtype=torch.float32)
        model.l2 = FSDP(model.l2, mixed_precision=float16)
        fsdp = FSDP(model, mixed_precision=float16)

        fsdp(x).sum().backward()

        # Inputs are casted in root module in default, inputs of submodules are not
        # explicitly casted, so the external inputs ``y`` of module ``self.l2`` is
        # not casted.
        self.assertEqual(forward_inputs["model_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_y"].dtype, torch.float32)


class TestFSDPTrainEval(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_train_ema_eval_flow(self):
        """
        Tests a train -> EMA update -> eval flow with mixed precision enabled.
        """
        self.run_subtests(
            {
                "sharding_strategy": [
                    # We mainly want to test `SHARD_GRAD_OP` since it surfaced
                    # the original bug of not using the right EMA parameters
                    # for eval, but we also test the others for completeness
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_train_ema_eval_flow,
        )

    def _test_train_ema_eval_flow(self, sharding_strategy: ShardingStrategy):
        class TransformerWithEMA(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                self.module = nn.Transformer(device=device)
                self.ema_module = AveragedModel(
                    nn.Transformer(device=device),
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(),
                    use_buffers=True,
                )

            def forward(self, *args, **kwargs):
                # Use main copy for training and EMA copy for eval
                if self.training:
                    return self.module(*args, **kwargs)
                return self.ema_module(*args, **kwargs)

        device = torch.device("cuda")
        model = TransformerWithEMA(device=device)
        policy = ModuleWrapPolicy(
            {nn.Transformer, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
        )
        mixed_precision = MixedPrecision(param_dtype=torch.float16)
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
        )
        optim = torch.optim.Adam(fsdp_model.module.parameters(), lr=1e-2)
        if self.rank == 0:
            print(fsdp_model)
        torch.manual_seed(1 + self.rank)
        eval_src = torch.randn((8, 1, 512), device=device)
        eval_tgt = torch.randn((16, 1, 512), device=device)
        eval_out_sums: List[torch.Tensor] = []
        # An iteration consists of training forward/backward/optimizer,
        # updating the EMA copy with the main copy, and eval forward
        for _ in range(3):
            fsdp_model.train()
            train_src = torch.randn((8, 4, 512), device=device)
            train_tgt = torch.randn((16, 4, 512), device=device)
            train_out = fsdp_model(train_src, train_tgt)
            train_out.sum().backward()
            optim.step()
            optim.zero_grad()
            with FSDP.summon_full_params(fsdp_model):
                fsdp_model.ema_module.update_parameters(fsdp_model.module)
            fsdp_model.eval()
            with torch.no_grad():
                eval_out = fsdp_model(eval_src, eval_tgt)
            eval_out_sums.append(eval_out.sum())
        # Check that the eval outputs differ from iteration to iteration as a
        # proxy for eval using the correct EMA parameters
        for i in range(len(eval_out_sums) - 1):
            self.assertNotEqual(eval_out_sums[i], eval_out_sums[i + 1])
        self.assertNotEqual(eval_out_sums[0], eval_out_sums[-1])


if __name__ == "__main__":
    run_tests()
