# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch import distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
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


class TestPureFP16(FSDPTest):
    @property
    def world_size(self):
        # Test fails due to inaccuracies when using more than 4 GPUs
        return min(4, super().world_size)

    @skip_if_lt_x_gpu(2)
    def test_pure_fp16_training(self):
        """Tests pure FP16 training, including when the parameter's dtype is
        changed after FSDP initialization and before training."""
        self.run_subtests(
            {
                "cpu_offload": [
                    CPUOffload(offload_params=True),
                    CPUOffload(offload_params=False),
                ]
            },
            self._test_pure_fp16_training,
        )

    def _test_pure_fp16_training(self, cpu_offload: CPUOffload):
        self._test_fsdp_parity(
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cuda_init_mode=CUDAInitMode.CUDA_BEFORE,
            # Run one iteration to avoid NaN without a gradient scaler
            num_iters=1,
            cpu_offload=cpu_offload,
            use_pure_fp16=True,
        )

    @skip_if_lt_x_gpu(2)
    def test_fp16_dtypes(self):
        """
        Tests that both user-facing parameter/gradient dtypes and internal
        saved dtype attributes are as expected when using an FP16 model
        possibly with explicit mixed precision enabled.
        """
        self.run_subtests(
            {
                "to_half_before_fsdp_init": [False, True],
                "use_orig_params": [False, True],
                "mixed_precision": [
                    MixedPrecision(),
                    MixedPrecision(
                        param_dtype=torch.float16,
                        reduce_dtype=torch.float32,
                    ),
                    MixedPrecision(
                        param_dtype=torch.float32,
                    ),
                ],
            },
            self._test_fp16_dtypes,
        )

    def _test_fp16_dtypes(
        self,
        to_half_before_fsdp_init: bool,
        use_orig_params: bool,
        mixed_precision: MixedPrecision,
    ):
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_NEVER,
            {},
        )
        fsdp_kwargs = {
            "use_orig_params": use_orig_params,
            "device_id": torch.cuda.current_device(),
            "mixed_precision": mixed_precision,
        }
        if to_half_before_fsdp_init:
            model = model.half()
        fsdp_model = FSDP(model, **fsdp_kwargs)
        if not to_half_before_fsdp_init:
            fsdp_model = fsdp_model.half()
        for param in fsdp_model.parameters():
            self.assertEqual(param.dtype, torch.float16)
        inp = tuple(
            t.half() if torch.is_tensor(t) else t
            for t in fsdp_model.module.get_input(torch.device("cuda"))
        )
        out = fsdp_model(*inp)
        out.sum().backward()

        # Check handle dtype attributes
        for handle in traversal_utils._get_fsdp_handles(fsdp_model):
            self.assertEqual(handle.flat_param.dtype, torch.float16)
            self.assertEqual(handle.flat_param.grad.dtype, torch.float16)
            self.assertEqual(handle._orig_param_dtype, torch.float16)
            # Specifying `mixed_precision` takes precedence over the model
            # dtype for both `param_dtype` and `reduce_dtype`
            if mixed_precision.param_dtype is not None:
                self.assertEqual(
                    handle._fwd_bwd_param_dtype, mixed_precision.param_dtype
                )
            else:
                self.assertEqual(handle._fwd_bwd_param_dtype, torch.float16)
            if mixed_precision.reduce_dtype is not None:
                self.assertEqual(handle._reduce_dtype, mixed_precision.reduce_dtype)
            elif (
                mixed_precision.reduce_dtype is None
                and mixed_precision.param_dtype is not None
            ):
                # Special case: infer reduce dtype from parameter dtype
                self.assertEqual(handle._reduce_dtype, mixed_precision.param_dtype)
            else:
                self.assertEqual(handle._reduce_dtype, torch.float16)

            # Check parameter/gradient dtypes
            for param in fsdp_model.parameters():
                self.assertEqual(param.dtype, torch.float16)
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)


instantiate_parametrized_tests(TestPureFP16)

if __name__ == "__main__":
    run_tests()
