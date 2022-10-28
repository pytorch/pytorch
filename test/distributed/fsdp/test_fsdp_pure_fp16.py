# Owner(s): ["oncall: distributed"]

import sys

from torch import distributed as dist
from torch.distributed.fsdp import CPUOffload
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
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


class TestPureFP16(FSDPTest):
    @property
    def world_size(self):
        # Test fails due to inaccuracies when using more than 5 GPUs
        return min(5, super().world_size)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    def test_pure_fp16(self, cpu_offload: CPUOffload):
        """Tests pure FP16 training, including when the parameter's dtype is
        changed after FSDP initialization and before training."""
        self._test_fsdp_parity(
            NestedWrappedModule,
            FSDPInitMode.RECURSIVE,
            cuda_init_mode=CUDAInitMode.CUDA_AFTER,
            # Run one iteration to avoid NaN without a gradient scaler
            num_iters=1,
            cpu_offload=cpu_offload,
            use_pure_fp16=True,
        )


instantiate_parametrized_tests(TestPureFP16)

if __name__ == "__main__":
    run_tests()
