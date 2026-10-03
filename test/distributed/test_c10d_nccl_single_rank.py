# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class ProcessGroupNCCLSingleRankTest(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("collective", ["list", "single", "coalesced"])
    @parametrize("tensor_factor", [False, True])
    @parametrize("async_op", [False, True])
    def test_reduce_scatter_premul_sum(
        self, device, dtype, collective, tensor_factor, async_op
    ):
        dist.init_process_group(
            "nccl-legacy",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
            device_id=torch.device(device),
        )
        try:
            # Also exercise the size that triggered NCCL issue #1950.
            input_tensor = torch.randn(8193, dtype=dtype, device=device)
            original = input_tensor.clone()
            output = torch.empty_like(input_tensor)
            factor = (
                torch.tensor([0.5], dtype=dtype, device=device)
                if tensor_factor
                else 0.5
            )
            op = dist._make_nccl_premul_sum(factor)

            if collective == "coalesced":
                with dist._coalescing_manager(async_ops=async_op) as cm:
                    dist.reduce_scatter_single(output, input_tensor, op=op)
                if async_op:
                    cm.wait()
            else:
                if collective == "list":
                    work = dist.reduce_scatter(
                        output, [input_tensor], op=op, async_op=async_op
                    )
                else:
                    work = dist.reduce_scatter_single(
                        output, input_tensor.view(1, -1), op=op, async_op=async_op
                    )
                if work is not None:
                    work.wait()

            self.assertEqual(output, original * 0.5)
            self.assertEqual(input_tensor, original)
        finally:
            dist.destroy_process_group()


instantiate_device_type_tests(
    ProcessGroupNCCLSingleRankTest, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
