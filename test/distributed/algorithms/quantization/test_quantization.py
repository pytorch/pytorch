# Owner(s): ["oncall: distributed"]

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.distributed.algorithms._quantization.quantization as quant
from torch.distributed.algorithms._quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
    requires_gloo,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.common_utils import (
    _restore_fp32_precision,
    _snapshot_fp32_precision,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
)


# The BFP16 quantization kernels (quantization::_FloatToBfloat16Quantized and
# its inverse) are only registered for CPU and CUDA in
# torch/csrc/distributed/c10d/quantization/, so BFP16 collectives cannot run on
# XPU yet.
_BFP16_XPU_SKIP_MSG = (
    "XPU lacks quantization::_FloatToBfloat16Quantized; "
    "see https://github.com/intel/torch-xpu-ops/issues/4941"
)


_PRIOR_FP32_PRECISION: tuple[str, ...] | None = None


def setUpModule():
    global _PRIOR_FP32_PRECISION
    # allow_tf32 writes both the legacy Float32MatmulPrecision enum and the
    # backend-specific fp32_precision, so snapshot and restore all of it.
    _PRIOR_FP32_PRECISION = _snapshot_fp32_precision()
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        _restore_fp32_precision(_PRIOR_FP32_PRECISION)
        _PRIOR_FP32_PRECISION = None


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return (
            torch.empty(size, dtype=dtype)
            .fill_(value)
            .to(torch.device(device_type, device_id))
        )


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

BACKEND = os.environ["BACKEND"]
ACCELERATOR_BACKEND = dist.get_default_backend_for_device(device_type)
if BACKEND in ("gloo", "nccl", "xccl"):

    @unittest.skipIf(
        not dist.is_backend_available(BACKEND)
        or (
            BACKEND == "nccl"
            and torch.cuda.device_count() < int(os.environ["WORLD_SIZE"])
        ),
        "Requested distributed backend or devices unavailable",
    )
    class DistQuantizationTests(MultiProcContinuousTest):
        world_size = int(os.environ["WORLD_SIZE"])
        timeout = dist.distributed_c10d._get_default_timeout(BACKEND)

        @classmethod
        def backend_str(cls):
            return BACKEND

        @property
        def op_timeout_sec(self):
            return 1

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports all_gather_fp16"
        )
        def test_all_gather_fp16(self):
            group = list(range(self.world_size))
            group_id = dist.group.WORLD
            self._test_all_gather(
                group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.FP16
            )

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports all_gather_fp16"
        )
        def test_all_gather_bfp16(self):
            group = list(range(self.world_size))
            group_id = dist.group.WORLD
            self._test_all_gather(
                group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.BFP16
            )

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_but_pass_in_sandcastle_if(
            BACKEND != ACCELERATOR_BACKEND,
            f"Only {ACCELERATOR_BACKEND} backend supports all_to_all_fp16",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm_multiprocess
        def test_all_to_all_fp16(self):
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all(
                group,
                group_id,
                self.rank,
                use_accelerator=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.FP16,
            )
            dist.destroy_process_group(group_id)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_but_pass_in_sandcastle_if(
            BACKEND != ACCELERATOR_BACKEND,
            f"Only {ACCELERATOR_BACKEND} backend supports all_to_all_fp16",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm_multiprocess
        @skip_but_pass_in_sandcastle_if(TEST_XPU, _BFP16_XPU_SKIP_MSG)
        def test_all_to_all_bfp16(self):
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all(
                group,
                group_id,
                self.rank,
                use_accelerator=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.BFP16,
            )
            dist.destroy_process_group(group_id)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_but_pass_in_sandcastle_if(
            BACKEND != ACCELERATOR_BACKEND,
            f"Only {ACCELERATOR_BACKEND} backend supports all_to_all_single_fp16",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_all_to_all_single_fp16(self):
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all_single(
                group,
                group_id,
                self.rank,
                use_accelerator=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.FP16,
            )
            dist.destroy_process_group(group_id)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_but_pass_in_sandcastle_if(
            BACKEND != ACCELERATOR_BACKEND,
            f"Only {ACCELERATOR_BACKEND} backend supports all_to_all_single_bfp16",
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_but_pass_in_sandcastle_if(TEST_XPU, _BFP16_XPU_SKIP_MSG)
        def test_all_to_all_single_bfp16(self):
            group = list(range(self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all_single(
                group,
                group_id,
                self.rank,
                use_accelerator=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.BFP16,
            )
            dist.destroy_process_group(group_id)

        def _test_all_gather(
            self,
            group,
            group_id,
            rank,
            use_accelerator=False,
            rank_to_GPU=None,
            dtype=torch.float,
            qtype=None,
        ):
            for dest in group:
                tensor = _build_tensor([dest + 1, dest + 1], rank, dtype=dtype)
                tensors = [
                    _build_tensor([dest + 1, dest + 1], -1, dtype=dtype) for i in group
                ]
                expected_tensors = [
                    _build_tensor([dest + 1, dest + 1], i, dtype=dtype) for i in group
                ]
                if use_accelerator:
                    tensor = tensor.to(torch.device(device_type, rank_to_GPU[rank][0]))
                    tensors = [
                        t.to(torch.device(device_type, rank_to_GPU[rank][0]))
                        for t in tensors
                    ]
                allgather = quant.auto_quantize(dist.all_gather, qtype, quant_loss=None)
                allgather(tensors, tensor, group=group_id, async_op=False)

                for t1, t2 in zip(tensors, expected_tensors):
                    self.assertEqual(t1, t2)

        def _test_all_to_all(
            self,
            group,
            group_id,
            rank,
            use_accelerator=False,
            rank_to_GPU=None,
            dtype=torch.float,
            qtype=None,
        ):
            if group_id is not None:
                size = len(group)
                in_splits = [i + 1 for i in group]
                in_tensors = [
                    torch.ones([in_splits[i], size], dtype=dtype) * rank
                    for i, _ in enumerate(group)
                ]
                out_tensors = [
                    torch.ones([(rank + 1), size], dtype=dtype) for _ in group
                ]
                expected_tensors = [
                    torch.ones([rank + 1, size], dtype=dtype) * i for i in group
                ]
                if use_accelerator:
                    dev = torch.device(device_type, rank_to_GPU[rank][0])
                    in_tensors = [t.to(dev) for t in in_tensors]
                    expected_tensors = [t.to(dev) for t in expected_tensors]
                    out_tensors = [t.to(dev) for t in out_tensors]
                quantize_alltoall = quant.auto_quantize(
                    dist.all_to_all, qtype, quant_loss=None
                )
                quantize_alltoall(out_tensors, in_tensors, group=group_id)
                for t1, t2 in zip(out_tensors, expected_tensors):
                    self.assertEqual(t1, t2)

        def _test_all_to_all_single(
            self,
            group,
            group_id,
            rank,
            use_accelerator=False,
            rank_to_GPU=None,
            dtype=torch.float,
            qtype=DQuantType.FP16,
        ):
            if group_id is not None:
                size = len(group)
                in_splits = [i + 1 for i in group]
                out_splits = [rank + 1 for _ in group]
                in_tensor = torch.ones([sum(in_splits), size], dtype=dtype) * rank
                out_tensor = torch.ones([(rank + 1) * size, size], dtype=dtype)
                expected_tensor = torch.cat(
                    [torch.ones([rank + 1, size], dtype=dtype) * i for i in group]
                )
                if use_accelerator:
                    dev = torch.device(device_type, rank_to_GPU[rank][0])
                    in_tensor = in_tensor.to(dev)
                    expected_tensor = expected_tensor.to(dev)
                    out_tensor = out_tensor.to(dev)
                    quantize_alltoall_single = quant.auto_quantize(
                        dist.all_to_all_single, qtype, quant_loss=None
                    )
                    quantize_alltoall_single(
                        out_tensor,
                        in_tensor,
                        out_splits=out_splits,
                        in_splits=in_splits,
                        group=group_id,
                    )
                    self.assertEqual(out_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
