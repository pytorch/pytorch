# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.algorithms._quantization.quantization as quant
from torch.distributed.algorithms._quantization.quantization import DQuantType
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    onlyAccelerator,
    requires_capabilities,
)
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    init_multigpu_helper,
    requires_gloo,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


_PRIOR_FP32_PRECISION: str | None = None


def setUpModule():
    global _PRIOR_FP32_PRECISION
    # Snapshot fp32_precision (not allow_tf32) so tearDownModule restores the
    # exact original; writing allow_tf32 back can't reproduce the "none" default.
    _PRIOR_FP32_PRECISION = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        torch.backends.cuda.matmul.fp32_precision = _PRIOR_FP32_PRECISION
        _PRIOR_FP32_PRECISION = None


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, dtype=dtype, device=device_id).fill_(value)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class DistQuantizationTests(DistributedTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        if self.device_type == "cuda":
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    @requires_gloo()
    def test_all_gather_fp16(self, device):
        if self.backend(device) != "gloo":
            self.skipTest("Only gloo backend supports all_gather_fp16")
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        group = list(range(self.world_size))
        group_id = dist.group.WORLD
        self._test_all_gather(
            group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.FP16
        )

    @requires_gloo()
    def test_all_gather_bfp16(self, device):
        if self.backend(device) != "gloo":
            self.skipTest("Only gloo backend supports test_all_gather_bfp16")
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        group = list(range(self.world_size))
        group_id = dist.group.WORLD
        self._test_all_gather(
            group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.BFP16
        )

    @requires_capabilities(Capability.distributed.backend)
    @onlyAccelerator
    @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
    @skip_if_rocm_multiprocess
    def test_all_to_all_fp16(self, device):
        backend = self.backend(device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, backend)
        self._test_all_to_all(
            group,
            group_id,
            self.rank,
            use_accelerator=True,
            rank_to_GPU=rank_to_GPU,
            dtype=torch.float32,
            qtype=DQuantType.FP16,
        )

    @requires_capabilities(Capability.distributed.backend)
    @onlyAccelerator
    @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
    @skip_if_rocm_multiprocess
    def test_all_to_all_bfp16(self, device):
        backend = self.backend(device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, backend)
        self._test_all_to_all(
            group,
            group_id,
            self.rank,
            use_accelerator=True,
            rank_to_GPU=rank_to_GPU,
            dtype=torch.float32,
            qtype=DQuantType.BFP16,
        )

    @requires_capabilities(Capability.distributed.backend)
    @onlyAccelerator
    @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
    def test_all_to_all_single_fp16(self, device):
        backend = self.backend(device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, backend)
        self._test_all_to_all_single(
            group,
            group_id,
            self.rank,
            use_accelerator=True,
            rank_to_GPU=rank_to_GPU,
            dtype=torch.float32,
            qtype=DQuantType.FP16,
        )

    @requires_capabilities(Capability.distributed.backend)
    @onlyAccelerator
    @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
    def test_all_to_all_single_bfp16(self, device):
        backend = self.backend(device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        group = list(range(self.world_size))
        group_id = dist.new_group(range(self.world_size))
        rank_to_GPU = init_multigpu_helper(self.world_size, backend)
        self._test_all_to_all_single(
            group,
            group_id,
            self.rank,
            use_accelerator=True,
            rank_to_GPU=rank_to_GPU,
            dtype=torch.float32,
            qtype=DQuantType.BFP16,
        )

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
                dev = rank_to_GPU[rank][0]
                tensor = tensor.to(f"{self.device_type}:{dev}")
                tensors = [t.to(f"{self.device_type}:{dev}") for t in tensors]
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
            out_tensors = [torch.ones([(rank + 1), size], dtype=dtype) for _ in group]
            expected_tensors = [
                torch.ones([rank + 1, size], dtype=dtype) * i for i in group
            ]
            if use_accelerator:
                dev = rank_to_GPU[rank][0]
                target_device = torch.device(self.device_type, dev)
                in_tensors = [t.to(target_device) for t in in_tensors]
                expected_tensors = [t.to(target_device) for t in expected_tensors]
                out_tensors = [t.to(target_device) for t in out_tensors]
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
                dev = rank_to_GPU[rank][0]
                in_tensor = in_tensor.to(f"{self.device_type}:{dev}")
                expected_tensor = expected_tensor.to(f"{self.device_type}:{dev}")
                out_tensor = out_tensor.to(f"{self.device_type}:{dev}")
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


instantiate_device_type_tests(DistQuantizationTests, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
