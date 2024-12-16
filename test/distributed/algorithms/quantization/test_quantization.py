# Owner(s): ["oncall: distributed"]
# ruff: noqa: F841

import os
import sys

import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms._quantization.quantization as quant
from torch.distributed.algorithms._quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.common_utils import (
    NO_MULTIPROCESSING_SPAWN,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


torch.backends.cuda.matmul.allow_tf32 = False

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, dtype=dtype).fill_(value).cuda(device_id)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

if NO_MULTIPROCESSING_SPAWN:
    print("Spawn not available, skipping tests.", file=sys.stderr)
    sys.exit(0)

BACKEND = os.environ["BACKEND"]
if BACKEND == "gloo" or BACKEND == "nccl":

    class DistQuantizationTests(MultiProcessTestCase):
        def setUp(self):
            super().setUp()
            self._spawn_processes()
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

        def tearDown(self):
            super().tearDown()
            try:
                os.remove(self.file_name)
            except OSError:
                pass

        @property
        def op_timeout_sec(self):
            return 1

        @property
        def world_size(self):
            return int(os.environ["WORLD_SIZE"])

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports all_gather_fp16"
        )
        def test_all_gather_fp16(self):
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
            )
            device = torch.device(f"cuda:{self.rank}")
            group = list(range(0, self.world_size))
            group_id = dist.group.WORLD
            self._test_all_gather(
                group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.FP16
            )

        @requires_gloo()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "gloo", "Only gloo backend supports all_gather_fp16"
        )
        def test_all_gather_bfp16(self):
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
            )
            device = torch.device(f"cuda:{self.rank}")
            group = list(range(0, self.world_size))
            group_id = dist.group.WORLD
            self._test_all_gather(
                group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.BFP16
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_fp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm_multiprocess
        def test_all_to_all_fp16(self):
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            group = list(range(0, self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.FP16,
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_fp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        @skip_if_rocm_multiprocess
        def test_all_to_all_bfp16(self):
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            group = list(range(0, self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.BFP16,
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_single_fp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_all_to_all_single_fp16(self):
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            group = list(range(0, self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all_single(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.FP16,
            )

        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(
            BACKEND != "nccl", "Only nccl backend supports all_to_all_single_bfp16"
        )
        @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
        def test_all_to_all_single_bfp16(self):
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            group = list(range(0, self.world_size))
            group_id = dist.new_group(range(self.world_size))
            rank_to_GPU = init_multigpu_helper(self.world_size, BACKEND)
            self._test_all_to_all_single(
                group,
                group_id,
                self.rank,
                cuda=True,
                rank_to_GPU=rank_to_GPU,
                dtype=torch.float32,
                qtype=DQuantType.BFP16,
            )

        def _test_all_gather(
            self,
            group,
            group_id,
            rank,
            cuda=False,
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
                if cuda:
                    tensor = tensor.cuda(rank_to_GPU[rank][0])
                    tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
                if tensors[0].dtype == torch.complex64:
                    tensor_shapes = [torch.view_as_real(tensors[0]).shape]
                else:
                    tensor_shapes = [tensors[0].shape]
                allgather = quant.auto_quantize(dist.all_gather, qtype, quant_loss=None)
                allgather(tensors, tensor, group=group_id, async_op=False)

                for t1, t2 in zip(tensors, expected_tensors):
                    self.assertEqual(t1, t2)

        def _test_all_to_all(
            self,
            group,
            group_id,
            rank,
            cuda=False,
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
                if cuda:
                    in_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in in_tensors]
                    expected_tensors = [
                        t.cuda(rank_to_GPU[rank][0]) for t in expected_tensors
                    ]
                    out_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in out_tensors]
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
            cuda=False,
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
                if cuda:
                    rank_to_GPU = rank_to_GPU[rank][0]
                    in_tensor = in_tensor.cuda(rank_to_GPU)
                    expected_tensor = expected_tensor.cuda(rank_to_GPU)
                    out_tensor = out_tensor.cuda(rank_to_GPU)
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
