import torch
import os
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.quantization.quantization as quant
from torch.distributed.algorithms.quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    requires_gloo,
    skip_if_lt_x_gpu,
    requires_nccl,
)
from torch.testing._internal.distributed.distributed_test import (
    DistributedTest, TestDistBackend, BACKEND
)
from torch.testing._internal.common_utils import sandcastle_skip_if

def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, size, size, dtype=dtype).fill_(value).cuda(device_id)

class DistQuantizationTests(TestDistBackend, DistributedTest._DistTestBase):
    def setUp(self):
        super().setUp()
        self._fork_processes()

    @requires_gloo()
    @sandcastle_skip_if(BACKEND != "gloo", "Only gloo backend supports all_gather_fp16")
    def test_all_gather_fp16(self):
        group, group_id, rank = self._init_global_test()
        self._test_all_gather(group, group_id, rank, dtype=torch.float32, qtype=DQuantType.FP16)

    @requires_nccl()
    @sandcastle_skip_if(BACKEND != "nccl", "Only nccl backend supports all_to_all_fp16")
    @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
    def test_all_to_all_fp16(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = self._init_multigpu_helper()
        self._test_all_to_all(
            group,
            group_id,
            rank,
            cuda=True,
            rank_to_GPU=rank_to_GPU,
            dtype=torch.float32,
            qtype=DQuantType.FP16)

    def _test_all_gather(
            self, group, group_id, rank, cuda=False, rank_to_GPU=None, dtype=torch.float, qtype=None):
        for dest in group:
            tensor = _build_tensor(dest + 1, rank, dtype=dtype)
            tensors = [_build_tensor(dest + 1, -1, dtype=dtype) for i in group]
            expected_tensors = [_build_tensor(dest + 1, i, dtype=dtype) for i in group]
            if (qtype is not None):
                allgather = quant.auto_quantize(dist.all_gather, qtype, quant_loss=None)
            else:
                allgather = dist.all_gather
            if cuda:
                tensor = tensor.cuda(rank_to_GPU[rank][0])
                tensors = [t.cuda(rank_to_GPU[rank][0]) for t in tensors]
            if tensors[0].dtype == torch.complex64:
                tensor_shapes = [torch.view_as_real(tensors[0]).shape]
            else:
                tensor_shapes = [tensors[0].shape]
            allgather(tensors, tensor, group=group_id, async_op=False)

            for t1, t2 in zip(tensors, expected_tensors):
                self.assertEqual(t1, t2)

        self._barrier()

    def _test_all_to_all(
        self,
        group,
        group_id,
        rank,
        cuda=False,
        rank_to_GPU=None,
        dtype=torch.float,
        qtype=None
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
            if(qtype is not None):
                quantize_alltoall = quant.auto_quantize(dist.all_to_all, qtype, quant_loss=None)
                quantize_alltoall(out_tensors, in_tensors, group=group_id)
            else:
                dist.all_to_all(out_tensors, in_tensors, group=group_id)
            for t1, t2 in zip(out_tensors, expected_tensors):
                self.assertEqual(t1, t2)
        self._barrier()
