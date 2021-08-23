import torch
import os
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.quantization.quantization as quant
from torch.distributed.algorithms.quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_gloo,
    skip_if_lt_x_gpu,
    requires_nccl,
)
from torch.testing._internal.distributed.distributed_test import (
    apply_hack_for_nccl, BACKEND
)
from torch.testing._internal.common_utils import sandcastle_skip_if

def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, size, size, dtype=dtype).fill_(value).cuda(device_id)

class DistQuantizationTests(MultiProcessTestCase):
    def setUp(self):
        super(DistQuantizationTests, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        super(DistQuantizationTests, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 2

    def _init_multigpu_helper(self):
        """Multigpu tests are designed to simulate the multi nodes with multi
        GPUs on each node. Nccl backend requires equal #GPUs in each process.
        On a single node, all visible GPUs are evenly
        divided to subsets, each process only uses a subset.
        """
        nGPUs = torch.cuda.device_count()
        world_size = dist.get_world_size()
        visible_devices = range(nGPUs)

        if BACKEND == "nccl":
            apply_hack_for_nccl()

        # If rank is lesser than or equal to number of available GPU's
        # then each rank can be mapped to corresponding GPU.
        nGPUs_per_process = 1
        if world_size > nGPUs:
            nGPUs_per_process = nGPUs // world_size
        rank_to_GPU = {
            i: list(
                visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process]
            )
            for i in range(world_size)
        }
        return rank_to_GPU

    @requires_gloo()
    @sandcastle_skip_if(BACKEND != "gloo", "Only gloo backend supports all_gather_fp16")
    def test_all_gather_fp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='gloo')
        device = torch.device(f"cuda:{self.rank}")
        group = list(range(0, self.world_size))
        group_id = dist.group.WORLD
        self._test_all_gather(group, group_id, self.rank, dtype=torch.float32, qtype=DQuantType.FP16)

    @requires_nccl()
    @sandcastle_skip_if(BACKEND != "nccl", "Only nccl backend supports all_to_all_fp16")
    @skip_if_lt_x_gpu(int(os.environ["WORLD_SIZE"]))
    def test_all_to_all_fp16(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='nccl')
        device = torch.device(f"cuda:{self.rank}")
        group = list(range(0, self.world_size))
        group_id = dist.group.WORLD
        rank_to_GPU = self._init_multigpu_helper()
        self._test_all_to_all(
            group,
            group_id,
            self.rank,
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
