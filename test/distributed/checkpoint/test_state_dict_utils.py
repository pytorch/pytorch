# Owner(s): ["oncall: distributed"]
import copy
import io

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from torch.distributed._state_dict_utils import (
    _check_state_dict_similarity,
    _create_cpu_state_dict,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
)
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

import pytest


class TestStateDictUtils(DTensorTestBase):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_gather_state_dict_dtensor(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        torch.random.manual_seed(dist.get_rank())
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        state_dict = {"dtensor": dist_tensor}

        gathered_state_dict = _gather_state_dict(state_dict)
        expected_gathered_dtensor = funcol.all_gather_tensor(
            dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )
        self.assertEqual(expected_gathered_dtensor, gathered_state_dict["dtensor"])
        self.assertTrue(gathered_state_dict["dtensor"].is_cuda)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_with_cpu_and_ranks_only(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        torch.random.manual_seed(dist.get_rank())
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        state_dict = {"dtensor": dist_tensor}

        gathered_state_dict = _gather_state_dict(
            state_dict, cpu_offload=True, ranks_only=(0, 2)
        )
        expected_gathered_dtensor = funcol.all_gather_tensor(
            dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )
        if dist.get_rank() in (0, 2):
            self.assertEqual(expected_gathered_dtensor, gathered_state_dict["dtensor"])
            self.assertFalse(gathered_state_dict["dtensor"].is_cuda)
        else:
            self.assertEqual(gathered_state_dict, {})

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_cpu_and_ranks_only(self):
        device = torch.device("cuda")
        state_dict = {
            "tensor1": torch.arange(10, device=device),
            "tensor2": torch.ones(10, device=device),
        }

        cpu_state_dict = _offload_state_dict_to_cpu(state_dict, ranks_only=(0, 2))
        if dist.get_rank() in (0, 2):
            for v in cpu_state_dict.values():
                self.assertFalse(v.is_cuda)
            self.assertEqual(cpu_state_dict["tensor1"], torch.arange(10))
            self.assertEqual(cpu_state_dict["tensor2"], torch.ones(10))
        else:
            self.assertEqual(cpu_state_dict, {})

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_complicated_dict(self):
        def create_dtensor():
            device_mesh = self.build_device_mesh()
            shard_spec = [Shard(0)]
            torch.random.manual_seed(dist.get_rank())
            local_tensor = torch.randn(3, 3, 3)
            dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
            tensor = funcol.all_gather_tensor(
                dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )
            return tensor, dist_tensor

        ltensor, ldtensor = [], []
        for i in range(10):
            tensor, dtensor = create_dtensor()
            ltensor.append(tensor)
            ltensor.append(torch.ones(10, device=torch.device("cuda")))
            ldtensor.append(dtensor)
            ldtensor.append(torch.ones(10, device=torch.device("cuda")))

        tensor, dtensor = create_dtensor()
        dist_state_dict = {
            "local": dtensor,
            "list": ldtensor,
            "arange": torch.arange(10, device=torch.device("cuda")),
        }
        state_dict = {
            "local": tensor,
            "list": ltensor,
            "arange": torch.arange(10, device=torch.device("cuda")),
        }
        self.assertEqual(state_dict, _gather_state_dict(dist_state_dict))

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_create_cpu_state_dict(self):
        device = torch.device("cuda")
        buffer = io.BytesIO()
        torch.save(torch.ones(10), buffer)
        buffer.seek(0)
        state_dict = {
            "tensor1": torch.arange(10, device=device),
            "tensor2": torch.ones(10, device=device),
            "non_tensor_bytes_io": copy.deepcopy(buffer),
            "non_tensor_bytes": buffer.read(),
            "step": torch.tensor(7, dtype=torch.float),
            "lr": 1.5,
            "nested": {"list": [1, 2, 3, 4]},
        }

        def _verify(cpu_state_dict):
            # Verify the correctness of _check_state_dict_similarity()
            self.assertTrue(_check_state_dict_similarity(state_dict, cpu_state_dict))
            tensor1 = cpu_state_dict["tensor1"]
            cpu_state_dict["tensor1"] = torch.arange(11)
            self.assertFalse(_check_state_dict_similarity(state_dict, cpu_state_dict))
            cpu_state_dict["tensor1"] = tensor1

            cpu_state_dict = _offload_state_dict_to_cpu(
                state_dict, cpu_offload_state_dict=cpu_state_dict, type_check=True
            )

            # Verify if _offload_state_dict_to_cpu works
            for v in cpu_state_dict.values():
                if isinstance(v, torch.Tensor):
                    self.assertFalse(v.is_cuda)
            self.assertEqual(cpu_state_dict["tensor1"], torch.arange(10))
            self.assertEqual(cpu_state_dict["tensor2"], torch.ones(10))
            buffer.seek(0)
            cpu_state_dict["non_tensor_bytes_io"].seek(0)
            self.assertEqual(
                cpu_state_dict["non_tensor_bytes_io"].read(), buffer.read()
            )
            buffer.seek(0)
            self.assertEqual(cpu_state_dict["non_tensor_bytes"], buffer.read())
            self.assertEqual(cpu_state_dict["lr"], 1.5)
            self.assertEqual(cpu_state_dict["step"], 7)
            self.assertEqual(cpu_state_dict["nested"], {"list": [1, 2, 3, 4]})

        cpu_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True)
        _verify(cpu_state_dict)
        cpu_state_dict = _create_cpu_state_dict(state_dict, share_memory=True)
        _verify(cpu_state_dict)
        cpu_state_dict = _create_cpu_state_dict(state_dict, share_memory=True, pin_memory=True)
        _verify(cpu_state_dict)

    @pytest.mark.parametrize("use_shared", [True, False])
    def test_async_issue_repro(use_shared):
        # from torch.distributed.checkpoint._state_dict_utils import (
        #     _offload_state_dict_to_cpu,
        #     _create_cpu_state_dict,
        # )
        import timeit
        import torch
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("torch_zeros"):
                tensor = torch.zeros(50000, 50000, device="cuda")
            state_dict = {"a": tensor}
            with record_function("_create_cpu_state_dict"):
                cache = _create_cpu_state_dict(state_dict, share_memory=use_shared, pin_memory=not use_shared)
            with record_function("_offload_state_dict_to_cpu-sync-1st"):
                copy = _offload_state_dict_to_cpu(state_dict, cpu_offload_state_dict=cache)
            with record_function("_offload_state_dict_to_cpu-sync-2nd"):
                copy = _offload_state_dict_to_cpu(state_dict, cpu_offload_state_dict=cache)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                with record_function("_offload_state_dict_to_cpu-async-1st"):
                    copy = _offload_state_dict_to_cpu(state_dict, cpu_offload_state_dict=cache, cpu_offload_sync=False)
            with record_function("stream.synchronize-1st"):
                stream.synchronize()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                with record_function("_offload_state_dict_to_cpu-async-2nd"):
                    copy = _offload_state_dict_to_cpu(state_dict, cpu_offload_state_dict=cache, cpu_offload_sync=False)
            with record_function("stream.synchronize-2nd"):
                stream.synchronize()

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

if __name__ == "__main__":
    run_tests()
