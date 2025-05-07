# Owner(s): ["oncall: distributed"]
import copy
import io

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._state_dict_utils import (
    _check_state_dict_similarity,
    _copy_state_dict,
    _create_cpu_state_dict,
    _distribute_tensors,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
)
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Shard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


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
        for _ in range(10):
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

            _copy_state_dict(state_dict, cpu_state_dict)

            # Verify if _copy_state_dict works
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
        cpu_state_dict = _create_cpu_state_dict(
            state_dict, share_memory=True, pin_memory=True
        )
        _verify(cpu_state_dict)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_state_dict_util_distribute_tensors(self):
        even_tensor = torch.randn(self.world_size, 2)
        uneven_tensor = torch.randn(1, 2)

        mesh = init_device_mesh("cuda", mesh_shape=(self.world_size,))
        even_dtensor = distribute_tensor(
            torch.randn(self.world_size, 2), mesh, [Shard(0)]
        )
        uneven_dtensor = distribute_tensor(torch.randn(1, 2), mesh, [Shard(0)])

        # the dtensor and tensor are different before _distribute_tensors is called.
        local_state_dict = {
            "even": [even_dtensor, even_tensor],
            "uneven": [uneven_dtensor, uneven_tensor],
        }
        ref_local_state_dict = copy.deepcopy(local_state_dict)
        keys = ["even", "uneven"]

        _distribute_tensors(local_state_dict, keys, self.device_type)
        for local_v, ref_v in zip(
            local_state_dict.values(), ref_local_state_dict.values()
        ):
            self.assertEqual(local_v.size(), ref_v[0].size())
            self.assertEqual(local_v.stride(), ref_v[0].stride())
            self.assertNotEqual(
                local_v_full_tensor := local_v.full_tensor(), ref_v[0].full_tensor()
            )
            self.assertEqual(local_v_full_tensor, ref_v[1])

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_cpu_offload_for_dtensor(self):
        device_mesh = init_device_mesh("cuda", mesh_shape=(self.world_size,))
        sd = {
            "k": DTensor.from_local(
                torch.ones(8, 8, device="cuda"), device_mesh, [Shard(0)]
            )
        }
        cpu_sd = _create_cpu_state_dict(sd)

        self.assertTrue(isinstance(cpu_sd["k"], DTensor))
        self.assertTrue(isinstance(sd["k"], DTensor))
        self.assertTrue(cpu_sd["k"].is_cpu)
        self.assertTrue(cpu_sd["k"]._local_tensor.is_cpu)
        self.assertFalse(sd["k"].is_cpu)
        self.assertFalse(sd["k"]._local_tensor.is_cpu)

        self.assertFalse(torch.equal(sd["k"].cpu(), cpu_sd["k"]))
        _copy_state_dict(sd, cpu_sd, non_blocking=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(sd["k"].cpu(), cpu_sd["k"]))
        sd["k"] += 1
        self.assertFalse(torch.equal(sd["k"].cpu(), cpu_sd["k"]))
        _copy_state_dict(sd, cpu_sd, non_blocking=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(sd["k"].cpu(), cpu_sd["k"]))


if __name__ == "__main__":
    run_tests()
