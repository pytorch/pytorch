from torch.distributed._checkpoint.state_dict_saver import save_state_dict
from torch.distributed._checkpoint.storage_writer import FileSystemWriter
from torch.distributed._checkpoint.state_dict_loader import load_state_dict
from torch.distributed._checkpoint.storage_reader import FileSystemReader
from torch.testing._internal.common_utils import (
    TestCase,

)
from typing import Dict
import torch
import shutil

from torch.testing._internal.distributed._sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.distributed import _sharded_tensor

from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
)
from torch.distributed._sharded_tensor import (
    state_dict_hook,
    ShardedTensor,
)
import torch.distributed as dist


def assert_state_dict_equal(self, state_dict_1: Dict[str, torch.Tensor], state_dict_2: Dict[str, torch.Tensor]):
    self.assertEqual(len(state_dict_1), len(state_dict_2), "state_dict must be the same size")
    self.assertEqual(set(state_dict_1.keys()), set(state_dict_2.keys()), "state_dict keys do not match")

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, torch.Tensor):
            self.assertTrue(torch.equal(value_1, value_2), f"Key {key}'s tensor does not match")
        elif isinstance(value_1, ShardedTensor):
            for local_shard_1, local_shard_2 in zip(value_1.local_shards(), value_2.local_shards()):
                self.assertTrue(torch.equal(local_shard_1.tensor, local_shard_1.tensor), f"Key {key}'s shard does not match")

    return True



class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# The ShardedModels are borrowed from ../_sharded_tensor/test_sharded_tensor.py
class MyShardedModel2(torch.nn.Module):
    def __init__(self, spec=None, group=None):
        super(MyShardedModel2, self).__init__()
        if spec is not None:
            self.sharded_tensor2 = _sharded_tensor.empty(spec, 10, 20, process_group=group, init_rrefs=False)
        else:
            self.sharded_tensor2 = None
        self.random_tensor2 = torch.nn.Parameter(torch.rand(2, 2))


class MyShardedModel1(torch.nn.Module):
    def __init__(self, spec=None, group=None):
        super(MyShardedModel1, self).__init__()
        if spec is not None:
            self.sharded_tensor1 = _sharded_tensor.empty(spec, 10, 20, process_group=group, init_rrefs=False)
        else:
            self.sharded_tensor1 = None
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        self.submodule = MyShardedModel2(spec, group)

class TestStateDictSaveLoad(TestCase):
    def test_read_write_only_tensor(self):
        state_dict_to_save = TestModule().state_dict()
        base_dir = "/tmp/_test_state_dict_save_load_torch_tensor_"
        shutil.rmtree(base_dir, ignore_errors=True)

        fs_writer = FileSystemWriter(base_folder_name=base_dir)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # Genrate a new modle
        state_dict_to_load_to = TestModule().state_dict()


        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Load from file without any resharding
        fs_reader = FileSystemReader(base_folder_name=base_dir)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

class TestStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_read_write_shard_tenosr(self):
        base_dir = "/tmp/_test_state_dict_save_load_shared_tensor_"
        shutil.rmtree(base_dir, ignore_errors=True)

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        model_to_save = MyShardedModel1(spec)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(base_folder_name=base_dir)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(base_folder_name=base_dir)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()
