# Owner(s): ["oncall: distributed"]

import sys
import tempfile

import torch
from torch import distributed as dist
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, SkipModel
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


_DISTRIBUTED_STATE_DICT_IMPLS = {
    StateDictType.LOCAL_STATE_DICT,
    StateDictType.SHARDED_STATE_DICT,
}


class TestDistributedCheckpoint(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _DISTRIBUTED_STATE_DICT_IMPLS)
    def test_distributed_checkpoint(self, state_dict_type) -> None:
        with enable_wrap(wrapper_cls=FSDP):
            torch.manual_seed(100)
            model = wrap(SkipModel(double_nest=True))
            torch.manual_seed(200)
            new_model = wrap(SkipModel(double_nest=True))

        with FullyShardedDataParallel.summon_full_params(
            model
        ), FullyShardedDataParallel.summon_full_params(new_model):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertNotEqual(params, new_params)

        with tempfile.TemporaryDirectory() as path:
            paths = [path]
            dist.broadcast_object_list(paths)
            path = paths[0]
            writer = FileSystemWriter(path)
            reader = FileSystemReader(path)
            with FSDP.state_dict_type(model, state_dict_type), FSDP.state_dict_type(
                new_model, state_dict_type
            ):
                state_dict = model.state_dict()

            save_state_dict(state_dict, writer)

            with FSDP.state_dict_type(model, state_dict_type), FSDP.state_dict_type(
                new_model, state_dict_type
            ):
                state_dict = new_model.state_dict()
                load_state_dict(state_dict, reader)
                new_model.load_state_dict(state_dict)

        with FullyShardedDataParallel.summon_full_params(
            model
        ), FullyShardedDataParallel.summon_full_params(new_model):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertEqual(params, new_params)

        # TODO: add resharding test case.


instantiate_parametrized_tests(TestDistributedCheckpoint)

if __name__ == "__main__":
    run_tests()
