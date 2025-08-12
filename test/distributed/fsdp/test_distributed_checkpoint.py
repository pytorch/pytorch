# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, SkipModel
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


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
        if torch.cuda.is_available():
            gpu_cnt = torch.cuda.device_count()
            if gpu_cnt < 2:
                return gpu_cnt
        return 2

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    @parametrize("state_dict_type", _DISTRIBUTED_STATE_DICT_IMPLS)
    def test_distributed_checkpoint(self, state_dict_type) -> None:
        with enable_wrap(wrapper_cls=FSDP):
            torch.manual_seed(100)
            model = wrap(SkipModel(double_nest=True))
            torch.manual_seed(200)
            new_model = wrap(SkipModel(double_nest=True))

        with (
            FullyShardedDataParallel.summon_full_params(model),
            FullyShardedDataParallel.summon_full_params(new_model),
        ):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertNotEqual(params, new_params)

        writer = FileSystemWriter(self.temp_dir)
        reader = FileSystemReader(self.temp_dir)
        with (
            FSDP.state_dict_type(model, state_dict_type),
            FSDP.state_dict_type(new_model, state_dict_type),
        ):
            state_dict = model.state_dict()

        save(state_dict, writer)

        with (
            FSDP.state_dict_type(model, state_dict_type),
            FSDP.state_dict_type(new_model, state_dict_type),
        ):
            state_dict = new_model.state_dict()
            load(state_dict, reader)
            new_model.load_state_dict(state_dict)

        with (
            FullyShardedDataParallel.summon_full_params(model),
            FullyShardedDataParallel.summon_full_params(new_model),
        ):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertEqual(params, new_params)

        # TODO: add resharding test case.


devices = ("cuda", "hpu")
instantiate_device_type_tests(TestDistributedCheckpoint, globals(), only_for=devices)
if __name__ == "__main__":
    run_tests()
