# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist

from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class FsdpOptimStateCheckpoint(DTensorTestBase):
    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_distributed_tensor_planner(self) -> None:
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(torch.nn.Linear(8, 8, device="meta"))
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optim),
            }

            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
                planner=DefaultSavePlanner(),
            )

        # now load the model and ensure the values are the same
        model_2 = FSDP(torch.nn.Linear(8, 8, device="meta"))
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertNotEqual(model.weight, model_2.weight)
                self.assertNotEqual(model.bias, model_2.bias)

        # Adam lazily creates its state
        self.assertEqual(0, len(optim_2.state))

        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model_2.state_dict(),
                # cannot load the optimizer together with the model
            }

            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                planner=DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                model_2, optim_2, optim_state["optim"]
            )
            optim_2.load_state_dict(flattened_osd)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertEqual(model.weight, model_2.weight)
                self.assertEqual(model.bias, model_2.bias)

        def opt_at(opt, idx):
            return list(iter(opt.state.values()))[idx]

        # Adam lazily creates its state
        self.assertEqual(
            opt_at(optim, 0)["exp_avg"], opt_at(optim_2, 0)["exp_avg"]
        )
        self.assertEqual(
            opt_at(optim, 0)["exp_avg_sq"], opt_at(optim_2, 0)["exp_avg_sq"]
        )


if __name__ == "__main__":
    run_tests()
