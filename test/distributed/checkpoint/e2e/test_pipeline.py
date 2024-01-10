# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
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


DIM = 500


class PipelineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.layer4 = nn.Linear(DIM, DIM)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = self.relu(self.layer1(batch))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        return x


class TestPipeline(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    def save_with_pipeline(self, pipeline_dir: str) -> None:
        with torch.device("meta"):
            model = PipelineModel()

        pipeline_modules = [model.layer1, model.layer2, model.layer3, model.layer4]

        # Materialize the model
        submodule = pipeline_modules[self.rank]
        submodule.to_empty(device=torch.device("cuda"))
        # submodule.reset_parameters()
        optim = torch.optim.Adam(submodule.parameters(), lr=1e-3)

        # Ignore the training as we don't have a real pipeline parallelism.

        # Save state_dict
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        dcp.save_state_dict(
            state_dict=saved_state_dict,
            storage_writer=dcp.FileSystemWriter(pipeline_dir),
        )

    def load_with_fsdp(self, pipeline_dir: str) -> None:
        model = FSDP(PipelineModel().cuda())
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Load the checkpoint
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        dcp.load_state_dict(
            {"model": model_state_dict, "optim": optim_state_dict},
            storage_reader=dcp.FileSystemReader(pipeline_dir),
        )
        set_state_dict(
            model,
            optimizers=optim,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
        )

    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_pipeline(self) -> None:
        self.assertTrue(os.path.exists(self.temp_dir))
        pipeline_dir = os.path.join(self.temp_dir, "pipeline")
        if self.rank == 0:
            os.mkdir(pipeline_dir)
        os.sync()
        dist.barrier()
        self.assertTrue(os.path.exists(pipeline_dir))
        self.save_with_pipeline(pipeline_dir)
        self.load_with_fsdp(pipeline_dir)


if __name__ == "__main__":
    run_tests()
