# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
from torch.distributed._tensor import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
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


DIM = 500


class PreTrainedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.sequential = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
        self.module_list = nn.ModuleList([nn.Linear(DIM, DIM), nn.ReLU()])
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = self.relu(self.layer1(batch))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sequential(x)
        x = self.module_list[1](self.module_list[0](x))
        return x


class FineTuningModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain = PreTrainedModel()
        for p in self.pretrain.parameters():
            p.requires_grad = False

        self.layer1 = nn.Linear(DIM, DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, DIM)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = self.relu(self.pretrain(batch))
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


class TestFineTuning(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"

    def pretrain(self, pretrain_dir: str) -> None:
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        model = PreTrainedModel().cuda()
        model = FSDP(model, device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
        for i in range(3):
            batch = torch.rand(32, DIM, device="cuda")
            loss = model(batch).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Save state_dict
        model_state_dict, optim_state_dict = get_state_dict(model, optimizers=optim)
        saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        dist_cp.save_state_dict(
            state_dict=saved_state_dict,
            storage_writer=dist_cp.FileSystemWriter(pretrain_dir),
        )

    def finetune(self, pretrain_dir: str, finetune_dir: str) -> None:
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        model = FineTuningModel().cuda()
        # TODO: make the parallelism more complicated, e.g., using 2D + DDP.
        model = FSDP(model, use_orig_params=True, device_mesh=device_mesh)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate that the fine tuning restart after 3 iterations
        for i in range(2):
            # Load pretrain submodules checkpoint
            pretrain_state_dict = get_model_state_dict(
                model,
                submodules={model.pretrain},
                options=StateDictOptions(keep_submodule_prefixes=False),
            )
            dist_cp.load_state_dict(
                {"model": pretrain_state_dict},
                storage_reader=dist_cp.FileSystemReader(pretrain_dir),
            )
            set_model_state_dict(
                model,
                model_state_dict={model.pretrain: pretrain_state_dict},
                options=StateDictOptions(strict=False),
            )

            try:
                # Load training submodules checkpoint
                model_state_dict, optim_state_dict = get_state_dict(
                    model,
                    optimizers=optim,
                    options=StateDictOptions(ignore_frozen_params=True),
                )
                dist_cp.load_state_dict(
                    {"model": model_state_dict, "optim": optim_state_dict},
                    storage_reader=dist_cp.FileSystemReader(pretrain_dir),
                )
                set_state_dict(
                    model,
                    optimizers=optim,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                    options=StateDictOptions(strict=False),
                )
            except KeyError:
                # If this is the first round of the fine tuning, then nothing is saved.
                # If this is the restart of the fine tuning, then checkpoint should exit.
                self.assertEqual(i, 0)

            # Training
            for j in range(3):
                batch = torch.rand(32, DIM, device="cuda")
                loss = model(batch).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()

            # Save state_dict
            model_state_dict, optim_state_dict = get_state_dict(
                model,
                optimizers=optim,
                options=StateDictOptions(ignore_frozen_params=True),
            )
            saved_state_dict = {"model": model_state_dict, "optim": optim_state_dict}
            dist_cp.save_state_dict(
                state_dict=saved_state_dict,
                storage_writer=dist_cp.FileSystemWriter(finetune_dir),
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_fine_tuning(self) -> None:
        self.assertTrue(os.path.exists(self.temp_dir))
        pretrain_dir = os.path.join(self.temp_dir, "pretrain")
        finetune_dir = os.path.join(self.temp_dir, "finetune")
        print(pretrain_dir, finetune_dir)
        if self.rank == 0:
            os.mkdir(pretrain_dir)
            os.mkdir(finetune_dir)
        dist.barrier()
        os.sync()
        self.assertTrue(os.path.exists(pretrain_dir))
        self.assertTrue(os.path.exists(finetune_dir))

        self.pretrain(pretrain_dir)
        self.finetune(pretrain_dir, finetune_dir)


if __name__ == "__main__":
    run_tests()
