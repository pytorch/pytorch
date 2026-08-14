# Owner(s): ["oncall: distributed"]


import sys

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_ucc_available():
    print("c10d UCC not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        hw_classification = HardwareClassification.ACCELERATOR

        def _test_broadcast(self, backend, device):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend=backend
            )
            device = torch.device(f"{device}:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            y = torch.distributed.nn.broadcast(x, 1)
            self.assertEqual(y, 1 + torch.ones(5, 5))
            z = y.sin().sum()
            z.backward()
            if self.rank == 1:
                self.assertEqual(x.grad, 2 * torch.cos(x))
            elif self.rank == 0:
                self.assertEqual(x.grad, torch.zeros(5, 5, device=device))

        def _test_reduce(self, backend, device):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend=backend
            )
            device = torch.device(f"{device}:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            y = torch.distributed.nn.reduce(x, 1, op=c10d.ReduceOp.SUM)

            if self.rank == 1:
                self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

            z = y.sin().sum()
            z.backward()
            x_g = (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_g)

        def _test_allreduce(self, backend, device):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend=backend
            )
            device = torch.device(f"{device}:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            y = torch.distributed.nn.all_reduce(x, op=c10d.ReduceOp.SUM)

            self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

            z = y.sin().sum()
            z.backward()
            x_g = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_g)

        def _test_all_gather(self, backend, device):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend=backend
            )
            device = torch.device(f"{device}:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            tensors = torch.distributed.nn.all_gather(x)
            for i, t in enumerate(tensors):
                self.assertEqual(t, torch.ones(5, 5, device=device) + i)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)

        def _test_all_to_all(self, backend, device):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend=backend
            )
            device = torch.device(f"{device}:{self.rank}")
            x0 = torch.ones(5, 5, device=device) + 2 * self.rank
            x1 = torch.ones(5, 5, device=device) + 2 * self.rank
            x0.requires_grad = True
            x1.requires_grad = True
            y0 = torch.empty_like(x0)
            y1 = torch.empty_like(x1)
            tensors = torch.distributed.nn.all_to_all([y0, y1], [x0, x1])
            for i, t in enumerate(tensors):
                self.assertEqual(t, torch.ones(5, 5, device=device) + 2 * i)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()
            x_s = (4 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s)
            self.assertEqual(x1.grad, x_s)

        def _test_all_to_all_single(self, backend, device):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend=backend
            )
            device = torch.device(f"{device}:{self.rank}")
            row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
            x = torch.ones(int(row), 5, device=device) * (self.rank + 1)
            x.requires_grad = True
            y = torch.empty_like(x)
            split_sizes = [(i + 1) * (self.rank + 1) for i in range(self.world_size)]
            y = torch.distributed.nn.all_to_all_single(
                y, x, output_split_sizes=split_sizes, input_split_sizes=split_sizes
            )
            expected = []
            for idx, tensor in enumerate(torch.split(x, split_sizes)):
                expected.append(torch.full_like(tensor, (idx + 1)))
            expected = torch.cat(expected)
            self.assertEqual(y, expected)
            z = y.sin().sum()
            z.backward()
            x_s = ((self.rank + 1) * torch.ones(int(row), 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)

        @skip_but_pass_in_sandcastle_if(
            torch.accelerator.is_available() and torch.accelerator.device_count() < 2,
            "test requires 2+ accelerators",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self, device):
            self._test_broadcast("ucc", device)

        @skip_but_pass_in_sandcastle_if(
            torch.accelerator.is_available() and torch.accelerator.device_count() < 2,
            "test requires 2+ accelerators",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self, device):
            self._test_reduce("ucc", device)

        @skip_but_pass_in_sandcastle_if(
            torch.accelerator.is_available() and torch.accelerator.device_count() < 2,
            "test requires 2+ accelerators",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self, device):
            self._test_allreduce("ucc", device)

        @skip_but_pass_in_sandcastle_if(
            torch.accelerator.is_available() and torch.accelerator.device_count() < 2,
            "test requires 2+ accelerators",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        def test_all_gather(self, device):
            self._test_all_gather("ucc", device)

        @skip_but_pass_in_sandcastle_if(
            torch.accelerator.is_available() and torch.accelerator.device_count() < 2,
            "test requires 2+ accelerators",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self, device):
            self._test_all_to_all("ucc", device)

        @skip_but_pass_in_sandcastle_if(
            torch.accelerator.is_available() and torch.accelerator.device_count() < 2,
            "test requires 2+ accelerators",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self, device):
            self._test_all_to_all_single("ucc", device)


    instantiate_device_type_tests(
        TestDistributedNNFunctionsUcc,
        globals(),
    )


if __name__ == "__main__":
    run_tests()
