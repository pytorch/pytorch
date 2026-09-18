# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as c10d
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import load_tests, run_tests


_torch_dist_nn_available = True
try:
    import torch.distributed.nn
except ImportError:
    _torch_dist_nn_available = False

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class TestDistributedNNFunctions(MultiProcessTestCase):
    BACKEND: str = ""

    def setUp(self):
        super().setUp()
        if not self.BACKEND:
            self.skipTest("BACKEND not set; run backend-specific subclass tests")
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
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

    def _init_process_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            backend=self.BACKEND,
        )

    def _get_device(self):
        return torch.device(f"cuda:{self.rank}")

    def test_broadcast(self):
        self._init_process_group()
        device = self._get_device()
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.broadcast(x, 1)
        self.assertEqual(y, 1 + torch.ones(5, 5))
        z = y.sin().sum()
        z.backward()
        # We can't check the gradient of communications numerically
        # so we have to do some calculations
        if self.rank == 1:
            self.assertEqual(x.grad, 2 * torch.cos(x))
        elif self.rank == 0:
            self.assertEqual(x.grad, torch.zeros(5, 5, device=device))

    def test_reduce(self):
        self._init_process_group()
        device = self._get_device()
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.reduce(x, 1, op=c10d.ReduceOp.SUM)

        if self.rank == 1:
            self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

        z = y.sin().sum()
        z.backward()
        x_g = (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)

    def test_allreduce(self):
        self._init_process_group()
        device = self._get_device()
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.all_reduce(x, op=c10d.ReduceOp.SUM)

        self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

        z = y.sin().sum()
        z.backward()
        x_g = 2 * (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)

    def test_all_gather(self):
        self._init_process_group()
        device = self._get_device()
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

    def test_all_to_all(self):
        self._init_process_group()
        device = self._get_device()
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

    def test_all_to_all_single(self):
        self._init_process_group()
        device = self._get_device()
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


if __name__ == "__main__":
    run_tests()
