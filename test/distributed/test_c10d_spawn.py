# Owner(s): ["oncall: distributed"]

import os
import sys
import tempfile

import torch
import torch.distributed as c10d
import torch.multiprocessing as mp
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import load_tests, NO_MULTIPROCESSING_SPAWN


# Torch distributed.nn is not available in windows
# check #42095, it errors on import.
_torch_dist_nn_available = True
try:
    import torch.distributed.nn
except ImportError:
    _torch_dist_nn_available = False

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if NO_MULTIPROCESSING_SPAWN:
    print("spawn not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class AbstractProcessGroupShareTensorTest:
    world_size = 2

    def _test_multiprocess(self, f, shared_tensors, init_pg, n_output):
        ws = self.world_size
        # file store will delete the test file on destruction
        file = tempfile.NamedTemporaryFile(delete=False)
        ctx = mp.get_context("spawn")
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f, args=(i, file.name, shared_tensors, ws, init_pg, c2p, p2c)
            )

            p.start()
            ps.append(p)

        for _ in range(ws * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                msg=f"Expect rank {pid} to receive tensor {expected} but got {result}.",
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(2)

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode. See https://bugs.python.org/issue33884.
    @classmethod
    def _test_broadcast_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.broadcast(xs).wait()
        c2p.put((rank, torch.zeros(2, 2), xs[0].to("cpu")))
        p2c.get()

    @classmethod
    def _test_allreduce_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.allreduce(xs, op=c10d.ReduceOp.SUM).wait()
        c2p.put((rank, torch.ones(2, 2) * 2, xs[0].to("cpu")))
        p2c.get()

    @classmethod
    def _test_allgather_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        ys = [[torch.zeros_like(xs[0]) for i in range(world_size)]]
        pg.allgather(ys, xs).wait()
        for i in range(world_size):
            c2p.put((rank, torch.ones(2, 2) * i, ys[0][i].to("cpu")))

        p2c.get()


class TestDistributedNNFunctions(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
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

    def _test_broadcast(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.broadcast(x, 1)
        self.assertEqual(y, 1 + torch.ones(5, 5))
        z = y.sin().sum()
        z.backward()
        # We can't check the gradient of communications numerically so we have to do some calculations
        if self.rank == 1:
            self.assertEqual(x.grad, 2 * torch.cos(x))
        elif self.rank == 0:
            self.assertEqual(x.grad, torch.zeros(5, 5, device=device))

    def _test_reduce(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.reduce(x, 1, op=c10d.ReduceOp.SUM)

        if self.rank == 1:
            self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

        z = y.sin().sum()
        z.backward()
        # Gradients are broadcasted to both ranks
        x_g = (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)

    def _test_allreduce(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.all_reduce(x, op=c10d.ReduceOp.SUM)

        self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

        z = y.sin().sum()
        z.backward()
        x_g = 2 * (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)

    def _test_all_gather(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
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

    def _test_all_to_all(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
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

    def _test_all_to_all_single(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
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
