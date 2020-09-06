import copy
import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as c10d
import torch.multiprocessing as mp
import torch.nn as nn

from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_distributed import requires_gloo
from torch.testing._internal.common_utils import TestCase, load_tests, run_tests, skipIfRocm
from torch.testing._internal.common_utils import NO_MULTIPROCESSING_SPAWN, TEST_WITH_TSAN


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)


if NO_MULTIPROCESSING_SPAWN:
    print('spawn not available, skipping tests', file=sys.stderr)
    sys.exit(0)


NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")


class ProcessGroupShareTensorTest(TestCase):

    world_size = 2

    @classmethod
    def opts(cls, threads=2):
        opts = c10d.ProcessGroupGloo.Options()
        opts.devices = [c10d.ProcessGroupGloo.create_device(interface="lo")]
        opts.timeout = 5.0
        opts.threads = threads
        return opts

    @classmethod
    def _init_pg_gloo(cls, rank, filename, world_size):
        store = c10d.FileStore(filename, world_size)
        return c10d.ProcessGroupGloo(
            store, rank, world_size, ProcessGroupShareTensorTest.opts())

    @classmethod
    def _init_pg_nccl(cls, rank, filename, world_size):
        store = c10d.FileStore(filename, world_size)
        return c10d.ProcessGroupNCCL(store, rank, world_size)

    def _test_multiprocess(self, f, shared_tensors, init_pg, n_output):
        ws = self.world_size
        # file store will delete the test file on destruction
        file = tempfile.NamedTemporaryFile(delete=False)
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f,
                args=(i, file.name, shared_tensors, ws, init_pg, c2p, p2c))

            p.start()
            ps.append(p)

        for _ in range(ws * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                msg=(
                    "Expect rank {} to receive tensor {} but got {}."
                ).format(pid, expected, result)
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(2)

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode. See https://bugs.python.org/issue33884.
    @classmethod
    def _test_broadcast_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.broadcast(xs).wait()
        c2p.put((rank, torch.zeros(2, 2), xs[0].to("cpu")))
        p2c.get()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    def test_shared_broadcast_gloo(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_broadcast_process,
            [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_gloo,
            1)


    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_broadcast_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_broadcast_process,
            [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            1)

    @classmethod
    def _test_allreduce_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.allreduce(xs, op=c10d.ReduceOp.SUM).wait()
        c2p.put((rank, torch.ones(2, 2) * 2, xs[0].to("cpu")))
        p2c.get()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    def test_shared_allreduce_gloo(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allreduce_process,
            [torch.ones(2, 2).to(i) for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_gloo,
            1)

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_allreduce_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allreduce_process,
            [torch.ones(2, 2).to(i) for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            1)

    @classmethod
    def _test_reduce_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        x = shared_tensors[rank]
        pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
        if rank == 0:
            c2p.put((rank, torch.ones(2, 2) * 2, x.to("cpu")))
        else:
            c2p.put((rank, torch.ones(2, 2), x.to("cpu")))
        p2c.get()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_reduce_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_reduce_process,
            [torch.ones(2, 2).to(i) for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            1)

    @classmethod
    def _test_allgather_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        ys = [[torch.zeros_like(xs[0]) for i in range(world_size)]]
        pg.allgather(ys, xs).wait()
        for i in range(world_size):
            c2p.put((rank, torch.ones(2, 2) * i, ys[0][i].to("cpu")))

        p2c.get()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    def test_shared_allgather_gloo(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allgather_process,
            [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_gloo,
            self.world_size)

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @unittest.skipIf(NO_NCCL, "NCCL needed")
    def test_shared_allgather_nccl(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allgather_process,
            [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
            ProcessGroupShareTensorTest._init_pg_nccl,
            self.world_size)

    @classmethod
    def _test_allgather_chunk_process(
            cls, rank, filename, shared_tensor, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        chunks = torch.chunk(shared_tensor, world_size, dim=0)
        x = chunks[rank]
        ys = [torch.zeros_like(x) for _ in range(world_size)]
        pg.allgather(ys, x).wait()
        c2p.put((rank, chunks[0].to("cpu"), ys[0].to("cpu")))
        c2p.put((rank, chunks[1].to("cpu"), ys[1].to("cpu")))
        p2c.get()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    def test_shared_allgather_chunk_gloo(self):
        self._test_multiprocess(
            ProcessGroupShareTensorTest._test_allgather_chunk_process,
            torch.tensor(range(4)).reshape(2, 2),
            ProcessGroupShareTensorTest._init_pg_gloo,
            self.world_size)


@unittest.skipIf(TEST_WITH_TSAN, "TSAN is not fork-safe since we're forking in a multi-threaded environment")
class DistributedDataParallelSingleProcessTest(TestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)  # noqa: P201

    def tearDown(self):
        try:
            os.remove(self.file.name)
        except OSError:
            pass

    def _test_base(self, net, inp, check_allclose=True):
        store = c10d.FileStore(self.file.name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        ddp = nn.parallel.DistributedDataParallel(
            copy.deepcopy(net),
            process_group=process_group
        )

        net_opt = torch.optim.Adam(net.parameters(), lr=0.001)
        ddp_opt = torch.optim.Adam(ddp.parameters(), lr=0.001)

        for i, j in zip(ddp.parameters(), net.parameters()):
            self.assertTrue(i.allclose(j))

        for _ in range(10):
            net_out = net(*inp)
            ddp_out = ddp(*inp)

            net_out.sum().backward()
            ddp_out.sum().backward()

            net_opt.step()
            ddp_opt.step()

        if check_allclose:
            for i, j in zip(ddp.parameters(), net.parameters()):
                self.assertTrue(i.allclose(j))

    @requires_gloo()
    def test_cpu(self):
        self._test_base(nn.Linear(2, 2), [torch.randn(30, 2)])

    @requires_gloo()
    @unittest.skipIf(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    def test_cuda(self):
        self._test_base(nn.Linear(2, 2).to(0), [torch.randn(30, 2).to(0)])

    @requires_gloo()
    @unittest.skipIf(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    @skipIfRocm
    def test_rnn(self):
        # This test is inspired by the bug reported in
        # https://github.com/pytorch/pytorch/issues/36268
        BATCH_SIZE = 4
        INPUT_DIM = 256
        OUTPUT_DIM = 256
        HIDDEN_DIM = 256
        N_LAYERS = 3
        SEQ_LEN = 100

        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
                super(Net, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.hidden_layers = hidden_layers

                self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True)
                self.h2o = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, y):
                self.lstm.flatten_parameters()
                h_t, _ = self.lstm(x)
                output = self.h2o(h_t)
                loss = nn.functional.mse_loss(output, y)
                return loss

        net = Net(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(0)
        inp = [
            torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM)).to(0),
            torch.rand((BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)).to(0)
        ]

        # Not checking result allclose as the parameter inconsistency exist
        # prior to this change. See #37079
        self._test_base(net, inp, check_allclose=False)


if __name__ == '__main__':
    run_tests()
