import sys
import tempfile

import torch
import torch.distributed as c10d
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import NO_MULTIPROCESSING_SPAWN
from torch.testing._internal.common_utils import load_tests

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
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)

if NO_MULTIPROCESSING_SPAWN:
    print('spawn not available, skipping tests', file=sys.stderr)
    sys.exit(0)


class AbstractProcessGroupShareTensorTest(object):
    world_size = 2

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

    @classmethod
    def _test_allreduce_process(
            cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c):
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.allreduce(xs, op=c10d.ReduceOp.SUM).wait()
        c2p.put((rank, torch.ones(2, 2) * 2, xs[0].to("cpu")))
        p2c.get()

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
