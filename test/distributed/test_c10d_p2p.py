# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

from torch.testing._internal.common_utils import run_tests


class AbstractP2PTest(C10dBackendTest):
    def _peers(self):
        next_rank = (self.rank + 1) % self.world_size
        previous_rank = (self.rank - 1) % self.world_size
        return next_rank, previous_rank

    def test_send_recv(self):
        self._init_pg()
        next_rank, previous_rank = self._peers()
        send = torch.full((8,), float(self.rank), device=self.device)
        recv = torch.empty_like(send)
        if self.rank % 2 == 0:
            dist.send(send, next_rank)
            dist.recv(recv, previous_rank)
        else:
            dist.recv(recv, previous_rank)
            dist.send(send, next_rank)
        self.assertEqual(recv, torch.full_like(recv, previous_rank))

    def _test_batch_isend_irecv(self, recv_first, num_ops):
        next_rank, previous_rank = self._peers()
        sends = [
            torch.full((1,), float(self.rank + i * 100), device=self.device)
            for i in range(num_ops)
        ]
        recvs = [torch.empty_like(sends[0]) for _ in range(num_ops)]
        ops = []
        for send, recv in zip(sends, recvs):
            pair = [
                dist.P2POp(dist.isend, send, next_rank),
                dist.P2POp(dist.irecv, recv, previous_rank),
            ]
            ops.extend(reversed(pair) if recv_first else pair)
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        for i, recv in enumerate(recvs):
            self.assertEqual(recv, torch.full_like(recv, previous_rank + i * 100))

    def test_batch_isend_irecv(self):
        self._init_pg()
        self._test_batch_isend_irecv(recv_first=False, num_ops=1)

    def test_batch_isend_irecv_recv_first(self):
        self._init_pg()
        self._test_batch_isend_irecv(recv_first=True, num_ops=1)

    def test_batch_isend_irecv_multiple_ops(self):
        self._init_pg()
        self._test_batch_isend_irecv(recv_first=False, num_ops=2)


instantiate_backend_tests(globals(), "P2P", AbstractP2PTest, C10D_BACKENDS)


if __name__ == "__main__":
    run_tests()
