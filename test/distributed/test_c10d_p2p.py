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
    C10dBackendTestContinuous,
    instantiate_backend_tests,
)

from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import run_tests


COUNTS = (0, 4)


class P2PHelpers:
    def _peers(self):
        next_rank = (self.rank + 1) % self.world_size
        previous_rank = (self.rank - 1) % self.world_size
        return next_rank, previous_rank

    def _tensor(self, count, dtype, value):
        return torch.full((count,), value, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _test_send_recv(self, count, dtype):
        next_rank, previous_rank = self._peers()
        send = self._tensor(count, dtype, self.rank)
        recv = torch.empty_like(send)
        if self.rank % 2 == 0:
            dist.send(send, next_rank)
            dist.recv(recv, previous_rank)
        else:
            dist.recv(recv, previous_rank)
            dist.send(send, next_rank)
        self.assertEqual(recv, self._tensor(count, dtype, previous_rank))

    def _test_batch_isend_irecv(self, dtype, recv_first, num_ops):
        next_rank, previous_rank = self._peers()
        sends = [self._tensor(1, dtype, self.rank + i * 100) for i in range(num_ops)]
        recvs = [torch.empty_like(sends[0]) for _ in range(num_ops)]
        ops = []
        for send, recv in zip(sends, recvs):
            pair = [
                dist.P2POp(dist.isend, send, next_rank),
                dist.P2POp(dist.irecv, recv, previous_rank),
            ]
            ops.extend(reversed(pair) if recv_first else pair)
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
        for i, recv in enumerate(recvs):
            self.assertEqual(recv, self._tensor(1, dtype, previous_rank + i * 100))


class AbstractP2PTest(P2PHelpers, C10dBackendTest):
    def test_async_work_lifetime(self):
        if not self.supports_dropped_p2p_work:
            self.skipTest(f"{self.backend_name} does not retain dropped P2P work")
        self._init_pg()
        next_rank, previous_rank = self._peers()
        send = self._tensor(4, torch.float32, self.rank)
        recv = torch.empty_like(send)
        if self.rank % 2 == 0:
            send_work = dist.isend(send, next_rank)
            recv_work = dist.irecv(recv, previous_rank)
        else:
            recv_work = dist.irecv(recv, previous_rank)
            send_work = dist.isend(send, next_rank)
        del send_work
        recv_work.wait()
        self.assertEqual(recv, self._tensor(4, torch.float32, previous_rank))


class AbstractContinuousP2PTest(P2PHelpers, C10dBackendTestContinuous):
    def test_send_recv(self):
        for count in COUNTS:
            for dtype in self.dtypes:
                with self.subTest(count=count, dtype=dtype):
                    self._test_send_recv(count, dtype)

    def test_isend_irecv(self):
        next_rank, previous_rank = self._peers()
        for count in COUNTS:
            for dtype in self.dtypes:
                with self.subTest(count=count, dtype=dtype):
                    send = self._tensor(count, dtype, self.rank)
                    recv = torch.empty_like(send)
                    if self.rank % 2 == 0:
                        works = (
                            dist.isend(send, next_rank),
                            dist.irecv(recv, previous_rank),
                        )
                    else:
                        works = (
                            dist.irecv(recv, previous_rank),
                            dist.isend(send, next_rank),
                        )
                    for work in works:
                        work.wait()
                    self.assertEqual(recv, self._tensor(count, dtype, previous_rank))

    def test_batch_isend_irecv(self):
        for dtype in self.dtypes:
            for recv_first in (False, True):
                for num_ops in (1, 2):
                    with self.subTest(
                        dtype=dtype,
                        recv_first=recv_first,
                        num_ops=num_ops,
                    ):
                        self._test_batch_isend_irecv(dtype, recv_first, num_ops)

    def test_send_recv_event_ordering(self):
        # Regression test for the single-op P2P group-wrap fix (TorchComms
        # D109625550): the nccl2 sendImpl/recvImpl now wrap ncclSend/ncclRecv in
        # ncclGroupStart/End so the kernel is enqueued before the end CUDA event
        # is recorded. Without it a non-blocking comm could complete the event
        # ahead of the transfer and wait() would return stale/partial data. Loop
        # with distinct per-iteration values and read back immediately after
        # wait(): an early completion surfaces as a value mismatch. Generic P2P
        # ordering check, so it runs across every backend.
        next_rank, previous_rank = self._peers()
        numel = 1024 * 1024
        for i in range(50):
            send = self._tensor(numel, torch.float32, self.rank + i)
            recv = torch.empty_like(send)
            if self.rank % 2 == 0:
                send_work = dist.isend(send, next_rank)
                recv_work = dist.irecv(recv, previous_rank)
            else:
                recv_work = dist.irecv(recv, previous_rank)
                send_work = dist.isend(send, next_rank)
            send_work.wait()
            recv_work.wait()
            self.assertEqual(
                recv, self._tensor(numel, torch.float32, previous_rank + i)
            )


instantiate_backend_tests(globals(), "P2P", AbstractP2PTest, C10D_BACKENDS)


instantiate_backend_tests(
    globals(),
    "ContinuousP2P",
    AbstractContinuousP2PTest,
    C10D_BACKENDS,
    harness=MultiProcContinuousTest,
)

if __name__ == "__main__":
    run_tests()
