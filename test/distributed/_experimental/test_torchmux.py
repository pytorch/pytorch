# Owner(s): ["oncall: distributed"]

import asyncio
import collections
import queue
import threading
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed._experimental.torchmux.backend import (
    _finish_reduce,
    _reduce_tensor,
    _run_coordinator_in_thread,
    ProcessGroupTorchmux,
)
from torch.distributed._experimental.torchmux.coord_client import (
    CollectiveMismatch,
    CoordClient,
    PeerGone,
)
from torch.distributed._experimental.torchmux.coordinator import (
    _PeerGone,
    _Prepare,
    Coordinator,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _start_thread(fn):
    q = queue.Queue()

    def target():
        try:
            q.put((True, fn()))
        except BaseException as e:
            q.put((False, e))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, q


class TestTorchMux(TestCase):
    def _thread_result(self, thread, q, timeout=5.0):
        thread.join(timeout)
        self.assertFalse(thread.is_alive())
        ok, result = q.get_nowait()
        if not ok:
            raise result
        return result

    def test_partial_recv_keeps_early_deposit(self):
        addr = _run_coordinator_in_thread()
        clients = [CoordClient(addr=addr) for _ in range(3)]
        try:
            clients[0].register(0)
            self.assertIsNone(clients[0].prepare({}, (1, 2)))
            thread, q = _start_thread(clients[0].wait_for_recv)

            t1, q1 = _start_thread(lambda: clients[1].register(1))
            self._thread_result(t1, q1)
            self.assertEqual(clients[1].prepare({(0,): None}, ()), {})
            clients[1].done()

            t2, q2 = _start_thread(lambda: clients[2].register(2))
            self._thread_result(t2, q2)
            self.assertEqual(clients[2].prepare({(0,): None}, ()), {})
            clients[2].done()

            self.assertEqual(self._thread_result(thread, q), {1: None, 2: None})
        finally:
            for client in clients:
                try:
                    client.done()
                except Exception:
                    client.close()

    def test_send_only_fast_paths(self):
        """A send-only prepare with no recv returns immediately (fast path)."""
        addr = _run_coordinator_in_thread()
        c0 = CoordClient(addr=addr)
        try:
            c0.register(0)

            result = c0.prepare({(1,): None}, ())
            self.assertEqual(result, {})
        finally:
            try:
                c0.done()
            except Exception:
                c0.close()

    def test_rank0_unblocks_first(self):
        """Rank 0 unblocks from register first."""
        addr = _run_coordinator_in_thread()
        c0 = CoordClient(addr=addr)
        c1 = CoordClient(addr=addr)
        try:
            t0, q0 = _start_thread(lambda: c0.register(0))
            t1, q1 = _start_thread(lambda: c1.register(1))

            self._thread_result(t0, q0)

            # rank 1 is still blocked in register
            self.assertTrue(t1.is_alive())

            # rank 0 does a collective that requires rank 1's data — it blocks,
            # which lets the coordinator unblock rank 1
            self.assertIsNone(c0.prepare({(1,): None}, (1,)))
            t_wait, q_wait = _start_thread(c0.wait_for_recv)

            # rank 1 should now be unblocked
            self._thread_result(t1, q1)

            # rank 1 deposits for rank 0
            self.assertEqual(c1.prepare({(0,): None}, ()), {})

            # rank 1 still owns the baton, so rank 0 cannot resume yet.
            t_wait.join(0.1)
            self.assertTrue(t_wait.is_alive())

            # Once rank 1 yields, rank 0 gets its data.
            c1.done()
            self.assertEqual(self._thread_result(t_wait, q_wait), {1: None})
        finally:
            for client in (c0, c1):
                try:
                    client.done()
                except Exception:
                    client.close()

    def test_satisfied_pending_waits_for_active_rank_to_yield(self):
        loop = asyncio.new_event_loop()
        try:
            coord = Coordinator()
            coord.active_rank = 1

            prep = _Prepare([], [1])
            prep.release_future = loop.create_future()
            coord.pending[0] = prep
            coord.mailboxes[(1, 0)] = collections.deque([(None, b"")])

            coord._try_complete_pending()

            self.assertFalse(prep.release_future.done())
            self.assertEqual(coord.active_rank, 1)
            self.assertIn(0, coord.pending)

            coord.active_rank = None
            coord._try_complete_pending()

            self.assertTrue(prep.release_future.done())
            self.assertEqual(coord.active_rank, 0)
            self.assertNotIn(0, coord.pending)
            self.assertEqual(prep.release_future.result(), {1: (None, b"")})
        finally:
            loop.close()

    def test_cleanup_collects_deposit_before_marking_peer_gone(self):
        loop = asyncio.new_event_loop()
        try:
            coord = Coordinator()
            coord.active_rank = 1
            writer0 = object()
            writer1 = object()
            coord.clients = {0: writer0, 1: writer1}
            coord.writer_to_rank = {writer0: 0, writer1: 1}

            prep = _Prepare([], [1])
            prep.release_future = loop.create_future()
            coord.pending[0] = prep
            coord.mailboxes[(1, 0)] = collections.deque([(None, b"")])

            loop.run_until_complete(coord._cleanup_client(writer1))

            self.assertTrue(prep.release_future.done())
            self.assertEqual(coord.active_rank, 0)
            self.assertNotIn(0, coord.pending)
            self.assertEqual(prep.release_future.result(), {1: (None, b"")})
        finally:
            loop.close()

    def test_peer_gone_before_release_gpu(self):
        """If a peer disconnects before the waiting rank calls release_gpu,
        the pending prepare should survive with a failure marker so that
        handle_release_gpu can report PeerGone instead of 'no pending prepare'."""
        loop = asyncio.new_event_loop()
        try:
            coord = Coordinator()
            coord.active_rank = 0
            writer0 = object()
            writer1 = object()
            coord.clients = {0: writer0, 1: writer1}
            coord.writer_to_rank = {writer0: 0, writer1: 1}

            # Rank 0 has a pending prepare waiting for rank 1's data.
            # release_future is None — rank 0 hasn't called release_gpu yet.
            prep = _Prepare([], [1])
            coord.pending[0] = prep

            # Rank 1 disconnects.
            loop.run_until_complete(coord._cleanup_client(writer1))

            # The pending entry must survive (not be popped) so that
            # handle_release_gpu can find it and report PeerGone.
            self.assertIn(0, coord.pending)
            self.assertIsInstance(coord.pending[0].failed, _PeerGone)
        finally:
            loop.close()

    def test_mismatch_when_pending_peer_waits_on_current_rank(self):
        addr = _run_coordinator_in_thread()
        c0 = CoordClient(addr=addr)
        c1 = CoordClient(addr=addr)
        try:
            c0.register(0)
            t1, q1 = _start_thread(lambda: c1.register(1))

            self.assertIsNone(c0.prepare({}, (1,)))
            t_wait, q_wait = _start_thread(c0.wait_for_recv)
            self._thread_result(t1, q1)
            with self.assertRaisesRegex(
                CollectiveMismatch, "rank 0 recv from 1; 1 does not send to 0"
            ):
                c1.prepare({}, ())
            c1.done()
            with self.assertRaises(PeerGone):
                self._thread_result(t_wait, q_wait)
        finally:
            for client in (c0, c1):
                try:
                    client.done()
                except Exception:
                    client.close()

    def test_exchange_releases_cuda_tensors(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def prepare(self, send, recv):
                self.calls.append("prepare")
                return None

            def release_gpu(self):
                self.calls.append("release_gpu")
                return {"released": None}

            def wait_for_recv(self):
                self.calls.append("wait_for_recv")
                return {"waited": None}

        class CudaLike:
            is_cuda = True

        pg = type("FakePG", (), {})()
        pg._client = FakeClient()
        result = ProcessGroupTorchmux._exchange(pg, {(1,): CudaLike()}, ())
        self.assertEqual(result, {"released": None})
        self.assertEqual(pg._client.calls, ["prepare", "release_gpu"])

        pg._client = FakeClient()
        result = ProcessGroupTorchmux._exchange(pg, {(1,): None}, ())
        self.assertEqual(result, {"waited": None})
        self.assertEqual(pg._client.calls, ["prepare", "wait_for_recv"])

    def test_async_allreduce_gets_completed_work(self):
        store = dist.HashStore()
        dist.init_process_group("torchmux", store=store, rank=0, world_size=1)
        try:
            t = torch.tensor([1.0])
            work = dist.all_reduce(t, async_op=True)

            self.assertIsNotNone(work)
            self.assertTrue(work.wait())
            self.assertEqual(t, torch.tensor([1.0]))
        finally:
            dist.destroy_process_group()

    def test_recv_returns_completed_work(self):
        test_case = self

        class FakePG:
            def _exchange(self, send, recv, tensors=(), force_cuda=False):
                test_case.assertEqual(send, {})
                test_case.assertEqual(recv, (1,))
                return {1: torch.tensor([5.0])}

        pg = FakePG()
        t = torch.zeros(1)
        work = ProcessGroupTorchmux.recv(pg, [t], 1, 0)

        self.assertIsNotNone(work)
        self.assertTrue(work.wait())
        self.assertEqual(t, torch.tensor([5.0]))

    def test_release_gpu_restores_before_decoding_recv(self):
        client = object.__new__(CoordClient)
        calls = []

        def mark(name):
            def f(*args, **kwargs):
                calls.append(name)

            return f

        def wait_for_recv_raw():
            calls.append("wait_for_recv_raw")
            return {"recv": [{"src": 1, "tensor": None}]}, b""

        def decode_recv(entries, payload):
            calls.append("decode_recv")
            self.assertEqual(
                calls,
                [
                    "synchronize",
                    "empty_cache",
                    "checkpoint",
                    "wait_for_recv_raw",
                    "restore",
                    "decode_recv",
                ],
            )
            return {1: None}

        client._wait_for_recv_raw = wait_for_recv_raw
        client._decode_recv = decode_recv

        with (
            patch("torch.cuda.synchronize", mark("synchronize")),
            patch("torch.cuda.empty_cache", mark("empty_cache")),
            patch(
                "torch.distributed._experimental.torchmux.cuda_checkpoint.checkpoint_self",
                mark("checkpoint"),
            ),
            patch(
                "torch.distributed._experimental.torchmux.cuda_checkpoint.restore_self",
                mark("restore"),
            ),
        ):
            self.assertEqual(CoordClient.release_gpu(client), {1: None})

        self.assertEqual(
            calls,
            [
                "synchronize",
                "empty_cache",
                "checkpoint",
                "wait_for_recv_raw",
                "restore",
                "decode_recv",
            ],
        )

    def test_tensor_serialization_round_trip(self):
        """Tensor serialize/deserialize preserves values, dtype, and shape."""
        from torch.distributed._experimental.torchmux.coord_client import (
            _deserialize_tensor,
            _serialize_tensor,
        )

        cases = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor(42),
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([0.5, -0.5], dtype=torch.bfloat16),
            torch.zeros(3, 4, 5, dtype=torch.float64),
            torch.tensor([True, False, True]),
        ]
        for sent in cases:
            header, payload = _serialize_tensor(sent)
            received = _deserialize_tensor(header, payload)
            self.assertEqual(received, sent)
            self.assertEqual(received.dtype, sent.dtype)
            self.assertEqual(received.shape, sent.shape)

    def test_tensor_round_trip(self):
        """Real tensors survive the full coordinator wire path."""
        addr = _run_coordinator_in_thread()
        c0 = CoordClient(addr=addr)
        c1 = CoordClient(addr=addr)
        try:
            t0, q0 = _start_thread(lambda: c0.register(0))
            t1, q1 = _start_thread(lambda: c1.register(1))
            self._thread_result(t0, q0)

            sent_0to1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            sent_1to0 = torch.tensor([10, 20, 30], dtype=torch.int64)

            # Bidirectional exchange: c0 sends to c1, expects recv from c1
            self.assertIsNone(c0.prepare({(1,): sent_0to1}, (1,)))
            t_wait, q_wait = _start_thread(c0.wait_for_recv)

            # c1 unblocks from register, sends to c0 and picks up c0's tensor
            self._thread_result(t1, q1)
            result1 = c1.prepare({(0,): sent_1to0}, (0,))
            self.assertIsNotNone(result1)
            self.assertEqual(result1[0], sent_0to1)
            self.assertEqual(result1[0].dtype, sent_0to1.dtype)
            self.assertEqual(result1[0].shape, sent_0to1.shape)

            # c1 yields baton, c0 gets c1's tensor via wait_for_recv
            c1.done()
            result0 = self._thread_result(t_wait, q_wait)
            self.assertEqual(result0[1], sent_1to0)
            self.assertEqual(result0[1].dtype, sent_1to0.dtype)
            self.assertEqual(result0[1].shape, sent_1to0.shape)
        finally:
            for client in (c0, c1):
                try:
                    client.done()
                except Exception:
                    client.close()

    def test_allreduce_ops(self):
        t = torch.tensor([2, 3])
        _reduce_tensor(t, torch.tensor([5, 7]), dist.ReduceOp.PRODUCT)
        self.assertEqual(t, torch.tensor([10, 21]))

        t = torch.tensor([2, 3])
        _reduce_tensor(t, torch.tensor([5, 1]), dist.ReduceOp.MAX)
        self.assertEqual(t, torch.tensor([5, 3]))

        t = torch.tensor([2.0, 4.0])
        _reduce_tensor(t, torch.tensor([4.0, 8.0]), dist.ReduceOp.AVG)
        _finish_reduce(t, dist.ReduceOp.AVG, 2)
        self.assertEqual(t, torch.tensor([3.0, 6.0]))


if __name__ == "__main__":
    run_tests()
