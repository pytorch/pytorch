# Owner(s): ["oncall: distributed"]

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._transport import new_transport, register_transport, Transport
from torch.testing._internal.common_utils import run_tests, TestCase


class _Transport(Transport):
    """Records connections; keeps the base ``connect_rank``."""

    def __init__(self, endpoint=b""):
        super().__init__()
        self.endpoint = endpoint
        self.binds = []
        self.peers = []
        self.closes = 0

    @staticmethod
    def supported():
        return True

    def bind(self, *, timeout=None):
        self.binds.append(timeout)
        return self.endpoint

    def connect(self, peer_url, *, timeout=None):
        self.peers.append((peer_url, timeout))
        return 0

    def connected(self):
        return bool(self.peers)

    def register_memory(self, tensor, *, timeout=None):
        raise NotImplementedError

    def write(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        raise NotImplementedError

    def read(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        raise NotImplementedError

    def close(self, *, timeout=None):
        self.closes += 1


class _SubTransport(_Transport):
    pass


_PREFIX = "transport_connect_rank/default"


def _hash_store():
    # Fail fast instead of waiting HashStore's 300 s default timeout.
    store = dist.HashStore()
    store.set_timeout(timedelta(seconds=5))
    return store


def _bootstrap_worker(rank, port, path, implicit):
    store = dist.TCPStore("127.0.0.1", port, is_master=False)
    if implicit:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{path}",
            rank=rank,
            world_size=3,
            timeout=timedelta(seconds=30),
        )
    try:
        register_transport("bootstrap-test", lambda: _Transport(bytes([rank, 0, 255])))
        if rank < 2:
            transport = new_transport("bootstrap-test")
            transport.connect_rank(
                1 - rank if implicit else 11 - rank,
                store=None if implicit else store,
                rank=None if implicit else 10 + rank,
            )
            if transport.peers != [(bytes([1 - rank, 0, 255]), None)]:
                raise AssertionError(f"incorrect peer endpoint: {transport.peers}")
        if implicit:
            # Rank 2 does not bootstrap, proving there is no collective in it.
            dist.barrier()
    finally:
        if implicit:
            dist.destroy_process_group()


class TestRankBootstrap(TestCase):
    def test_tcpstore_explicit_rank_without_process_group(self):
        self._run_tcpstore(implicit=False)

    def test_default_store_and_rank(self):
        self._run_tcpstore(implicit=True)

    def _run_tcpstore(self, implicit):
        store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _bootstrap_worker,
                args=(store.port, f"{directory}/store", implicit),
                nprocs=3 if implicit else 2,
            )

    def test_pair(self):
        store = _hash_store()
        for attempt, timeout in enumerate((2, timedelta(seconds=2))):
            self._connect_pair(store, attempt, timeout)

    def _connect_pair(self, store, attempt, timeout):
        transports = [_Transport(bytes([rank, attempt, 255])) for rank in range(2)]
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(t.connect_rank, 1 - r, store=store, rank=r, timeout=timeout)
                for r, t in enumerate(transports)
            ]
            self.assertEqual([future.result(timeout=5) for future in futures], [0, 0])
        for rank, transport in enumerate(transports):
            self.assertEqual(transport.peers, [(bytes([1 - rank, attempt, 255]), 2)])

    def test_connect_rank_multiple_peers(self):
        # Each thread needs its own client; a TCPStore wait blocks its client.
        store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
        # Peers pair by tag, not transport class.
        transports = [
            _Transport(bytes([0])),
            _Transport(bytes([1])),
            _SubTransport(bytes([2])),
        ]
        calls = [(0, 1), (0, 2), (1, 0), (2, 0)]
        with ThreadPoolExecutor(len(calls)) as pool:
            futures = [
                pool.submit(
                    transports[rank].connect_rank,
                    peer,
                    store=store.clone(),
                    rank=rank,
                    timeout=5.0,
                )
                for rank, peer in calls
            ]
            self.assertEqual([future.result(timeout=10) for future in futures], [0] * 4)
        self.assertEqual(
            sorted(transports[0].peers), [(bytes([1]), 5), (bytes([2]), 5)]
        )
        self.assertEqual(transports[1].peers, [(bytes([0]), 5)])
        self.assertEqual(transports[2].peers, [(bytes([0]), 5)])

    def test_connect_rank_default_store_and_tags(self):
        # Two transports per rank pair, separated by tag, with timeout=None.
        transports = {
            (rank, tag): _Transport(f"{rank}{tag}".encode())
            for rank in range(2)
            for tag in "ab"
        }
        store = _hash_store()
        with (
            patch(
                "torch.distributed.distributed_c10d._get_default_store",
                return_value=store,
            ),
            ThreadPoolExecutor(len(transports)) as pool,
        ):

            def submit(rank, tag):
                transport = transports[rank, tag]
                return pool.submit(
                    transport.connect_rank, 1 - rank, rank=rank, tag=tag, timeout=None
                )

            # Each peer waits under the other tag first, so tags must not pair.
            futures = [submit(0, "a"), submit(1, "b")]
            store.wait(["transport_connect_rank/a/0/1/attempts/0"])
            store.wait(["transport_connect_rank/b/0/1/attempts/1"])
            futures += [submit(0, "b"), submit(1, "a")]
            self.assertEqual([future.result(timeout=10) for future in futures], [0] * 4)
        for (rank, tag), transport in transports.items():
            self.assertEqual(transport.binds, [None])
            self.assertEqual(transport.peers, [(f"{1 - rank}{tag}".encode(), None)])

    def test_connect_rank_missing_peer_times_out(self):
        transport = _Transport(b"endpoint")
        store = dist.HashStore()
        store.set_timeout(timedelta(seconds=60))
        start = time.monotonic()
        with self.assertRaises(dist.DistStoreError):
            transport.connect_rank(1, store=store, rank=0, timeout=0.05)
        self.assertLess(time.monotonic() - start, 2)
        # timeout=None falls back to the Store's timeout.
        store.set_timeout(timedelta(milliseconds=50))
        start = time.monotonic()
        with self.assertRaises(dist.DistStoreError):
            transport.connect_rank(1, store=store, rank=0, timeout=None)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(transport.binds, [0.05, None])
        self.assertEqual(transport.peers, [])
        self.assertEqual(transport.closes, 0)

    def test_connect_rank_connect_failure_keeps_transport(self):
        transport = _Transport(b"endpoint")
        store = _hash_store()
        store.set(f"{_PREFIX}/0/1/1/1", b"peer")
        with (
            patch.object(transport, "connect", side_effect=RuntimeError("refused")),
            self.assertRaisesRegex(RuntimeError, "refused"),
        ):
            transport.connect_rank(1, store=store, rank=0, timeout=1)
        self.assertEqual(transport.closes, 0)

    def test_connect_rank_attempt_count(self):
        store = _hash_store()
        transport = _Transport()
        with (
            patch(
                "torch.distributed.get_rank", side_effect=AssertionError("unexpected")
            ),
            patch.object(
                transport, "bind", side_effect=RuntimeError("bind stopped")
            ) as bind,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "bind stopped"):
                    transport.connect_rank(5, store=store, rank=10)
            # Invalid arguments are rejected before an attempt is taken.
            keys = store.num_keys()
            for rank, peer in ((-1, 0), (True, 0), (0, -1), (0, True), (0, 0)):
                with self.assertRaisesRegex(ValueError, "rank"):
                    transport.connect_rank(peer, store=store, rank=rank)
            too_long = 2**31 / 1000
            bad = (0, 0.0005, -1, too_long, float("inf"), float("nan"))
            for timeout in (*bad, timedelta(microseconds=500)):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    transport.connect_rank(5, store=store, rank=10, timeout=timeout)
            for timeout in (True, "1"):
                with self.assertRaisesRegex(TypeError, "timeout"):
                    transport.connect_rank(5, store=store, rank=10, timeout=timeout)
            with self.assertRaisesRegex(TypeError, "tag"):
                transport.connect_rank(5, store=store, rank=10, tag=1)
            transport._default_timeout = 0.0
            with self.assertRaisesRegex(ValueError, "got 0.0 s"):
                transport.connect_rank(5, store=store, rank=10)
            transport._default_timeout = None
            self.assertEqual(store.num_keys(), keys)
            with self.assertRaisesRegex(RuntimeError, "bind stopped"):
                transport.connect_rank(5, store=store, rank=10, tag="other")
            for timeout in (0.001, (2**31 - 1) / 1000, timedelta(seconds=2)):
                with self.assertRaisesRegex(RuntimeError, "bind stopped"):
                    transport.connect_rank(
                        5, store=store, rank=10, tag="bounds", timeout=timeout
                    )
            self.assertEqual(bind.call_count, 6)
            self.assertEqual(bind.call_args.kwargs, {"timeout": 2.0})
        self.assertEqual(store.get(f"{_PREFIX}/5/10/attempts/10"), b"2")
        self.assertEqual(
            store.get("transport_connect_rank/other/5/10/attempts/10"), b"1"
        )

    def test_implicit_rank_lookup(self):
        store = _hash_store()
        transport = _Transport()
        # timeout=None resolves to the transport's default.
        transport._default_timeout = 5.0
        with (
            patch("torch.distributed.get_rank", return_value=4) as get_rank,
            patch.object(
                transport, "bind", side_effect=RuntimeError("bind stopped")
            ) as bind,
            self.assertRaisesRegex(RuntimeError, "bind stopped"),
        ):
            transport.connect_rank(8, store=store)
        bind.assert_called_once_with(timeout=5.0)
        get_rank.assert_called_once_with()
        self.assertEqual(store.get(f"{_PREFIX}/4/8/attempts/4"), b"1")


if __name__ == "__main__":
    run_tests()
