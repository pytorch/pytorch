# Owner(s): ["oncall: distributed"]

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._transport import (
    new_transport_rank,
    register_transport,
    Transport,
)
from torch.testing._internal.common_utils import run_tests, TestCase


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

        def factory():
            transport = Mock(spec=Transport)
            transport.supported.return_value = True
            transport.bind.return_value = bytes([rank, 0, 255])
            return transport

        register_transport("bootstrap-test", factory)
        if rank < 2:
            transport = new_transport_rank(
                "bootstrap-test",
                store=None if implicit else store,
                peer_rank=1 - rank if implicit else 11 - rank,
                rank=None if implicit else 10 + rank,
            )
            if transport.connect.call_args.args != (bytes([1 - rank, 0, 255]),):
                raise AssertionError("incorrect peer endpoint")
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
        store = dist.HashStore()
        for attempt, timeout in enumerate((2, timedelta(seconds=2))):
            self._connect_pair(store, attempt, timeout)

    def _connect_pair(self, store, attempt, timeout):
        transports = [Mock(spec=Transport), Mock(spec=Transport)]
        for rank, transport in enumerate(transports):
            transport.bind.return_value = bytes([rank, attempt, 255])
        with (
            patch(
                "torch.distributed._transport._bootstrap.new_transport",
                side_effect=lambda backend, device, fixture: fixture,
            ),
            ThreadPoolExecutor(2) as pool,
        ):
            futures = [
                pool.submit(
                    new_transport_rank,
                    "test",
                    store=store,
                    rank=rank,
                    peer_rank=1 - rank,
                    bootstrap_timeout=timeout,
                    fixture=transport,
                )
                for rank, transport in enumerate(transports)
            ]
            for future in futures:
                future.result(timeout=5)
        for rank, transport in enumerate(transports):
            self.assertEqual(
                transport.connect.call_args.args, (bytes([1 - rank, attempt, 255]),)
            )
            self.assertEqual(transport.connect.call_args.kwargs["timeout"], 2)
            transport.close.assert_not_called()

    def test_missing_peer_times_out_and_closes(self):
        transport = Mock(spec=Transport)
        transport.bind.return_value = b"endpoint"
        with (
            patch(
                "torch.distributed._transport._bootstrap.new_transport",
                return_value=transport,
            ),
            self.assertRaises(dist.DistStoreError),
        ):
            new_transport_rank(
                "test",
                store=dist.HashStore(),
                rank=0,
                peer_rank=1,
                bootstrap_timeout=0.05,
            )
        transport.connect.assert_not_called()
        transport.close.assert_called_once()

    def test_cleanup_preserves_error(self):
        transport = Mock(spec=Transport)
        transport.bind.side_effect = ValueError("bind failed")
        transport.close.side_effect = RuntimeError("close failed")
        with (
            patch(
                "torch.distributed._transport._bootstrap.new_transport",
                return_value=transport,
            ),
            self.assertRaisesRegex(ValueError, "bind failed"),
            self.assertLogs(
                "torch.distributed._transport._bootstrap", level="WARNING"
            ) as logs,
        ):
            new_transport_rank(
                "test",
                store=dist.HashStore(),
                rank=0,
                peer_rank=1,
                bootstrap_timeout=1,
            )
        self.assertIn("close failed", logs.output[0])

    def test_rank_override_and_attempt_count(self):
        store = dist.HashStore()
        with (
            patch(
                "torch.distributed.get_rank", side_effect=AssertionError("unexpected")
            ),
            patch(
                "torch.distributed._transport._bootstrap.new_transport",
                side_effect=RuntimeError("construction stopped"),
            ) as factory,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "construction stopped"):
                    new_transport_rank("test", store=store, rank=10, peer_rank=5)
            for rank, peer in ((-1, 0), (True, 0), (0, -1), (0, True), (0, 0)):
                with self.assertRaises(ValueError):
                    new_transport_rank(
                        "test",
                        store=store,
                        rank=rank,
                        peer_rank=peer,
                    )
            self.assertEqual(factory.call_count, 2)
        attempts = "transport/test/5/10/attempts/10"
        self.assertEqual(store.get(attempts), b"2")

    def test_implicit_rank_lookup(self):
        store = dist.HashStore()
        with (
            patch("torch.distributed.get_rank", return_value=4) as get_rank,
            patch(
                "torch.distributed._transport._bootstrap.new_transport",
                side_effect=RuntimeError("construction stopped"),
            ),
            self.assertRaisesRegex(RuntimeError, "construction stopped"),
        ):
            new_transport_rank("test", store=store, peer_rank=8)
        get_rank.assert_called_once_with()
        attempts = "transport/test/4/8/attempts/4"
        self.assertEqual(store.get(attempts), b"1")


if __name__ == "__main__":
    run_tests()
