# Owner(s): ["oncall: distributed"]

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._transport import new_transport, register_transport, Transport
from torch.distributed._transport._bootstrap import _RankBootstrap
from torch.testing._internal.common_utils import run_tests, TestCase


def _bootstrap_worker(rank, path):
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
            transport = new_transport(
                "bootstrap-test", peer_rank=1 - rank, bootstrap_tag="default-pair"
            )
            if transport.connect.call_args.args != (bytes([1 - rank, 0, 255]),):
                raise AssertionError("incorrect default-group endpoint")
        # Rank 2 deliberately does not bootstrap; no whole-group collective.
        dist.barrier()
        subgroup = dist.new_group([1, 2], backend="gloo")
        if rank > 0:
            transport = new_transport(
                "bootstrap-test",
                peer_rank=2 - rank,
                group=subgroup,
                bootstrap_tag="subgroup-pair",
            )
            if transport.connect.call_args.args != (bytes([3 - rank, 0, 255]),):
                raise AssertionError("incorrect subgroup endpoint")
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestRankBootstrap(TestCase):
    def test_process_group_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_bootstrap_worker, args=(f"{directory}/store",), nprocs=3)

    def test_pair(self):
        store = dist.HashStore()
        transports = [Mock(spec=Transport), Mock(spec=Transport)]
        for rank, transport in enumerate(transports):
            transport.bind.return_value = bytes([rank, 0, 255])
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(_RankBootstrap(store, rank, 1 - rank, 2).connect, transport)
                for rank, transport in enumerate(transports)
            ]
            for future in futures:
                future.result(timeout=5)
        for rank, transport in enumerate(transports):
            self.assertEqual(
                transport.connect.call_args.args, (bytes([1 - rank, 0, 255]),)
            )
            self.assertGreater(transport.connect.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(transport.connect.call_args.kwargs["timeout"], 2)
            transport.close.assert_not_called()

    def test_stale_peer_times_out_and_closes(self):
        store = dist.HashStore()
        store.set("1", "old")
        store.set("endpoint/old", "b2xkIGVuZHBvaW50")
        transport = Mock(spec=Transport)
        transport.bind.return_value = b"new endpoint"
        with self.assertRaises(dist.DistStoreError):
            _RankBootstrap(store, 0, 1, 0.05).connect(transport)
        transport.connect.assert_not_called()
        transport.close.assert_called_once()

    def test_cleanup_preserves_error(self):
        transport = Mock(spec=Transport)
        transport.bind.side_effect = ValueError("bind failed")
        transport.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(ValueError, "bind failed") as error:
            _RankBootstrap(dist.HashStore(), 0, 1, 1).connect(transport)
        self.assertIn("close failed", error.exception.__notes__[0])

    def test_group_relative_ranks_and_duplicate_tag(self):
        store = dist.HashStore()
        group = Mock(spec=dist.ProcessGroup)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed._transport._bootstrap._get_process_group_store",
                return_value=store,
            ),
        ):
            bootstrap = _RankBootstrap.create("test", 0, group, "pair", 1)
            self.assertEqual((bootstrap.rank, bootstrap.peer_rank), (1, 0))
            with self.assertRaisesRegex(ValueError, "already used"):
                _RankBootstrap.create("test", 0, group, "pair", 1)
            for peer in (-1, 1, 2, True):
                with self.assertRaises(ValueError):
                    _RankBootstrap.create("test", peer, group, "other", 1)


if __name__ == "__main__":
    run_tests()
