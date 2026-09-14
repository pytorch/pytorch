# Owner(s): ["oncall: distributed checkpointing"]

import unittest.mock as mock

from torch.distributed.checkpoint._experimental.barriers import TCPStoreBarrier
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBarriers(TestCase):
    @mock.patch("torch.distributed.TCPStore")
    @mock.patch("torch.distributed.elastic.utils.store.barrier")
    def test_tcpstore_barrier_initialization(self, _, mock_tcpstore):
        """Test that TCPStoreBarrier initializes correctly."""
        # Setup
        timeout_barrier_init_secs = 60
        barrier_prefix = "test_barrier"
        world_size = 4
        use_checkpoint_barrier_tcpstore_libuv = True
        tcpstore_port = 12345
        master_address = "localhost"
        rank = 0
        timeout_secs = 30

        # Create rank_info
        rank_info = RankInfo(global_rank=rank, global_world_size=world_size)

        # Create the barrier (used for verification)
        _ = TCPStoreBarrier(
            global_rank=rank_info.global_rank,
            global_world_size=rank_info.global_world_size,
            barrier_prefix=barrier_prefix,
            timeout_barrier_init_secs=timeout_barrier_init_secs,
            use_checkpoint_barrier_tcpstore_libuv=use_checkpoint_barrier_tcpstore_libuv,
            tcpstore_port=tcpstore_port,
            master_address=master_address,
            timeout_secs=timeout_secs,
        )

        # Verify that TCPStore was initialized with the correct parameters
        mock_tcpstore.assert_called_once_with(
            master_address,
            tcpstore_port,
            world_size=rank_info.global_world_size,
            timeout=mock.ANY,  # timedelta is hard to compare directly
            is_master=(rank_info.global_rank == 0),
        )

    @mock.patch("torch.distributed.TCPStore")
    @mock.patch("torch.distributed.elastic.utils.store.barrier")
    def test_execute_barrier(self, mock_barrier, mock_tcpstore):
        """Test that execute_barrier calls the barrier function correctly."""
        # Setup
        barrier_prefix = "test_barrier"
        timeout_barrier_init_secs = 60
        world_size = 4
        use_checkpoint_barrier_tcpstore_libuv = True
        tcpstore_port = 12345
        master_address = "localhost"
        rank = 0
        timeout_secs = 30

        # Create rank_info
        rank_info = RankInfo(global_rank=rank, global_world_size=world_size)

        # Mock the TCPStore instance
        mock_tcpstore_instance = mock.MagicMock()
        mock_tcpstore.return_value = mock_tcpstore_instance

        # Create the barrier
        barrier = TCPStoreBarrier(
            global_rank=rank_info.global_rank,
            global_world_size=rank_info.global_world_size,
            barrier_prefix=barrier_prefix,
            timeout_barrier_init_secs=timeout_barrier_init_secs,
            use_checkpoint_barrier_tcpstore_libuv=use_checkpoint_barrier_tcpstore_libuv,
            tcpstore_port=tcpstore_port,
            master_address=master_address,
            timeout_secs=timeout_secs,
        )

        # Execute the barrier
        barrier.execute_barrier()

        # Verify that the TCPStore's set method was called with the correct parameters
        mock_tcpstore_instance.set.assert_called_once_with("rank0", "0")

        # Verify that the barrier function was called with the correct parameters
        mock_barrier.assert_called_once_with(
            store=mock_tcpstore_instance,
            world_size=rank_info.global_world_size,
            key_prefix=barrier_prefix + "0",
        )

        # Execute the barrier again to test sequence number increment
        barrier.execute_barrier()

        # Verify that the TCPStore's set method was called with the incremented sequence number
        mock_tcpstore_instance.set.assert_called_with("rank0", "1")

        # Verify that the barrier function was called with the incremented sequence number
        mock_barrier.assert_called_with(
            store=mock_tcpstore_instance,
            world_size=rank_info.global_world_size,
            key_prefix=barrier_prefix + "1",
        )


if __name__ == "__main__":
    run_tests()
