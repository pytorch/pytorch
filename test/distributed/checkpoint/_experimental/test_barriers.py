# Owner(s): ["oncall: distributed checkpointing"]

import unittest.mock as mock

from torch.distributed.checkpoint._experimental.barriers import TCPStoreBarrier
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBarriers(TestCase):
    @mock.patch("torch.distributed.TCPStore")
    @mock.patch("torch.distributed.elastic.utils.store.barrier")
    def test_tcpstore_barrier_initialization(self, _, mock_tcpstore):
        """Test that TCPStoreBarrier initializes correctly."""
        # Setup
        timeout_barrier_init_secs = 60
        barrier_prefix_list = ["test_barrier", "another_barrier"]
        world_size = 4
        use_checkpoint_barrier_tcpstore_libuv = True
        tcpstore_port = 12345
        master_address = "localhost"
        rank = 0
        local_world_size = 1

        # Create the barrier
        barrier = TCPStoreBarrier(
            timeout_barrier_init_secs=timeout_barrier_init_secs,
            barrier_prefix_list=barrier_prefix_list,
            world_size=world_size,
            use_checkpoint_barrier_tcpstore_libuv=use_checkpoint_barrier_tcpstore_libuv,
            tcpstore_port=tcpstore_port,
            master_address=master_address,
            rank=rank,
            local_world_size=local_world_size,
        )

        # Verify that TCPStore was initialized correctly for each barrier prefix
        self.assertEqual(len(barrier._tcp_store_dict), len(barrier_prefix_list))
        for prefix in barrier_prefix_list:
            self.assertIn(prefix, barrier._tcp_store_dict)

        # Verify that TCPStore was initialized with the correct parameters
        mock_tcpstore.assert_any_call(
            master_address,
            tcpstore_port,
            world_size=world_size,
            timeout=mock.ANY,  # timedelta is hard to compare directly
            use_libuv=use_checkpoint_barrier_tcpstore_libuv,
        )

    @mock.patch("torch.distributed.TCPStore")
    @mock.patch("torch.distributed.elastic.utils.store.barrier")
    def test_execute_barrier(self, mock_barrier, mock_tcpstore):
        """Test that execute_barrier calls the barrier function correctly."""
        # Setup
        barrier_prefix = "test_barrier"
        timeout_barrier_init_secs = 60
        barrier_prefix_list = ["test_barrier"]
        world_size = 4
        use_checkpoint_barrier_tcpstore_libuv = True
        tcpstore_port = 12345
        master_address = "localhost"
        rank = 0
        local_world_size = 1
        timeout_secs = 30

        # Mock the TCPStore instance
        mock_tcpstore_instance = mock.MagicMock()
        mock_tcpstore.return_value = mock_tcpstore_instance

        # Create the barrier
        barrier = TCPStoreBarrier(
            timeout_barrier_init_secs=timeout_barrier_init_secs,
            barrier_prefix_list=barrier_prefix_list,
            world_size=world_size,
            use_checkpoint_barrier_tcpstore_libuv=use_checkpoint_barrier_tcpstore_libuv,
            tcpstore_port=tcpstore_port,
            master_address=master_address,
            rank=rank,
            local_world_size=local_world_size,
        )

        # Execute the barrier
        barrier.execute_barrier(barrier_prefix, timeout_secs)

        # Verify that the TCPStore's set method was called with the correct parameters
        mock_tcpstore_instance.set.assert_called_once_with("rank0", "0")

        # Verify that the barrier function was called with the correct parameters
        mock_barrier.assert_called_once_with(
            store=mock_tcpstore_instance,
            world_size=world_size,
            key_prefix=barrier_prefix + "0",
        )

        # Execute the barrier again to test sequence number increment
        barrier.execute_barrier(barrier_prefix, timeout_secs)

        # Verify that the TCPStore's set method was called with the incremented sequence number
        mock_tcpstore_instance.set.assert_called_with("rank0", "1")

        # Verify that the barrier function was called with the incremented sequence number
        mock_barrier.assert_called_with(
            store=mock_tcpstore_instance,
            world_size=world_size,
            key_prefix=barrier_prefix + "1",
        )


if __name__ == "__main__":
    run_tests()
