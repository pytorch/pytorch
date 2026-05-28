#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from integration.helpers.TorchCommTestHelpers import (
    create_store,
    destroy_root_store,
    TorchCommTestWrapper,
    verify_tensor_equality,
    wrap_prefix_store,
)

import torch
from torch.comms import ReduceOp


def store_deletion_barrier(torchcomm):
    # Call barrier to ensure all processes have initialized their TorchComm instances
    # and deleted their store objects, so we can safely create a new one with the
    # same port for the next test
    torchcomm.barrier(False)

    # Synchronize based on device type
    if torch.accelerator.is_available():
        # Calculate device index as rank % num_devices and synchronize
        rank = torchcomm.get_rank()
        device_index = rank % torch.accelerator.device_count()
        torch.accelerator.current_stream(device_index).synchronize()
    # For CPU devices, no synchronization needed


class MultiCommTest(unittest.TestCase):
    """Test class for multiple communicator operations in TorchComm."""

    def setUp(self):
        """Set up test environment before each test."""

    def tearDown(self):
        """Clean up after each test."""

    # Helper function to verify communication within a communicator
    def _verify_communication(self, wrapper):
        """Verify communication within a communicator."""
        # Get the communicator from the wrapper
        comm = wrapper.get_torchcomm()

        # Skip if communicator is null
        if comm is None:
            return

        # Get rank and size from the communicator
        rank = comm.get_rank()
        size = comm.get_size()

        device = comm.get_device()
        options = {"dtype": torch.float, "device": device}
        input_tensor = torch.ones(10, **options) * float(rank + 1)

        # For ranks in groups, test all_reduce
        comm.all_reduce(input_tensor, ReduceOp.SUM, False)

        # Verify the result using tensor equality
        expected_sum = size * (size + 1) / 2
        verify_tensor_equality(input_tensor.cpu(), expected_sum)

    # Helper function to verify simultaneous communication across multiple communicators
    def _verify_simultaneous_communication(self, wrappers):
        """Verify simultaneous communication across multiple communicators."""
        if not wrappers:
            return

        inputs = []
        expected_values = []

        # Create input tensors for each communicator
        for wrapper in wrappers:
            comm = wrapper.get_torchcomm()
            comm_rank = comm.get_rank()
            comm_size = comm.get_size()

            # Calculate device index as rank % num_devices
            device = comm.get_device()
            options = {"dtype": torch.float, "device": device}
            input_tensor = torch.ones(10, **options) * float(comm_rank + 1)
            inputs.append(input_tensor)

            # Calculate expected result for this communicator
            expected = comm_size * (comm_size + 1) / 2
            expected_values.append(expected)

        # Issue all_reduce operations on all communicators simultaneously
        for i, wrapper in enumerate(wrappers):
            wrapper.get_torchcomm().all_reduce(inputs[i], ReduceOp.SUM, False)

        # Verify results for all communicators
        for i, input_tensor in enumerate(inputs):
            description = f"comm_{i} simultaneous all_reduce result"
            verify_tensor_equality(input_tensor.cpu(), expected_values[i], description)

    def test_two_comms_separate_stores(self):
        """Test with two communicators with separate stores."""
        # Create two communicators with separate stores
        wrappers = []
        comms = []

        # Create a store
        store = create_store()

        # Create first communicator with the store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Reset and recreate the store
        store = None
        store_deletion_barrier(comms[-1])
        store = create_store()

        # Create second communicator with the recreated store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Test communication on each communicator individually
        for wrapper in wrappers:
            self._verify_communication(wrapper)

        # Test simultaneous communication across all communicators
        self._verify_simultaneous_communication(wrappers)

        # Clean up store and free MASTER_PORT for subsequent no-store tests
        store = None
        store_deletion_barrier(comms[-1])
        destroy_root_store()

    def test_three_comms_separate_stores(self):
        """Test with three communicators with separate stores."""
        # Create three communicators with separate stores
        wrappers = []
        comms = []

        # Create a store
        store = create_store()

        # Create first communicator with the store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Reset and recreate the store
        store = None
        store_deletion_barrier(comms[-1])
        store = create_store()

        # Create second communicator with the recreated store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Reset and recreate the store
        store = None
        store_deletion_barrier(comms[-1])
        store = create_store()

        # Create third communicator with the recreated store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Test communication on each communicator individually
        for wrapper in wrappers:
            self._verify_communication(wrapper)

        # Test simultaneous communication across all communicators
        self._verify_simultaneous_communication(wrappers)

        # Clean up store and free MASTER_PORT for subsequent no-store tests
        store = None
        store_deletion_barrier(comms[-1])
        destroy_root_store()

    def test_mixed_ops_separate_stores(self):
        """Test mixed operations across multiple communicators with separate stores."""
        # Create two communicators with separate stores
        wrappers = []
        comms = []

        # Create a store
        store = create_store()

        # Create first communicator with the store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Reset and recreate the store
        store = None
        store_deletion_barrier(comms[-1])
        store = create_store()

        # Create second communicator with the recreated store
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Prepare tensors for different operations
        # Calculate device indices as rank % num_devices
        device0 = comms[0].get_device()
        device1 = comms[1].get_device()
        options0 = {"dtype": torch.float, "device": device0}
        options1 = {"dtype": torch.float, "device": device1}

        # For all_reduce on first communicator
        input1 = torch.ones(10, **options0) * float(comms[0].get_rank() + 1)
        expected1 = comms[0].get_size() * (comms[0].get_size() + 1) / 2

        # For broadcast on second communicator
        root_rank = 0
        broadcast_value = 42
        input2 = (
            torch.ones(10, **options1) * float(broadcast_value)
            if comms[1].get_rank() == root_rank
            else torch.zeros(10, **options1)
        )

        # Issue operations simultaneously
        comms[0].all_reduce(input1, ReduceOp.SUM, False)
        comms[1].broadcast(input2, root_rank, False)

        # Verify results
        verify_tensor_equality(input1.cpu(), expected1, "comm_0 all_reduce result")
        verify_tensor_equality(input2.cpu(), broadcast_value, "comm_1 broadcast result")

        # Clean up store and free MASTER_PORT for subsequent no-store tests
        store = None
        store_deletion_barrier(comms[-1])
        destroy_root_store()

    def test_two_comms_no_store(self):
        """Test with two communicators with no store."""
        wrappers = []
        comms = []

        # Create first communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Create second communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Test communication on each communicator individually
        for wrapper in wrappers:
            self._verify_communication(wrapper)

        # Test simultaneous communication across all communicators
        self._verify_simultaneous_communication(wrappers)

    def test_three_comms_no_store(self):
        """Test with three communicators with no store."""
        wrappers = []
        comms = []

        # Create first communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Create second communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Create third communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Test communication on each communicator individually
        for wrapper in wrappers:
            self._verify_communication(wrapper)

        # Test simultaneous communication across all communicators
        self._verify_simultaneous_communication(wrappers)

    def test_mixed_ops_no_store(self):
        """Test mixed operations across multiple communicators with no store."""
        wrappers = []
        comms = []

        # Create first communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Create second communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Prepare tensors for different operations
        device0 = comms[0].get_device()
        device1 = comms[1].get_device()
        options0 = {"dtype": torch.float, "device": device0}
        options1 = {"dtype": torch.float, "device": device1}

        # For all_reduce on first communicator
        input1 = torch.ones(10, **options0) * float(comms[0].get_rank() + 1)
        expected1 = comms[0].get_size() * (comms[0].get_size() + 1) / 2

        # For broadcast on second communicator
        root_rank = 0
        broadcast_value = 42
        input2 = (
            torch.ones(10, **options1) * float(broadcast_value)
            if comms[1].get_rank() == root_rank
            else torch.zeros(10, **options1)
        )

        # Issue operations simultaneously
        comms[0].all_reduce(input1, ReduceOp.SUM, False)
        comms[1].broadcast(input2, root_rank, False)

        # Verify results
        verify_tensor_equality(input1.cpu(), expected1, "comm_0 all_reduce result")
        verify_tensor_equality(input2.cpu(), broadcast_value, "comm_1 broadcast result")

    def test_two_comms_mixed_store(self):
        """Test with two communicators with mixed store (one explicit, one no-store)."""
        wrappers = []
        comms = []

        # Create first communicator using explicit store
        store = create_store()
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Destroy all store references and free MASTER_PORT
        store = None
        store_deletion_barrier(comms[0])
        destroy_root_store()

        # Create second communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Test communication on each communicator individually
        for wrapper in wrappers:
            self._verify_communication(wrapper)

        # Test simultaneous communication across all communicators
        self._verify_simultaneous_communication(wrappers)

    def test_three_comms_mixed_store(self):
        """Test with three communicators with mixed store (two explicit, one no-store)."""
        wrappers = []
        comms = []

        # Create first two communicators using PrefixStore wrappers
        store = create_store()
        store1 = wrap_prefix_store("comm1", store)
        store2 = wrap_prefix_store("comm2", store)

        wrappers.append(TorchCommTestWrapper(store=store1))
        comms.append(wrappers[-1].get_torchcomm())

        wrappers.append(TorchCommTestWrapper(store=store2))
        comms.append(wrappers[-1].get_torchcomm())

        # Destroy all store references and free MASTER_PORT
        store1 = None
        store2 = None
        store = None
        store_deletion_barrier(comms[0])
        destroy_root_store()

        # Create third communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        # Test communication on each communicator individually
        for wrapper in wrappers:
            self._verify_communication(wrapper)

        # Test simultaneous communication across all communicators
        self._verify_simultaneous_communication(wrappers)

    def test_mixed_ops_mixed_store(self):
        """Test mixed operations across multiple communicators with mixed store."""
        wrappers = []
        comms = []

        # Create first communicator using explicit store
        store = create_store()
        wrappers.append(TorchCommTestWrapper(store=store))
        comms.append(wrappers[-1].get_torchcomm())

        # Destroy all store references and free MASTER_PORT
        store = None
        store_deletion_barrier(comms[0])
        destroy_root_store()

        # Create second communicator with no store (internal bootstrap)
        wrappers.append(TorchCommTestWrapper())
        comms.append(wrappers[-1].get_torchcomm())

        device0 = comms[0].get_device()
        device1 = comms[1].get_device()
        options0 = {"dtype": torch.float, "device": device0}
        options1 = {"dtype": torch.float, "device": device1}

        # For all_reduce on first communicator
        input1 = torch.ones(10, **options0) * float(comms[0].get_rank() + 1)
        expected1 = comms[0].get_size() * (comms[0].get_size() + 1) / 2

        # For broadcast on second communicator
        root_rank = 0
        broadcast_value = 42
        input2 = (
            torch.ones(10, **options1) * float(broadcast_value)
            if comms[1].get_rank() == root_rank
            else torch.zeros(10, **options1)
        )

        # Issue operations simultaneously
        comms[0].all_reduce(input1, ReduceOp.SUM, False)
        comms[1].broadcast(input2, root_rank, False)

        # Verify results
        verify_tensor_equality(input1.cpu(), expected1, "comm_0 all_reduce result")
        verify_tensor_equality(input2.cpu(), broadcast_value, "comm_1 broadcast result")


if __name__ == "__main__":
    unittest.main(failfast=True)
