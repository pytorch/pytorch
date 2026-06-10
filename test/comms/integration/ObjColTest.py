#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest
from contextlib import contextmanager
from datetime import timedelta

from integration.helpers.TorchCommTestHelpers import is_full_sweep, TorchCommTestWrapper

import torch
from torch.comms import objcol


@contextmanager
def report_error():
    try:
        yield
    except unittest.SkipTest:
        raise
    except Exception as e:
        import traceback

        print(f"Error: {e}")
        traceback.print_exception(e)
        raise e


class ObjColTest(unittest.TestCase):
    """Test class for broadcast operations in TorchComm."""

    # Class variables for test parameters
    counts = [0, 4, 1024, 1024 * 1024] if is_full_sweep() else [4, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8] if is_full_sweep() else [torch.float]
    num_replays = 4

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    @report_error()
    def test_all_gather_object(self) -> None:
        inp = str(self.rank)
        out = [None] * self.num_ranks
        expected = [str(i) for i in range(self.num_ranks)]
        objcol.all_gather_object(
            self.torchcomm,
            object_list=out,
            obj=inp,
            timeout=timedelta(seconds=60),
            weights_only=True,
        )
        self.assertEqual(out, expected)

    @report_error()
    def test_gather_object(self) -> None:
        """Test gathering objects from all ranks to root rank only."""
        root = 0
        inp = f"rank_{self.rank}_data"
        expected = [f"rank_{i}_data" for i in range(self.num_ranks)]

        # Root rank provides the gather list
        gather_list = [None] * self.num_ranks

        objcol.gather_object(
            self.torchcomm,
            obj=inp,
            root=root,
            object_gather_list=gather_list,
            timeout=timedelta(seconds=60),
            weights_only=True,
        )

        # Only root rank should have the gathered objects
        if self.rank == root:
            self.assertEqual(gather_list, expected)

    @report_error()
    def test_send_recv_object_list(self) -> None:
        """Test point-to-point object list communication."""
        if self.num_ranks < 2:
            self.skipTest("This test requires at least 2 ranks.")

        sender_rank = 0
        receiver_rank = 1

        if self.rank == sender_rank:
            # Sender prepares list of objects to send
            objects_to_send = [
                f"string_from_rank_{self.rank}",
                42 + self.rank,
                {"rank": self.rank, "data": "test"},
            ]
            objcol.send_object_list(
                self.torchcomm,
                object_list=objects_to_send,
                dst=receiver_rank,
                timeout=timedelta(seconds=60),
            )
        elif self.rank == receiver_rank:
            # Receiver prepares empty list to receive objects
            received_objects = [None, None, None]
            objcol.recv_object_list(
                self.torchcomm,
                object_list=received_objects,
                src=sender_rank,
                timeout=timedelta(seconds=60),
                weights_only=True,
            )
            # Verify received objects
            expected = [
                f"string_from_rank_{sender_rank}",
                42 + sender_rank,
                {"rank": sender_rank, "data": "test"},
            ]
            self.assertEqual(received_objects, expected)

    @report_error()
    def test_broadcast_object_list(self) -> None:
        """Test broadcasting object list from root to all ranks."""
        root = 0

        if self.rank == root:
            # Root rank provides the objects to broadcast
            objects = [
                "broadcast_string",
                123,
                {"broadcast": True, "root": root},
                [1, 2, 3],
            ]
        else:
            # Non-root ranks prepare empty list of same size
            objects = [None, None, None, None]

        objcol.broadcast_object_list(
            self.torchcomm,
            object_list=objects,
            root=root,
            timeout=timedelta(seconds=60),
            weights_only=True,
        )

        # All ranks should have the same objects after broadcast
        expected = [
            "broadcast_string",
            123,
            {"broadcast": True, "root": root},
            [1, 2, 3],
        ]
        self.assertEqual(objects, expected)

    @report_error()
    def test_scatter_object_list(self) -> None:
        """Test scattering objects from root to all ranks."""
        root = 0

        # Root rank provides list of objects to scatter (one per rank)
        scatter_input = [f"object_for_rank_{i}" for i in range(self.num_ranks)]

        # All ranks prepare output list with one element
        scatter_output = [None]

        objcol.scatter_object_list(
            self.torchcomm,
            root=root,
            scatter_object_output_list=scatter_output,
            scatter_object_input_list=scatter_input,
            timeout=timedelta(seconds=60),
            weights_only=True,
        )

        # Each rank should receive the object intended for it
        expected_object = f"object_for_rank_{self.rank}"
        self.assertEqual(scatter_output[0], expected_object)


class ObjColTestWithPickle(ObjColTest):
    def setUp(self):
        """Set up test environment before each test."""
        import os

        os.environ["TORCHCOMMS_SERIALIZATION"] = "pickle"
        super().setUp()


if __name__ == "__main__":
    unittest.main()
