#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import unittest
from unittest.mock import MagicMock, patch

from torch.distributed.elastic.events import EventMetadataValue, EventSource
from torch.distributed.elastic.multiprocessing.api import _construct_event
from torch.distributed.elastic.multiprocessing.subprocess_handler import (
    SubprocessHandler,
)


class TestSubprocessHandler(unittest.TestCase):
    @patch(
        "torch.distributed.elastic.multiprocessing.subprocess_handler.SubprocessHandler._popen"
    )
    def test_subprocess_handler_init_with_rank(self, mock_popen):
        # Mock the open function to return a mock file object

        # Mock the Popen object
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        # Initialize SubprocessHandler
        handler = SubprocessHandler(
            entrypoint="echo",
            args=("Hello, World!",),
            env={"RANK": "0"},
            stdout=None,
            stderr=None,
            local_rank_id=0,
        )

        self.assertEqual(handler.local_rank_id, 0)
        self.assertEqual(handler.global_rank, "0")

    @patch(
        "torch.distributed.elastic.multiprocessing.subprocess_handler.SubprocessHandler._popen"
    )
    def test_subprocess_handler_init_without_rank_env(self, mock_popen):
        # Mock the open function to return a mock file object

        # Mock the Popen object
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        # Initialize SubprocessHandler
        handler = SubprocessHandler(
            entrypoint="echo",
            args=("Hello, World!",),
            env={"LOCAL_RANK": "0"},
            stdout=None,
            stderr=None,
            local_rank_id=0,
        )

        self.assertEqual(handler.local_rank_id, 0)
        self.assertEqual(handler.global_rank, None)

    def test_construct_event(self):
        source: EventSource = EventSource.WORKER
        global_rank = 1
        raw_error = "Error message"
        local_rank = 0

        expected_metadata: dict[str, EventMetadataValue] = {}
        expected_metadata["global_rank"] = global_rank
        expected_metadata["raw_error"] = raw_error
        expected_metadata["local_rank"] = local_rank

        event = _construct_event(source, global_rank, raw_error, local_rank)

        self.assertEqual(event.name, "torchelastic.worker.closure")
        self.assertEqual(event.source, source)
        self.assertEqual(event.metadata, expected_metadata)
