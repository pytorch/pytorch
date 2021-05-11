#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import logging
import unittest
from unittest.mock import patch

from torch.distributed.elastic.events import _get_or_create_logger, Event, EventSource
from torch.testing._internal.common_utils import run_tests


class EventLibTest(unittest.TestCase):
    def assert_event(self, actual_event, expected_event):
        self.assertEqual(actual_event.name, expected_event.name)
        self.assertEqual(actual_event.source, expected_event.source)
        self.assertEqual(actual_event.timestamp, expected_event.timestamp)
        self.assertDictEqual(actual_event.metadata, expected_event.metadata)

    @patch("torch.distributed.elastic.events.get_logging_handler")
    def test_get_or_create_logger(self, logging_handler_mock):
        logging_handler_mock.return_value = logging.NullHandler()
        logger = _get_or_create_logger("test_destination")
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)

    def test_event_created(self):
        event = Event(
            name="test_event",
            source=EventSource.AGENT,
            metadata={"key1": "value1", "key2": 2},
        )
        self.assertEqual("test_event", event.name)
        self.assertEqual(EventSource.AGENT, event.source)
        self.assertDictEqual({"key1": "value1", "key2": 2}, event.metadata)

    def test_event_deser(self):
        event = Event(
            name="test_event",
            source=EventSource.AGENT,
            metadata={"key1": "value1", "key2": 2, "key3": 1.0},
        )
        json_event = event.serialize()
        deser_event = Event.deserialize(json_event)
        self.assert_event(event, deser_event)


if __name__ == "__main__":
    run_tests()
