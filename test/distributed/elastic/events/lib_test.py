#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs

import json
import logging
from dataclasses import asdict
from unittest.mock import patch

from torch.distributed.elastic.events import (
    Event,
    EventSource,
    NodeState,
    RdzvEvent,
    _get_or_create_logger,
    construct_and_record_rdzv_event,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class EventLibTest(TestCase):
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

class RdzvEventLibTest(TestCase):
    @patch("torch.distributed.elastic.events.record_rdzv_event")
    @patch("torch.distributed.elastic.events.get_logging_handler")
    def test_construct_and_record_rdzv_event(self, get_mock, record_mock):
        get_mock.return_value = logging.StreamHandler()
        construct_and_record_rdzv_event(
            run_id="test_run_id",
            message="test_message",
            node_state=NodeState.RUNNING,
        )
        record_mock.assert_called_once()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    @patch("torch.distributed.elastic.events.get_logging_handler")
    def test_construct_and_record_rdzv_event_does_not_run_if_invalid_dest(self, get_mock, record_mock):
        get_mock.return_value = logging.NullHandler()
        construct_and_record_rdzv_event(
            run_id="test_run_id",
            message="test_message",
            node_state=NodeState.RUNNING,
        )
        record_mock.assert_not_called()

    def assert_rdzv_event(self, actual_event: RdzvEvent, expected_event: RdzvEvent):
        self.assertEqual(actual_event.name, expected_event.name)
        self.assertEqual(actual_event.run_id, expected_event.run_id)
        self.assertEqual(actual_event.message, expected_event.message)
        self.assertEqual(actual_event.hostname, expected_event.hostname)
        self.assertEqual(actual_event.pid, expected_event.pid)
        self.assertEqual(actual_event.node_state, expected_event.node_state)
        self.assertEqual(actual_event.master_endpoint, expected_event.master_endpoint)
        self.assertEqual(actual_event.rank, expected_event.rank)
        self.assertEqual(actual_event.local_id, expected_event.local_id)
        self.assertEqual(actual_event.error_trace, expected_event.error_trace)

    def get_test_rdzv_event(self) -> RdzvEvent:
        return RdzvEvent(
            name="test_name",
            run_id="test_run_id",
            message="test_message",
            hostname="test_hostname",
            pid=1,
            node_state=NodeState.RUNNING,
            master_endpoint="test_master_endpoint",
            rank=3,
            local_id=4,
            error_trace="test_error_trace",
        )

    def test_rdzv_event_created(self):
        event = self.get_test_rdzv_event()
        self.assertEqual(event.name, "test_name")
        self.assertEqual(event.run_id, "test_run_id")
        self.assertEqual(event.message, "test_message")
        self.assertEqual(event.hostname, "test_hostname")
        self.assertEqual(event.pid, 1)
        self.assertEqual(event.node_state, NodeState.RUNNING)
        self.assertEqual(event.master_endpoint, "test_master_endpoint")
        self.assertEqual(event.rank, 3)
        self.assertEqual(event.local_id, 4)
        self.assertEqual(event.error_trace, "test_error_trace")


    def test_rdzv_event_deserialize(self):
        event = self.get_test_rdzv_event()
        json_event = event.serialize()
        deserialized_event = RdzvEvent.deserialize(json_event)
        self.assert_rdzv_event(event, deserialized_event)
        self.assert_rdzv_event(event, RdzvEvent.deserialize(event))

    def test_rdzv_event_str(self):
        event = self.get_test_rdzv_event()
        self.assertEqual(str(event), json.dumps(asdict(event)))


if __name__ == "__main__":
    run_tests()
