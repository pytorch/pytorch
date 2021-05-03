# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import codecs
import copy
import os
import pickle
import socket
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple, cast
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from torch.distributed import Store
from torch.distributed.elastic.rendezvous import RendezvousParameters, RendezvousStateError
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    DynamicRendezvousHandler,
    RendezvousBackend,
    RendezvousSettings,
    RendezvousTimeout,
    Token,
    _BackendRendezvousStateHolder,
    _NodeDesc,
    _NodeDescGenerator,
    _RendezvousState,
    create_handler,
)


class CustomAssertMixin:
    assertDictEqual: Callable

    def assert_state_equal(self, actual: _RendezvousState, expected: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(expected))

    def assert_state_empty(self, actual: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(_RendezvousState()))


class RendezvousTimeoutTest(TestCase):
    def test_init_initializes_timeout(self) -> None:
        timeout = RendezvousTimeout(
            timedelta(seconds=50),
            timedelta(seconds=60),
            timedelta(seconds=70),
            timedelta(seconds=80),
        )

        self.assertEqual(timeout.join, timedelta(seconds=50))
        self.assertEqual(timeout.last_call, timedelta(seconds=60))
        self.assertEqual(timeout.close, timedelta(seconds=70))
        self.assertEqual(timeout.heartbeat, timedelta(seconds=80))

    def test_init_initializes_timeout_if_no_timeout_is_specified(self) -> None:
        timeout = RendezvousTimeout()

        self.assertEqual(timeout.join, timedelta(seconds=600))
        self.assertEqual(timeout.last_call, timedelta(seconds=30))
        self.assertEqual(timeout.close, timedelta(seconds=30))
        self.assertEqual(timeout.heartbeat, timedelta(seconds=5))

    def test_init_raises_error_if_timeout_is_not_positive(self) -> None:
        join_timeouts = [timedelta(seconds=0), timedelta(seconds=-1)]

        for join_timeout in join_timeouts:
            with self.subTest(join_timeout=join_timeout):
                with self.assertRaisesRegex(
                    ValueError, rf"^The join timeout \({join_timeout}\) must be positive.$"
                ):
                    timeout = RendezvousTimeout(join_timeout)


class NodeDescTest(TestCase):
    def test_repr(self) -> None:
        desc = _NodeDesc("dummy_fqdn", 3, 5)

        self.assertEqual(repr(desc), "dummy_fqdn_3_5")

    def test_hash(self) -> None:
        desc1 = _NodeDesc("dummy_fqdn", 2, 4)
        desc2 = _NodeDesc("dummy_fqdn", 3, 5)

        descs = {desc1, desc2}

        self.assertIn(desc1, descs)
        self.assertIn(desc2, descs)


class NodeDescGeneratorTest(TestCase):
    def test_generate(self) -> None:
        desc_generator = _NodeDescGenerator()

        fqdn = socket.getfqdn()

        pid = os.getpid()

        for local_id in range(4):
            with self.subTest(fqdn=fqdn, pid=pid, local_id=local_id):
                desc = desc_generator.generate()

                self.assertEqual(repr(desc), f"{fqdn}_{pid}_{local_id}")


class RendezvousStateTest(TestCase):
    def test_encoded_size_is_within_expected_limit(self) -> None:
        state = _RendezvousState()
        state.round = 1
        state.complete = True
        state.deadline = datetime.utcnow()
        state.closed = True

        # fmt: off
        expected_max_sizes = (
            (   5,    2 * (2 ** 10),),  #    10 machines <=   2KB  # noqa: E201, E241, E262
            (  50,   16 * (2 ** 10),),  #   100 machines <=  16KB  # noqa: E201, E241, E262
            ( 500,  160 * (2 ** 10),),  #  1000 machines <= 160KB  # noqa: E201, E241, E262
            (5000, 1600 * (2 ** 10),),  # 10000 machines <= 1.6MB  # noqa: E201, E241, E262
        )
        # fmt: on

        for num_nodes, max_byte_size in expected_max_sizes:
            with self.subTest(num_nodes=num_nodes, max_byte_size=max_byte_size):
                for i in range(num_nodes):
                    node_running = _NodeDesc(f"dummy{i}.dummy1-dummy1-dummy1-dummy1.com", 12345, i)
                    node_waiting = _NodeDesc(f"dummy{i}.dummy2-dummy2-dummy2-dummy2.com", 67890, i)

                    state.participants[node_running] = i

                    state.wait_list.add(node_waiting)

                    state.last_heartbeats[node_running] = datetime.utcnow()
                    state.last_heartbeats[node_waiting] = datetime.utcnow()

                bits = pickle.dumps(state)

                base64_bits = codecs.encode(bits, "base64")

                self.assertLessEqual(len(base64_bits), max_byte_size)


class FakeRendezvousBackend(RendezvousBackend):
    _state: Optional[bytes]
    _token: int

    def __init__(self) -> None:
        self._state = None
        self._token = 0

    @property
    def name(self) -> str:
        return "fake_backend"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        if self._token == 0:
            return None

        return self._state, self._token  # type: ignore[return-value]

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        if token is None:
            token = 0

        if token == self._token:
            self._state = state
            self._token += 1

            has_set = True
        else:
            has_set = False

        return self._state, self._token, has_set  # type: ignore[return-value]

    def get_state_internal(self) -> _RendezvousState:
        return pickle.loads(cast(bytes, self._state))

    def set_state_internal(self, state: _RendezvousState) -> None:
        self._state = pickle.dumps(state)
        self._token += 1

    def corrupt_state(self) -> None:
        self._state = b"corrupt_state"
        self._token += 1


class BackendRendezvousStateHolderTest(TestCase, CustomAssertMixin):
    def setUp(self) -> None:
        self._backend = FakeRendezvousBackend()

        mock_get_state = MagicMock(wraps=self._backend.get_state)
        mock_set_state = MagicMock(wraps=self._backend.set_state)

        self._mock_backend = Mock()
        self._mock_backend.get_state = mock_get_state
        self._mock_backend.set_state = mock_set_state

        setattr(self._backend, "get_state", mock_get_state)  # noqa: B010
        setattr(self._backend, "set_state", mock_set_state)  # noqa: B010

        self._settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            timeout=RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

        self._cache_duration = 0

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now

    def tearDown(self) -> None:
        self._datetime_patch.stop()

    def _create_state(self) -> _RendezvousState:
        state = _RendezvousState()
        state.round = 999
        state.complete = True
        state.deadline = self._now
        state.closed = True
        state.participants = {
            _NodeDesc("dummy1", 1, 1): 0,
            _NodeDesc("dummy2", 1, 1): 1,
            _NodeDesc("dummy3", 1, 1): 2,
        }
        state.wait_list = {
            _NodeDesc("dummy4", 1, 1),
            _NodeDesc("dummy5", 1, 1),
        }
        state.last_heartbeats = {
            _NodeDesc("dummy1", 1, 1): self._now,
            _NodeDesc("dummy2", 1, 1): self._now - timedelta(seconds=15),
            _NodeDesc("dummy3", 1, 1): self._now - timedelta(seconds=30),
            _NodeDesc("dummy4", 1, 1): self._now - timedelta(seconds=60),
            _NodeDesc("dummy5", 1, 1): self._now - timedelta(seconds=90),
        }

        return state

    def _create_state_holder(self) -> _BackendRendezvousStateHolder:
        return _BackendRendezvousStateHolder(self._backend, self._settings, self._cache_duration)

    def test_init_initializes_state_holder(self) -> None:
        state_holder = self._create_state_holder()

        self.assert_state_empty(state_holder.state)

        self._mock_backend.assert_not_called()

    def test_sync_gets_empty_state_if_backend_state_does_not_exist(self) -> None:
        state_holder = self._create_state_holder()

        has_set = state_holder.sync()

        self.assertIsNone(has_set)

        self.assert_state_empty(state_holder.state)

        self.assertEqual(self._mock_backend.get_state.call_count, 1)
        self.assertEqual(self._mock_backend.set_state.call_count, 0)

    def test_sync_gets_backend_state_if_local_state_is_clean(self) -> None:
        state_holder = self._create_state_holder()

        expected_state = self._create_state()

        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                expected_state.round = attempt

                self._backend.set_state_internal(expected_state)

                has_set = state_holder.sync()

                self.assertIsNone(has_set)

                self.assert_state_equal(state_holder.state, expected_state)

                self.assertEqual(self._mock_backend.get_state.call_count, 1)
                self.assertEqual(self._mock_backend.set_state.call_count, 0)

                self._mock_backend.reset_mock()

    def test_sync_gets_backend_state_if_local_state_is_old_and_dirty(self) -> None:
        state_holder = self._create_state_holder()

        expected_state = self._create_state()

        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                self._backend.set_state_internal(expected_state)  # Increment token.

                state_holder.state.round = attempt
                state_holder.mark_dirty()

                has_set = state_holder.sync()

                self.assertFalse(has_set)

                self.assert_state_equal(state_holder.state, expected_state)

                self.assertEqual(self._mock_backend.get_state.call_count, 0)
                self.assertEqual(self._mock_backend.set_state.call_count, 1)

                self._mock_backend.reset_mock()

    def test_sync_sets_backend_state_if_local_state_is_new_and_dirty(self) -> None:
        state_holder = self._create_state_holder()

        for attempt in range(1, 4):
            with self.subTest(attempt=attempt):
                state_holder.state.round = attempt
                state_holder.mark_dirty()

                has_set = state_holder.sync()

                self.assertTrue(has_set)

                expected_state = self._backend.get_state_internal()

                self.assert_state_equal(state_holder.state, expected_state)

                self.assertEqual(self._mock_backend.get_state.call_count, 0)
                self.assertEqual(self._mock_backend.set_state.call_count, 1)

                self._mock_backend.reset_mock()

    def test_sync_uses_cached_state_if_cache_duration_is_specified(self) -> None:
        state = self._create_state()

        self._backend.set_state_internal(state)

        with patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous.time") as mock_time:
            for cache_duration in [1, 5, 10]:
                with self.subTest(cache_duration=cache_duration):
                    self._cache_duration = cache_duration

                    state_holder = self._create_state_holder()

                    mock_time.monotonic.return_value = 5

                    state_holder.sync()

                    has_set = state_holder.sync()

                    self.assertIsNone(has_set)

                    self.assertEqual(self._mock_backend.get_state.call_count, 1)
                    self.assertEqual(self._mock_backend.set_state.call_count, 0)

                    mock_time.monotonic.return_value = 5 + self._cache_duration

                    state_holder.sync()

                    has_set = state_holder.sync()

                    self.assertIsNone(has_set)

                    self.assertEqual(self._mock_backend.get_state.call_count, 1)
                    self.assertEqual(self._mock_backend.set_state.call_count, 0)

                    self._mock_backend.get_state.reset_mock()

    def test_sync_gets_backend_state_if_cached_state_has_expired(self) -> None:
        state = self._create_state()

        self._backend.set_state_internal(state)

        with patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous.time") as mock_time:
            self._cache_duration = 1

            state_holder = self._create_state_holder()

            mock_time.monotonic.return_value = 5

            state_holder.sync()

            has_set = state_holder.sync()

            self.assertIsNone(has_set)

            self.assertEqual(self._mock_backend.get_state.call_count, 1)
            self.assertEqual(self._mock_backend.set_state.call_count, 0)

            mock_time.monotonic.return_value = 5 + self._cache_duration + 0.01

            state_holder.sync()

            has_set = state_holder.sync()

            self.assertIsNone(has_set)

            self.assertEqual(self._mock_backend.get_state.call_count, 2)
            self.assertEqual(self._mock_backend.set_state.call_count, 0)

    def test_sync_sanitizes_state(self) -> None:
        expected_state = self._create_state()

        state = copy.deepcopy(expected_state)

        dead_node1 = _NodeDesc("dead1", 1, 1)
        dead_node2 = _NodeDesc("dead2", 1, 1)
        dead_node3 = _NodeDesc("dead3", 1, 1)
        dead_node4 = _NodeDesc("dead4", 1, 1)
        dead_node5 = _NodeDesc("dead5", 1, 1)

        state.last_heartbeats[dead_node1] = self._now - timedelta(seconds=91)
        state.last_heartbeats[dead_node2] = self._now - timedelta(seconds=100)
        state.last_heartbeats[dead_node3] = self._now - timedelta(seconds=110)
        state.last_heartbeats[dead_node4] = self._now - timedelta(seconds=120)
        state.last_heartbeats[dead_node5] = self._now - timedelta(seconds=130)

        state.participants[dead_node1] = 0
        state.participants[dead_node2] = 0
        state.participants[dead_node3] = 0

        state.wait_list.add(dead_node4)
        state.wait_list.add(dead_node5)

        self._backend.set_state_internal(state)

        state_holder = self._create_state_holder()

        state_holder.sync()

        self.assert_state_equal(state_holder.state, expected_state)

    def test_sync_raises_error_if_backend_state_is_corrupt(self) -> None:
        self._backend.corrupt_state()

        state_holder = self._create_state_holder()

        with self.assertRaisesRegex(
            RendezvousStateError,
            r"^The rendezvous state is corrupt. See inner exception for details.$",
        ):
            state_holder.sync()


class DummyStore(Store):
    pass


class DummyRendezvousBackend(RendezvousBackend):
    @property
    def name(self):
        return "dummy_backend"

    def get_state(self):
        return None

    def set_state(self, state, token):
        return None


class DynamicRendezvousHandlerTest(TestCase):
    def setUp(self) -> None:
        self._run_id = "dummy_run_id"
        self._store = DummyStore()
        self._backend = DummyRendezvousBackend()
        self._min_nodes = 3
        self._max_nodes = 6
        self._timeout: Optional[RendezvousTimeout] = RendezvousTimeout()

    def _create_handler(self) -> DynamicRendezvousHandler:
        return DynamicRendezvousHandler(
            run_id=self._run_id,
            store=self._store,
            backend=self._backend,
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
        )

    def test_init_initializes_handler(self) -> None:
        handler = self._create_handler()

        self.assertIs(handler.store, self._store)
        self.assertIs(handler.backend, self._backend)

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._run_id)
        self.assertEqual(handler.settings.run_id, self._run_id)
        self.assertEqual(handler.settings.min_nodes, self._min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._max_nodes)

        if self._timeout is None:
            self.assertIsNotNone(handler.settings.timeout)
        else:
            self.assertIs(handler.settings.timeout, self._timeout)

    def test_init_initializes_handler_if_timeout_is_not_specified(self) -> None:
        self._timeout = None

        self.test_init_initializes_handler()

    def test_init_initializes_handler_if_min_and_max_nodes_are_equal(self) -> None:
        self._min_nodes = 3
        self._max_nodes = 3

        self.test_init_initializes_handler()

    def test_init_raises_error_if_min_nodes_is_not_positive(self) -> None:
        for num in [0, -10]:
            with self.subTest(min_nodes=num):
                self._min_nodes = num

                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The minimum number of nodes \({num}\) must be greater than zero.$",
                ):
                    self._create_handler()

    def test_init_raises_error_if_max_nodes_is_less_than_min(self) -> None:
        self._min_nodes = 3
        self._max_nodes = 2

        with self.assertRaisesRegex(
            ValueError,
            rf"^The maximum number of nodes \({self._max_nodes}\) must be greater than or equal to "
            "the minimum number of nodes "
            rf"\({self._min_nodes}\).$",
        ):
            self._create_handler()


class CreateHandlerTest(TestCase):
    def setUp(self) -> None:
        self._store = DummyStore()

        self._backend = DummyRendezvousBackend()

        self._params = RendezvousParameters(
            backend=self._backend.name,
            endpoint="dummy_endpoint",
            run_id="dummy_run_id",
            min_nodes=3,
            max_nodes=6,
            store_port="1234",
            join_timeout="50",
            last_call_timeout="60",
            close_timeout="70",
        )

        self._expected_timeout = RendezvousTimeout(
            timedelta(seconds=50), timedelta(seconds=60), timedelta(seconds=70)
        )

    def test_create_handler_returns_handler(self) -> None:
        handler = create_handler(self._store, self._backend, self._params)

        self.assertIs(handler.store, self._store)
        self.assertIs(handler.backend, self._backend)

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._params.run_id)
        self.assertEqual(handler.settings.min_nodes, self._params.min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._params.max_nodes)
        self.assertEqual(handler.settings.timeout.join, self._expected_timeout.join)
        self.assertEqual(handler.settings.timeout.last_call, self._expected_timeout.last_call)
        self.assertEqual(handler.settings.timeout.close, self._expected_timeout.close)

    def test_create_handler_returns_handler_if_timeout_is_not_specified(self) -> None:
        del self._params.config["join_timeout"]
        del self._params.config["last_call_timeout"]
        del self._params.config["close_timeout"]

        self._expected_timeout = RendezvousTimeout()

        self.test_create_handler_returns_handler()
