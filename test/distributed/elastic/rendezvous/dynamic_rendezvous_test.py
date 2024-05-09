# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import pickle
import socket
import threading
import time
from abc import ABC, abstractmethod
from base64 import b64encode
from datetime import datetime, timedelta
from typing import Callable, cast, Optional, Tuple
from unittest import TestCase
from unittest.mock import call, MagicMock, Mock, patch

from torch.distributed import Store
from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousTimeoutError,
)
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    _Action,
    _BackendRendezvousStateHolder,
    _DistributedRendezvousOpExecutor,
    _NodeDesc,
    _NodeDescGenerator,
    _RendezvousCloseOp,
    _RendezvousContext,
    _RendezvousExitOp,
    _RendezvousJoinOp,
    _RendezvousKeepAliveOp,
    _RendezvousState,
    _RendezvousStateHolder,
    create_handler,
    DynamicRendezvousHandler,
    RendezvousBackend,
    RendezvousSettings,
    RendezvousTimeout,
    Token,
)


class CustomAssertMixin:
    assertDictEqual: Callable

    def assert_state_equal(
        self, actual: _RendezvousState, expected: _RendezvousState
    ) -> None:
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
                    ValueError,
                    rf"^The join timeout \({join_timeout}\) must be positive.$",
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
                    node_running = _NodeDesc(
                        f"dummy{i}.dummy1-dummy1-dummy1-dummy1.com", 12345, i
                    )
                    node_waiting = _NodeDesc(
                        f"dummy{i}.dummy2-dummy2-dummy2-dummy2.com", 67890, i
                    )

                    state.participants[node_running] = i

                    state.wait_list.add(node_waiting)

                    state.last_heartbeats[node_running] = datetime.utcnow()
                    state.last_heartbeats[node_waiting] = datetime.utcnow()

                bits = pickle.dumps(state)

                base64_bits = b64encode(bits)

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
        return _BackendRendezvousStateHolder(
            self._backend, self._settings, self._cache_duration
        )

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

        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        ) as mock_time:
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

        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        ) as mock_time:
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
        state = self._create_state()

        expected_state = copy.deepcopy(state)

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

    def test_sync_sanitizes_state_if_no_participants_is_left(self) -> None:
        state = self._create_state()

        expected_state = copy.deepcopy(state)

        for node in state.last_heartbeats:
            state.last_heartbeats[node] = self._now - timedelta(seconds=100)

        expected_state.complete = False
        expected_state.round = 1000
        expected_state.participants = {}
        expected_state.wait_list = set()
        expected_state.last_heartbeats = {}

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


class FakeRendezvousStateHolder(_RendezvousStateHolder):
    _state: _RendezvousState
    _dirty: Optional[bool]

    def __init__(self) -> None:
        self._state = _RendezvousState()
        self._dirty = None

    @property
    def state(self) -> _RendezvousState:
        return self._state

    @state.setter
    def state(self, value) -> None:
        self._state = value

    def sync(self) -> Optional[bool]:
        self._dirty, dirty = None, self._dirty

        return dirty

    def mark_dirty(self) -> None:
        self._dirty = True


class DistributedRendezvousOpExecutorTest(TestCase, CustomAssertMixin):
    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)

        self._state_holder = FakeRendezvousStateHolder()

        mock_sync = MagicMock(wraps=self._state_holder.sync)
        mock_mark = MagicMock(wraps=self._state_holder.mark_dirty)

        self._mock_state_holder = Mock()
        self._mock_state_holder.sync = mock_sync
        self._mock_state_holder.mark = mock_mark

        setattr(self._state_holder, "sync", mock_sync)  # noqa: B010
        setattr(self._state_holder, "mark_dirty", mock_mark)  # noqa: B010

        self._state = self._state_holder.state

        self._min_nodes = 1
        self._max_nodes = 1

        self._timeout = RendezvousTimeout()

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now

    def tearDown(self) -> None:
        self._datetime_patch.stop()

    def _create_settings(self) -> RendezvousSettings:
        return RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
            keep_alive_interval=timedelta(seconds=30),
            keep_alive_max_attempt=3,
        )

    def _create_op_executor(
        self, settings: Optional[RendezvousSettings] = None
    ) -> _DistributedRendezvousOpExecutor:
        self._state_holder.state = self._state

        if settings is None:
            settings = self._create_settings()

        return _DistributedRendezvousOpExecutor(
            self._node, self._state_holder, settings
        )

    def _run_action(self, action: _Action) -> None:
        op_executor = self._create_op_executor()

        op = MagicMock(side_effect=[action, _Action.FINISH])

        op_executor.run(op, deadline=1)

    def _assert_action(self, action: _Action, expected_state: _RendezvousState) -> None:
        self._run_action(action)

        self.assert_state_equal(self._state, expected_state)

        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.mark(), call.sync()]
        )

    def test_run_passes_expected_context_and_deadline_to_state_handler(self) -> None:
        settings = self._create_settings()

        op_executor = self._create_op_executor(settings)

        op = MagicMock(return_value=_Action.FINISH)

        op_executor.run(op, deadline=3)

        ctx, deadline = op.call_args[0]  # args

        self.assertIs(ctx.node, self._node)
        self.assertIs(ctx.state, self._state)
        self.assertIs(ctx.settings, settings)

        self.assertEqual(deadline, 3)

    def test_run_keeps_alive(self) -> None:
        expected_state = _RendezvousState()

        expected_state.last_heartbeats[self._node] = self._now

        self._assert_action(_Action.KEEP_ALIVE, expected_state)

    def test_run_adds_to_participants(self) -> None:
        expected_state = _RendezvousState()

        expected_state.participants[self._node] = 0

        expected_state.last_heartbeats[self._node] = self._now

        self._min_nodes = 2
        self._max_nodes = 2

        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    def test_run_adds_to_participants_if_node_was_in_waitlist(self) -> None:
        self._state.wait_list.add(self._node)

        expected_state = _RendezvousState()

        expected_state.participants[self._node] = 0

        expected_state.last_heartbeats[self._node] = self._now

        self._min_nodes = 2
        self._max_nodes = 2

        self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

    def _add_participants(
        self, num_participants: int, state: _RendezvousState, ranked: bool = False
    ) -> None:
        for i in range(num_participants):
            if ranked:
                node = _NodeDesc(f"dummy{i}", 1, 1)
                rank = i
            else:
                node = _NodeDesc(
                    f"dummy{num_participants - i - 1}", 1, 1
                )  # Add in reverse.
                rank = 0

            state.participants[node] = rank

            state.last_heartbeats[node] = self._now

    def test_run_adds_to_participants_and_starts_last_call_if_min_nodes_is_reached(
        self,
    ) -> None:
        for num_participants in range(3):
            self._state = _RendezvousState()

            self._add_participants(num_participants, self._state)

            self._state.wait_list.add(self._node)

            expected_state = _RendezvousState()

            self._add_participants(num_participants, expected_state)

            expected_state.participants[self._node] = 0

            expected_state.last_heartbeats[self._node] = self._now

            expected_state.deadline = self._now + self._timeout.last_call

            with self.subTest(num_participants=num_participants):
                self._min_nodes = num_participants + 1
                self._max_nodes = num_participants + 2

                self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

                self._mock_state_holder.reset_mock()

    def test_run_adds_to_participants_and_completes_rendezvous_if_max_nodes_is_reached(
        self,
    ) -> None:
        for min_max_nodes_equal in [False, True]:
            for num_participants in range(3):
                rank = num_participants

                self._state = _RendezvousState()

                self._add_participants(num_participants, self._state)

                self._state.wait_list.add(self._node)

                self._state.deadline = self._now + self._timeout.last_call

                expected_state = _RendezvousState()

                self._add_participants(num_participants, expected_state, ranked=True)

                expected_state.participants[self._node] = rank

                expected_state.last_heartbeats[self._node] = self._now

                expected_state.complete = True
                expected_state.deadline = None

                with self.subTest(num_participants=num_participants):
                    self._min_nodes = num_participants + 1 if min_max_nodes_equal else 0
                    self._max_nodes = num_participants + 1

                    self._assert_action(_Action.ADD_TO_PARTICIPANTS, expected_state)

                    self._mock_state_holder.reset_mock()

    def test_run_adds_to_waitlist(self) -> None:
        expected_state = _RendezvousState()

        expected_state.wait_list.add(self._node)

        expected_state.last_heartbeats[self._node] = self._now

        self._assert_action(_Action.ADD_TO_WAIT_LIST, expected_state)

    def test_run_removes_from_participants(self) -> None:
        for complete, last_call_deadline in [(False, self._now), (True, None)]:
            self._state = _RendezvousState()

            self._add_participants(2, self._state)

            self._state.participants[self._node] = 0

            self._state.last_heartbeats[self._node] = self._now

            self._state.complete = complete
            self._state.deadline = last_call_deadline

            self._state.round = 1

            expected_state = _RendezvousState()

            self._add_participants(2, expected_state)

            expected_state.complete = complete
            expected_state.deadline = last_call_deadline

            expected_state.round = 1

            with self.subTest(complete=complete):
                self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

                self._mock_state_holder.reset_mock()

    def test_run_removes_from_participants_and_moves_to_next_round_if_node_is_last_participant(
        self,
    ) -> None:
        self._state.participants[self._node] = 0

        self._state.last_heartbeats[self._node] = self._now

        self._state.complete = True

        self._state.round = 1

        expected_state = _RendezvousState()

        expected_state.complete = False

        expected_state.round = 2

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_participants_and_clears_last_call_if_rendezvous_has_less_than_min_nodes(
        self,
    ) -> None:
        self._add_participants(2, self._state)

        self._state.participants[self._node] = 0

        self._state.last_heartbeats[self._node] = self._now

        self._state.deadline = self._now

        expected_state = _RendezvousState()

        self._add_participants(2, expected_state)

        self._min_nodes = 3
        self._max_nodes = 4

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS, expected_state)

    def test_run_removes_from_waitlist(self) -> None:
        self._state.wait_list.add(self._node)

        self._state.last_heartbeats[self._node] = self._now

        expected_state = _RendezvousState()

        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST, expected_state)

    def test_run_marks_rendezvous_closed(self) -> None:
        expected_state = _RendezvousState()

        expected_state.closed = True

        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED, expected_state)

    def test_run_raises_error_if_rendezvous_is_closed(self) -> None:
        with self.assertRaises(RendezvousClosedError):
            self._run_action(_Action.ERROR_CLOSED)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    def test_run_raises_error_if_operation_timed_out(self) -> None:
        with self.assertRaises(RendezvousTimeoutError):
            self._run_action(_Action.ERROR_TIMEOUT)

        self.assertListEqual(self._mock_state_holder.mock_calls, [call.sync()])

    def test_run_delays_execution_if_sync_requested(self) -> None:
        with patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous._delay"
        ) as mock_delay:
            self._run_action(_Action.SYNC)

            mock_delay.assert_called_once_with(seconds=1)

        self.assertListEqual(
            self._mock_state_holder.mock_calls, [call.sync(), call.sync()]
        )


class AbstractTestRendezvousOp(ABC):
    assertEqual: Callable

    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)

        self._min_nodes = 1
        self._max_nodes = 2

        self._keep_alive_interval = timedelta(seconds=30)

        self._state = _RendezvousState()
        self._state.participants[_NodeDesc("dummy1", 1, 1)] = 1

        self._now = datetime(2000, 1, 1, hour=0, minute=0)

        self._deadline = 10

        self._datetime_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime"
        )

        mock_datetime = self._datetime_patch.start()
        mock_datetime.utcnow.return_value = self._now

        self._time_patch = patch(
            "torch.distributed.elastic.rendezvous.dynamic_rendezvous.time"
        )

        mock_time = self._time_patch.start()
        mock_time.monotonic.return_value = self._deadline

    def tearDown(self) -> None:
        self._time_patch.stop()
        self._datetime_patch.stop()

    def _get_next_action(self) -> _Action:
        op = self._create_op()

        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        ctx = _RendezvousContext(self._node, self._state, settings)

        return op(ctx, self._deadline)

    @abstractmethod
    def _create_op(self) -> Callable:
        pass

    def _assert_action(self, expected_action) -> None:
        action = self._get_next_action()

        self.assertEqual(action, expected_action)


class TestRendezvousExitOp(AbstractTestRendezvousOp, TestCase):
    def _create_op(self) -> Callable:
        return _RendezvousExitOp()

    def test_removes_from_participants_if_node_is_participant(self) -> None:
        self._state.participants[self._node] = 1

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        self._deadline = 0

        self._state.participants[self._node] = 1

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_finishes_if_node_is_not_participant(self) -> None:
        self._assert_action(_Action.FINISH)


class TestRendezvousJoinOp(AbstractTestRendezvousOp, TestCase):
    def _create_op(self) -> Callable:
        return _RendezvousJoinOp()

    def test_raises_closed_if_rendezvous_is_closed(self) -> None:
        self._state.closed = True

        self._assert_action(_Action.ERROR_CLOSED)

    def test_finishes_if_rendezvous_is_complete_and_node_is_participant(self) -> None:
        self._state.participants[self._node] = 0

        self._state.complete = True

        self._assert_action(_Action.FINISH)

    def _assert_waits_rendezvous_completion(self) -> None:
        keep_alive_time = self._now - self._keep_alive_interval

        for delta, expected_action in [
            (timedelta(seconds=0), _Action.KEEP_ALIVE),
            (timedelta(seconds=1), _Action.SYNC),
        ]:
            self._state.last_heartbeats[self._node] = keep_alive_time + delta

            self._assert_action(expected_action)

    def test_treat_as_redundancy_for_next_rendezvous_if_rendezvous_is_complete(
        self,
    ) -> None:
        self._max_nodes = 1

        self._state.complete = True

        self._assert_action(_Action.ADD_TO_REDUNDANCY_LIST)

    def test_waits_next_round_if_rendezvous_is_complete_and_node_is_redundant(
        self,
    ) -> None:
        self._state.redundancy_list.add(self._node)

        self._max_nodes = 1

        self._state.complete = True

        self._assert_waits_rendezvous_completion()

    def test_remove_from_rednundancy_list(self) -> None:
        self._state.redundancy_list.add(self._node)

        self._max_nodes = 2

        self._state.complete = True

        self._assert_action(_Action.REMOVE_FROM_REDUNDANCY_LIST)

    def test_waits_next_round_if_rendezvous_is_complete_and_node_is_in_wait_list(
        self,
    ) -> None:
        self._state.wait_list.add(self._node)

        self._state.complete = True

        self._assert_waits_rendezvous_completion()

    def test_adds_to_wait_list_if_rendezvous_is_complete_and_num_nodes_is_less_than_max_nodes(
        self,
    ) -> None:
        self._state.complete = True

        self._assert_action(_Action.ADD_TO_WAIT_LIST)

    def test_waits_rendezvous_to_complete_if_node_is_participant(self) -> None:
        self._max_nodes = 3

        self._state.participants[self._node] = 0

        self._state.deadline = self._now

        self._assert_waits_rendezvous_completion()

    def test_marks_rendezvous_complete_if_node_is_participant_and_last_call_deadline_exceeded(
        self,
    ) -> None:
        self._max_nodes = 3

        self._state.participants[self._node] = 0

        self._state.deadline = self._now - timedelta(seconds=1)

        self._assert_action(_Action.MARK_RENDEZVOUS_COMPLETE)

    def test_adds_to_participants(self) -> None:
        self._assert_action(_Action.ADD_TO_PARTICIPANTS)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        self._deadline = 0

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_raises_timeout_if_rollback_deadline_exceeded_and_node_is_participant(
        self,
    ) -> None:
        self._deadline = 0

        self._state.participants[self._node] = 0

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_raises_timeout_if_rollback_deadline_exceeded_and_node_is_in_wait_list(
        self,
    ) -> None:
        self._deadline = 0

        self._state.wait_list.add(self._node)

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_removes_from_participants_if_timed_out_but_rollback_deadline_is_not_reached(
        self,
    ) -> None:
        self._deadline = 5

        self._state.participants[self._node] = 0

        self._assert_action(_Action.REMOVE_FROM_PARTICIPANTS)

    def test_removes_from_wait_list_if_timed_out_but_rollback_deadline_is_not_reached(
        self,
    ) -> None:
        self._deadline = 5

        self._state.wait_list.add(self._node)

        self._assert_action(_Action.REMOVE_FROM_WAIT_LIST)

    def test_no_timeout_for_redundant_node(self) -> None:
        self._max_nodes = 1
        self._deadline = 0
        self._state.complete = True

        self._state.redundancy_list.add(self._node)

        self._assert_action(_Action.SYNC)

    def test_keep_alive_for_redundant_node(self) -> None:
        self._deadline = 0
        self._max_nodes = 1
        self._state.complete = True

        self._state.redundancy_list.add(self._node)

        keep_alive_time = self._now - self._keep_alive_interval
        self._state.last_heartbeats[self._node] = keep_alive_time
        self._assert_action(_Action.KEEP_ALIVE)


class TestRendezvousCloseOp(AbstractTestRendezvousOp, TestCase):
    def _create_op(self) -> Callable:
        return _RendezvousCloseOp()

    def test_finishes_if_rendezvous_is_closed(self) -> None:
        self._state.closed = True

        self._assert_action(_Action.FINISH)

    def test_raises_timeout_if_deadline_exceeded(self) -> None:
        self._deadline = 0

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_marks_rendezvous_closed(self) -> None:
        self._assert_action(_Action.MARK_RENDEZVOUS_CLOSED)


class TestRendezvousKeepAliveOp(AbstractTestRendezvousOp, TestCase):
    def _create_op(self) -> Callable:
        return _RendezvousKeepAliveOp()

    def test_updates_keep_alive_if_needed(self) -> None:
        keep_alive_time = self._now - self._keep_alive_interval

        for delta in [timedelta(seconds=0), timedelta(seconds=-1)]:
            with self.subTest(delta=delta):
                self._state.last_heartbeats[self._node] = keep_alive_time + delta

                self._assert_action(_Action.KEEP_ALIVE)

    def test_raises_timeout_if_deadlined_exceeded(self) -> None:
        self._deadline = 0

        self._state.last_heartbeats[self._node] = self._now - self._keep_alive_interval

        self._assert_action(_Action.ERROR_TIMEOUT)

    def test_finishes_if_no_keep_alive_update_is_needed(self) -> None:
        delta = timedelta(seconds=1)

        self._state.last_heartbeats[self._node] = (
            self._now - self._keep_alive_interval + delta
        )

        self._assert_action(_Action.FINISH)


class DummyStore(Store):
    pass


class DynamicRendezvousHandlerTest(TestCase):
    def setUp(self) -> None:
        self._node = _NodeDesc("this_node", 1, 1)

        self._min_nodes = 1
        self._max_nodes = 1

        self._join_timeout: Optional[timedelta] = None
        self._close_timeout: Optional[timedelta] = None
        self._heartbeat_timeout: Optional[timedelta] = None

        self._keep_alive_interval = timedelta(seconds=30)

        self._store = DummyStore()

        self._mock_store_get = MagicMock(return_value=b"dummy_value")

        setattr(self._store, "get", self._mock_store_get)  # noqa: B010

        self._state_holder = FakeRendezvousStateHolder()

        self._mock_sync = MagicMock(wraps=self._state_holder.sync)

        setattr(self._state_holder, "sync", self._mock_sync)  # noqa: B010

        self._state = self._state_holder.state

    def _create_handler(self) -> DynamicRendezvousHandler:
        settings = RendezvousSettings(
            run_id="dummy_run_id",
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=RendezvousTimeout(
                join=self._join_timeout,
                close=self._close_timeout,
                heartbeat=self._heartbeat_timeout,
            ),
            keep_alive_interval=self._keep_alive_interval,
            keep_alive_max_attempt=3,
        )

        self._state_holder.state = self._state

        return DynamicRendezvousHandler(
            self._node, settings, "dummy_backend", self._store, self._state_holder
        )

    @patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous._delay")
    def test_next_rendezvous_skews_the_first_join_attempt(self, mock_delay) -> None:
        for round, expected_call_count in [(0, True), (1, False)]:
            with self.subTest(round=round):
                self._state.round = round

                handler = self._create_handler()

                handler.next_rendezvous()

                self.assertEqual(mock_delay.call_count, expected_call_count)

                mock_delay.reset_mock()

    def test_next_rendezvous_returns_expected_value(self) -> None:
        self._state.participants[_NodeDesc("dummy1", 1, 1)] = 0
        self._state.participants[_NodeDesc("dummy2", 1, 1)] = 0

        self._max_nodes = 3

        handler = self._create_handler()

        store, rank, world_size = handler.next_rendezvous()

        self.assertEqual(rank, 2)
        self.assertEqual(world_size, 3)

        _ = store.get("dummy_key")

        self._mock_store_get.assert_called_once_with(
            "torch.rendezvous.dummy_run_id.0/dummy_key"
        )

    def test_next_rendezvous_respects_the_requested_timeout(self) -> None:
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        self._join_timeout = timedelta(seconds=0.2)

        handler = self._create_handler()

        with self.assertRaises(RendezvousTimeoutError):
            handler.next_rendezvous()

    def test_next_rendezvous_moves_to_next_round_if_called_repeatedly(self) -> None:
        handler = self._create_handler()

        for i in range(4):
            handler.next_rendezvous()

            self.assertEqual(self._state.round, i)

    def test_is_closed_returns_expected_value(self) -> None:
        for closed in [False, True]:
            with self.subTest(closed=closed):
                self._state.closed = closed

                handler = self._create_handler()

                self.assertEqual(handler.is_closed(), closed)

                self._mock_sync.assert_called_once()

                self._mock_sync.reset_mock()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_is_closed_records_and_raises_exceptions(self, record_mock) -> None:
        self._mock_sync.side_effect = RendezvousError("test error")
        handler = self._create_handler()
        with self.assertRaises(RendezvousError):
            handler.is_closed()
            record_mock.assert_called_once()

    def test_set_closed_closes_rendezvous(self) -> None:
        handler = self._create_handler()

        handler.set_closed()

        self.assertTrue(self._state.closed)

    def test_set_closed_respects_the_requested_timeout(self) -> None:
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        self._close_timeout = timedelta(seconds=0.2)

        handler = self._create_handler()

        with self.assertRaises(RendezvousTimeoutError):
            handler.set_closed()

    def test_set_closed_can_be_called_multiple_times(self) -> None:
        handler = self._create_handler()

        handler.set_closed()
        handler.set_closed()

        self.assertTrue(self._state.closed)

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_set_closed_records_and_raises_exceptions(self, record_mock) -> None:
        with patch.object(DynamicRendezvousHandler, "_close") as close_mock:
            close_mock.side_effect = RendezvousError("test error")
            handler = self._create_handler()
            with self.assertRaises(RendezvousError):
                handler.set_closed()
                record_mock.assert_called_once()

    def test_num_nodes_waiting_returns_expected_value(self) -> None:
        self._state.wait_list.add(_NodeDesc("dummy1", 1, 1))
        self._state.wait_list.add(_NodeDesc("dummy2", 1, 1))

        handler = self._create_handler()

        self.assertEqual(handler.num_nodes_waiting(), 2)

        self._mock_sync.assert_called_once()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_num_nodes_waiting_records_and_raises_exceptions(self, record_mock) -> None:
        self._mock_sync.side_effect = RendezvousError("test error")
        handler = self._create_handler()
        with self.assertRaises(RendezvousError):
            handler.num_nodes_waiting()
            record_mock.assert_called_once()

    def test_shutdown_closes_rendezvous_and_returns_true(self) -> None:
        handler = self._create_handler()

        result = handler.shutdown()

        self.assertTrue(result)

        self.assertTrue(self._state.closed)

    def test_shutdown_returns_false_if_rendezvous_cannot_be_closed(self) -> None:
        self._mock_sync.side_effect = [RendezvousError]

        handler = self._create_handler()

        result = handler.shutdown()

        self.assertFalse(result)

    def test_shutdown_can_be_called_multiple_times(self) -> None:
        handler = self._create_handler()

        handler.shutdown()
        handler.shutdown()

        self.assertTrue(self._state.closed)

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_shutdown_records_and_raises_exceptions(self, record_mock) -> None:
        with patch.object(DynamicRendezvousHandler, "_close") as close_mock:
            close_mock.side_effect = RuntimeError("test error")
            handler = self._create_handler()
            with self.assertRaises(RuntimeError):
                handler.shutdown()
                record_mock.assert_called_once()

    @patch("torch.distributed.elastic.rendezvous.dynamic_rendezvous.datetime")
    def test_keep_alive_updates_last_heartbeat(self, mock_datetime) -> None:
        now = datetime(2000, 1, 1, hour=0, minute=0)

        mock_datetime.utcnow.return_value = now

        self._state.last_heartbeats[self._node] = now - (self._keep_alive_interval * 2)

        handler = self._create_handler()

        handler._keep_alive()

        self.assertEqual(self._state.last_heartbeats[self._node], now)

    def _assert_keep_alive_swallows_rendezvous_errors(self) -> None:
        last_heartbeat_time = datetime.utcnow() - (self._keep_alive_interval * 2)

        self._state.last_heartbeats[self._node] = last_heartbeat_time

        handler = self._create_handler()

        handler._keep_alive()

        self.assertEqual(self._state.last_heartbeats[self._node], last_heartbeat_time)

    def test_keep_alive_swallows_rendezvous_errors(self) -> None:
        self._mock_sync.side_effect = [RendezvousError]

        self._assert_keep_alive_swallows_rendezvous_errors()

    def test_keep_alive_respects_the_requested_timeout(self) -> None:
        self._mock_sync.side_effect = lambda: time.sleep(0.3)

        self._heartbeat_timeout = timedelta(seconds=0.2)

        self._assert_keep_alive_swallows_rendezvous_errors()

    def test_keep_alive_thread_is_started_with_next_rendezvous_and_stopped_with_shutdown(
        self,
    ) -> None:
        self._node = _NodeDesc("this_node", 1, 2)

        name = "RendezvousKeepAliveTimer_2"

        handler = self._create_handler()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

        handler.next_rendezvous()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        handler.shutdown()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    def test_keep_alive_thread_is_started_with_next_rendezvous_and_stopped_with_finalizer(
        self,
    ) -> None:
        self._node = _NodeDesc("this_node", 1, 3)

        name = "RendezvousKeepAliveTimer_3"

        handler = self._create_handler()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

        handler.next_rendezvous()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        del handler

        self.assertTrue(all(t.name != name for t in threading.enumerate()))


class DummyRendezvousBackend(RendezvousBackend):
    @property
    def name(self):
        return "dummy_backend"

    def get_state(self):
        return None

    def set_state(self, state, token):
        return None


class DynamicRendezvousHandlerFromBackendTest(TestCase):
    def setUp(self) -> None:
        self._run_id = "dummy_run_id"
        self._store = DummyStore()
        self._backend = DummyRendezvousBackend()
        self._min_nodes = 3
        self._max_nodes = 6
        self._timeout: Optional[RendezvousTimeout] = RendezvousTimeout()

    def _create_handler(self) -> DynamicRendezvousHandler:
        return DynamicRendezvousHandler.from_backend(
            run_id=self._run_id,
            store=self._store,
            backend=self._backend,
            min_nodes=self._min_nodes,
            max_nodes=self._max_nodes,
            timeout=self._timeout,
        )

    def test_init_initializes_handler(self) -> None:
        handler = self._create_handler()

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
            join_timeout="50",
            last_call_timeout="60",
            close_timeout="70",
        )

        self._expected_timeout = RendezvousTimeout(
            timedelta(seconds=50), timedelta(seconds=60), timedelta(seconds=70)
        )

    def test_create_handler_returns_handler(self) -> None:
        handler = create_handler(self._store, self._backend, self._params)

        self.assertEqual(handler.get_backend(), self._backend.name)
        self.assertEqual(handler.get_run_id(), self._params.run_id)
        self.assertEqual(handler.settings.min_nodes, self._params.min_nodes)
        self.assertEqual(handler.settings.max_nodes, self._params.max_nodes)
        self.assertEqual(handler.settings.timeout.join, self._expected_timeout.join)
        self.assertEqual(
            handler.settings.timeout.last_call, self._expected_timeout.last_call
        )
        self.assertEqual(handler.settings.timeout.close, self._expected_timeout.close)

    def test_create_handler_returns_handler_if_timeout_is_not_specified(self) -> None:
        del self._params.config["join_timeout"]
        del self._params.config["last_call_timeout"]
        del self._params.config["close_timeout"]

        self._expected_timeout = RendezvousTimeout()

        self.test_create_handler_returns_handler()

    @patch("torch.distributed.elastic.events.record_rdzv_event")
    def test_create_handler_records_and_raises_exceptions(self, record_mock) -> None:
        with patch.object(DynamicRendezvousHandler, "from_backend") as from_mock:
            from_mock.side_effect = RendezvousError("test error")
            with self.assertRaises(RendezvousError):
                create_handler(self._store, self._backend, self._params)
                record_mock.assert_called_once()


def _ignore_exception(exception_type: Exception, fn: Callable):
    try:
        fn()
    except exception_type as e:
        pass


def _wait_for(condition, timeout=10, interval=1, name=None):
    def _wait_while():
        while True:
            if condition():
                break
            else:
                time.sleep(interval)

    wait_thread = threading.Thread(target=_wait_while, name=name)
    wait_thread.start()
    wait_thread.join(timeout=timeout)


class _CapturingThread(threading.Thread):
    def __init__(self, target=None, name=None, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        threading.Thread.__init__(
            self, target=target, args=args, kwargs=kwargs, name=name
        )
        self._result = None

    def run(self):
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._result


class IntegrationTest(TestCase):
    def setUp(self) -> None:
        self._store = DummyStore()
        self._handlers = []
        self._backend = _InMemoryRendezvousBackend()

    def tearDown(self) -> None:
        for handler in self._handlers:
            handler._stop_heartbeats()

    def _create_handler(self, **kwargs) -> DynamicRendezvousHandler:
        params = {
            "backend": self._backend.name,
            "endpoint": "dummy_endpoint",
            "run_id": "dummy_run_id",
            "min_nodes": 2,
            "max_nodes": 2,
            "join_timeout": "5",
            "local_addr": f"address_{len(self._handlers)}",
        }
        params.update(**kwargs)

        rzdv_params = RendezvousParameters(**params)

        handler = create_handler(self._store, self._backend, rzdv_params)
        self._handlers.append(handler)
        return handler

    def test_all_nodes_join_rendezvous(self) -> None:
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler1_thread.start()
        handler2_thread.start()

        store1, rank1, world_size1 = handler1_thread.join()
        store2, rank2, world_size2 = handler2_thread.join()
        self.assertEqual(store1.underlying_store, self._store)
        self.assertEqual(store2.underlying_store, self._store)

        self.assertNotEqual(rank1, rank2)

        self.assertEqual(world_size1, 2)
        self.assertEqual(world_size2, 2)

    def test_redundancy_list(self) -> None:
        handler1 = self._create_handler(min_nodes=2, max_nodes=2)
        handler2 = self._create_handler(min_nodes=2, max_nodes=2)
        handler3 = self._create_handler(min_nodes=2, max_nodes=2)

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)
        handler3_thread = _CapturingThread(
            target=_ignore_exception,
            args=(RendezvousTimeoutError, lambda: handler3.next_rendezvous()),
        )

        handler1_thread.start()
        handler2_thread.start()

        # establish successful rendezvous
        handler1_thread.join()
        handler2_thread.join()

        # expect to register in redundancy list
        handler3_thread.start()

        # wait until the handler3 is registered in the redundancy list
        _wait_for(lambda: pickle.loads(self._backend.get_state()[0]).redundancy_list)

        state_and_token = self._backend.get_state()
        state = pickle.loads(state_and_token[0])
        addresses = [node.addr for node in state.redundancy_list]
        self.assertListEqual(addresses, ["address_2"])

    def test_redundancy_transition_to_wait_list_then_join_rendezvous(self) -> None:
        handler1 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )
        handler2 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
            keep_alive_interval=timedelta(seconds=1),
        )
        handler3 = self._create_handler(
            min_nodes=1,
            max_nodes=2,
        )

        handler1_thread = _CapturingThread(target=handler1.next_rendezvous)
        handler2_thread = _CapturingThread(target=handler2.next_rendezvous)

        handler3_thread = _CapturingThread(
            target=_ignore_exception,
            args=(RendezvousTimeoutError, lambda: handler3.next_rendezvous()),
        )

        handler1_thread.start()
        handler2_thread.start()

        # establish successful rendezvous
        handler1_thread.join()
        handler2_thread.join()

        handler3_thread.start()

        _wait_for(lambda: pickle.loads(self._backend.get_state()[0]).redundancy_list)

        handler2._stop_heartbeats()

        _wait_for(
            lambda: len(pickle.loads(self._backend.get_state()[0]).participants) == 1
        )
        _wait_for(
            lambda: len(pickle.loads(self._backend.get_state()[0]).wait_list) == 1
        )


class _InMemoryRendezvousBackend(RendezvousBackend):
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None
        self._token = None

    @property
    def name(self):
        return "_in_memory_backend"

    def get_state(self):
        with self._lock:
            if self._state is None:
                return None
            return (self._state, self._token)

        return self._state

    def set_state(self, state, token):
        if state is None:
            raise ValueError("State cannot be None.")
        with self._lock:
            if token is None and self._token is not None:
                return None
            if self._token != token:
                return None

            self._state = state
            self._token = self._token + 1 if self._token is not None else 0
