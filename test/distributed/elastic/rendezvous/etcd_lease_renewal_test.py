# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
from unittest import mock

from torch.distributed.elastic.rendezvous.etcd_rendezvous import (
    _lease_renewal_worker,
    EtcdRendezvous,
    etcd,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class _TransientEtcdError(etcd.EtcdException):
    # Bypasses the _etcd_stub EtcdException.__init__, which raises when the real
    # python-etcd package is not installed, so this test runs with or without it.
    def __init__(self) -> None:
        Exception.__init__(self, "transient etcd failure")


class _KeyNotFoundError(etcd.EtcdKeyNotFound):
    def __init__(self) -> None:
        Exception.__init__(self, "key gone")


class _ScriptedStopEvent:
    # Stands in for threading.Event so the renewal loop runs synchronously and
    # deterministically in the test thread: wait() returns the scripted booleans
    # instead of sleeping for ttl/2.
    def __init__(self, returns):
        self._returns = iter(returns)

    def wait(self, timeout=None):
        return next(self._returns)


class _RecordingClient:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.refresh_calls = []

    def refresh(self, path, ttl):
        effect = self._side_effects[len(self.refresh_calls)]
        self.refresh_calls.append((path, ttl))
        if effect is not None:
            raise effect


class EtcdLeaseRenewalWorkerTest(TestCase):
    def test_transient_etcd_error_keeps_renewing(self):
        # First refresh raises a transient etcd error, the second succeeds.
        client = _RecordingClient([_TransientEtcdError(), None])
        stop_event = _ScriptedStopEvent([False, True])

        _lease_renewal_worker(client, "/rdzv/lease", 20, stop_event)

        # Before the fix the transient EtcdException escaped the loop and killed
        # the daemon thread after the first call, so the lease was never renewed
        # again. The worker must survive it and issue the second refresh.
        self.assertEqual(len(client.refresh_calls), 2)

    def test_unexpected_exception_still_propagates(self):
        # The transient-error handling must stay scoped to etcd.EtcdException;
        # an unrelated error must not be swallowed.
        client = _RecordingClient([RuntimeError("boom")])
        stop_event = _ScriptedStopEvent([True])

        with self.assertRaises(RuntimeError):
            _lease_renewal_worker(client, "/rdzv/lease", 20, stop_event)

    def test_key_not_found_stops(self):
        # EtcdKeyNotFound means the run is over: stop renewing.
        client = _RecordingClient([_KeyNotFoundError()])
        stop_event = _ScriptedStopEvent([])

        _lease_renewal_worker(client, "/rdzv/lease", 20, stop_event)

        self.assertEqual(len(client.refresh_calls), 1)

    def test_connection_refused_stops(self):
        # Happens when the server is already gone during teardown: stop renewing.
        client = _RecordingClient([ConnectionRefusedError()])
        stop_event = _ScriptedStopEvent([])

        _lease_renewal_worker(client, "/rdzv/lease", 20, stop_event)

        self.assertEqual(len(client.refresh_calls), 1)

    def test_setup_lease_renewal_runs_worker(self):
        # setup_lease_renewal wires the client into the worker on a daemon thread.
        # Run the worker synchronously (refresh raises EtcdKeyNotFound, so it breaks
        # on the first call) to exercise the wiring without a real thread or sleep.
        client = _RecordingClient([_KeyNotFoundError()])
        rdzv = mock.Mock()
        rdzv.client = client

        class _SyncThread:
            def __init__(self, target, args=(), **kwargs):
                self._target = target
                self._args = args
                self.daemon = False

            def start(self):
                self._target(*self._args)

        with mock.patch.object(threading, "Thread", _SyncThread):
            stop_event = EtcdRendezvous.setup_lease_renewal(rdzv, "/rdzv/lease", 20)

        self.assertIsInstance(stop_event, threading.Event)
        self.assertEqual(len(client.refresh_calls), 1)


if __name__ == "__main__":
    run_tests()
