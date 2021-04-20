# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Tuple

from torch.distributed import Store

from .api import RendezvousHandler, RendezvousParameters, RendezvousStateError


Token = Any
"""Represents an opaque fencing token used by the rendezvous backend."""


class RendezvousBackend(ABC):
    """Represents a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of the backend."""

    @abstractmethod
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """Gets the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

    @abstractmethod
    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token]]:
        """Sets the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified ``token`` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified ``token`` does not match the fencing token stored
            in the backend, the state won't be updated; instead the existing
            state along with its fencing token will be returned to the caller.
          - If the specified ``token`` is ``None``, the new state will be set
            only if there is no existing state in the backend. Either the new
            state or the existing state along with its fencing token will be
            returned to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to :py:meth:`get_state` or ``set_state()``.

        Returns:
            A tuple of the serialized rendezvous state and its fencing token.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """


class RendezvousTimeout:
    """Holds the timeout configuration of a rendezvous.

    Args:
        join:
            The total time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            minimum number of nodes has been reached.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
    """

    _ZERO = timedelta(0)

    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close": timedelta(seconds=30),
    }

    _join: timedelta
    _last_call: timedelta
    _close: timedelta

    def __init__(
        self,
        join: Optional[timedelta] = None,
        last_call: Optional[timedelta] = None,
        close: Optional[timedelta] = None,
    ) -> None:
        self._set_timeouts(join=join, last_call=last_call, close=close)

    @property
    def join(self) -> timedelta:
        """Gets the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Gets the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Gets the close timeout."""
        return self._close

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)


@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Holds the settings of the rendezvous."""

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int


@dataclass(eq=True, frozen=True)
class _NodeDesc:
    """Describes a node in the rendezvous.

    Attributes:
        fqdn:
            The FQDN of the node.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    fqdn: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.fqdn}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    """Generates node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an
    auto-incremented integer that uniquely identifies a node in the rendezvous.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # An integer that is incremented with each call to generate().
        self._local_id = 0

    def generate(self) -> _NodeDesc:
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id

            self._local_id += 1

        return _NodeDesc(socket.getfqdn(), os.getpid(), local_id)


class _RendezvousState:
    """Holds the state of a rendezvous.

    A rendezvous is synced across the nodes via a ``RendezvousBackend``.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The date and time at which the current round of the rendezvous will
            be considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        last_keep_alives:
            A dictionary containing each node's last keep-alive time.
    """

    round: int
    complete: bool
    deadline: Optional[datetime]
    closed: bool
    participants: Dict[_NodeDesc, int]
    wait_list: Set[_NodeDesc]
    last_keep_alives: Dict[_NodeDesc, datetime]

    def __init__(self) -> None:
        self.round = 0
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.last_keep_alives = {}


class _RendezvousStateHolder:
    """Holds the rendezvous state synced with other nodes."""

    backend: RendezvousBackend
    settings: RendezvousSettings
    cache_duration: int
    state: _RendezvousState
    _token: Token
    _dirty: bool
    _last_sync_time: float

    def __init__(
        self, backend: RendezvousBackend, settings: RendezvousSettings, cache_duration: int = 0
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.cache_duration = cache_duration
        self.state = _RendezvousState()
        self._token = None
        self._dirty = False
        self._last_sync_time = 0.0

    def sync(self) -> None:
        if self._dirty:
            state_bits = pickle.dumps(self.state)

            response = self.backend.set_state(state_bits, self._token)
        else:
            if self.cache_duration > 0:
                # Avoid overloading the backend if we are asked to retrieve the
                # state repeatedly. Try to serve the cached state.
                if self._last_sync_time > max(time.monotonic() - self.cache_duration, 0):
                    return

            response = self.backend.get_state()

        if response:
            state_bits, token = response

            try:
                self.state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
        else:
            token = None

            self.state = _RendezvousState()

        self._token = token
        self._dirty = False

        self._last_sync_time = time.monotonic()

        self._sanitize()

    def _sanitize(self) -> None:
        expire_time = datetime.utcnow() - (
            self.settings.keep_alive_interval * self.settings.keep_alive_max_attempt
        )

        # Filter out the dead nodes.
        dead_nodes = [
            node
            for node, last_keep_alive in self.state.last_keep_alives.items()
            if last_keep_alive < expire_time
        ]

        for dead_node in dead_nodes:
            del self.state.last_keep_alives[dead_node]

            try:
                del self.state.participants[dead_node]
            except KeyError:
                pass

            try:
                self.state.wait_list.remove(dead_node)
            except KeyError:
                pass

    def mark_dirty(self) -> None:
        self._dirty = True


class DynamicRendezvousHandler(RendezvousHandler):
    """Represents the dynamic rendezvous handler.

    Args:
        run_id:
            The run id of the rendezvous.
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend to use to hold the rendezvous state.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
    """

    _run_id: str
    _settings: RendezvousSettings
    _store: Store
    _backend: RendezvousBackend

    def __init__(
        self,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        timeout: Optional[RendezvousTimeout] = None,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of nodes ({min_nodes}) must be greater than zero."
            )

        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of nodes ({max_nodes}) must be greater than or equal to the "
                f"minimum number of nodes ({min_nodes})."
            )

        self._settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=5),
            keep_alive_max_attempt=3,
        )

        self._store = store
        self._backend = backend

    @property
    def settings(self) -> RendezvousSettings:
        """Gets the settings of the rendezvous."""
        return self._settings

    @property
    def store(self) -> Store:
        """Gets the C10d store returned as part of the rendezvous."""
        return self._store

    @property
    def backend(self) -> RendezvousBackend:
        """Gets the backend used to hold the rendezvous state."""
        return self._backend

    def get_backend(self) -> str:
        """See base class."""
        return self._backend.name

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        """See base class."""
        raise NotImplementedError()

    def is_closed(self) -> bool:
        """See base class."""
        raise NotImplementedError()

    def set_closed(self) -> None:
        """See base class."""
        raise NotImplementedError()

    def num_nodes_waiting(self) -> int:
        """See base class."""
        raise NotImplementedError()

    def get_run_id(self) -> str:
        """See base class."""
        return self.settings.run_id

    def shutdown(self) -> bool:
        """See base class."""
        raise NotImplementedError()


def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def create_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
    """Create a new :py:class:`DynamicRendezvousHandler` from the specified
    parameters.

    +-------------------+------------------------------------------------------+
    | Parameter         | Description                                          |
    +===================+======================================================+
    | join_timeout      | The total time, in seconds, within which the         |
    |                   | rendezvous is expected to complete. Defaults to 600  |
    |                   | seconds.                                             |
    +-------------------+------------------------------------------------------+
    | last_call_timeout | An additional wait amount, in seconds, before        |
    |                   | completing the rendezvous once the minimum number of |
    |                   | nodes has been reached. Defaults to 30 seconds.      |
    +-------------------+------------------------------------------------------+
    | close_timeout     | The time, in seconds, within which the rendezvous is |
    |                   | expected to close after a call to                    |
    |                   | :py:meth:`RendezvousHandler.set_closed` or           |
    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                   | 30 seconds.                                          |
    +-------------------+------------------------------------------------------+
    """
    timeout = RendezvousTimeout(
        _get_timeout(params, "join"),
        _get_timeout(params, "last_call"),
        _get_timeout(params, "close"),
    )

    return DynamicRendezvousHandler(
        params.run_id,
        store,
        backend,
        params.min_nodes,
        params.max_nodes,
        timeout,
    )
