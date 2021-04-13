# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Optional, Tuple

from torch.distributed import Store

from .api import RendezvousHandler, RendezvousParameters

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
    _store: Store
    _backend: RendezvousBackend
    _min_nodes: int
    _max_nodes: int
    _timeout: RendezvousTimeout

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

        self._run_id = run_id

        self._store = store
        self._backend = backend

        self._min_nodes = min_nodes
        self._max_nodes = max_nodes

        self._timeout = timeout or RendezvousTimeout()

    @property
    def run_id(self) -> str:
        """Gets the run id of the rendezvous."""
        return self._run_id

    @property
    def store(self) -> Store:
        """Gets the C10d store returned as part of the rendezvous."""
        return self._store

    @property
    def backend(self) -> RendezvousBackend:
        """Gets the backend used to hold the rendezvous state."""
        return self._backend

    @property
    def min_nodes(self) -> int:
        """Gets the minimum number of nodes to admit to the rendezvous."""
        return self._min_nodes

    @property
    def max_nodes(self) -> int:
        """Gets the maximum number of nodes to admit to the rendezvous."""
        return self._max_nodes

    @property
    def timeout(self) -> RendezvousTimeout:
        """Gets the timeout configuration of the rendezvous."""
        return self._timeout

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
        return self._run_id

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
