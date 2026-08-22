# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from datetime import timedelta
from typing import Any, cast

from torch.distributed import FileStore, Store, TCPStore
from torch.distributed.elastic.events import construct_and_record_rdzv_event, NodeState

from .api import (
    RendezvousConnectionError,
    RendezvousError,
    RendezvousParameters,
    RendezvousStateError,
)
from .dynamic_rendezvous import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint


logger = logging.getLogger(__name__)

# default port for the TCP store
DEFAULT_PORT = 29400


class C10dRendezvousBackend(RendezvousBackend):
    """Represents a C10d-backed rendezvous backend.

    Args:
        store:
            The :py:class:`torch.distributed.Store` instance to use to
            communicate with the C10d store.
        run_id:
            The run id of the rendezvous.
    """

    # See the explanation in the __init__ method.
    _NULL_SENTINEL = "Y2FuaW1hZGFt"

    _store: Store
    _key: str

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._store = store

        self._key = "torch.rendezvous." + run_id

        self._call_store("compare_set", self._key, "", self._NULL_SENTINEL)

    @property
    def name(self) -> str:
        """See base class."""
        return "c10d"

    def get_state(self) -> tuple[bytes, Token] | None:
        """See base class."""
        base64_state: bytes = self._call_store("get", self._key)

        return self._decode_state(base64_state)

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        """See base class."""
        base64_state_str: str = b64encode(state).decode()

        if token:
            if not isinstance(token, bytes):
                result = self.get_state()
                if result is not None:
                    return *result, False
                return None

            token = token.decode()
        else:
            token = self._NULL_SENTINEL

        base64_state: bytes = self._call_store(
            "compare_set", self._key, token, base64_state_str
        )

        state_token_pair = self._decode_state(base64_state)
        if state_token_pair is None:
            return None

        new_state, new_token = state_token_pair

        return new_state, new_token, new_state == state

    def _call_store(self, store_op: str, *args, **kwargs) -> Any:
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to the C10d store has failed. See inner exception for details."
            ) from exc

    def _decode_state(self, base64_state: bytes) -> tuple[bytes, Token] | None:
        if base64_state == self._NULL_SENTINEL.encode():
            return None

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        return state, base64_state


def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)

    cfg_is_host = params.get_as_bool("is_host")
    if cfg_is_host is not None:
        is_host = cfg_is_host
    else:
        is_host = _matches_machine_hostname(host)

    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    for is_server in [is_host, False]:
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
            )

            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)

            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc

    return store  # type: ignore[possibly-undefined]


def _create_file_store(params: RendezvousParameters) -> FileStore:
    # If a user specifies an endpoint, we treat it as a path to a file.
    if params.endpoint:
        path = params.endpoint
    else:
        try:
            # mkstemp returns (fd, path). The fd must be closed immediately
            # because FileStore opens the path independently. Leaving the fd
            # open leaks a file descriptor on every rendezvous creation and
            # can eventually exhaust the process limit (resolves #191394).
            fd, path = tempfile.mkstemp()
            try:
                os.close(fd)
            except OSError:
                pass  # best-effort; path is still valid
        except OSError as exc:
            raise RendezvousError(
                "The file creation for C10d store has failed. See inner exception for details."
            ) from exc

    try:
        store = FileStore(path)
    except (ValueError, RuntimeError) as exc:
        raise RendezvousConnectionError(
            "The connection to the C10d store has failed. See inner exception for details."
        ) from exc

    return store


def create_backend(params: RendezvousParameters) -> tuple[C10dRendezvousBackend, Store]:
    """Create a new :py:class:`C10dRendezvousBackend` from the specified parameters."""
    store_type = params.get("store_type", "tcp").strip().lower()
    store: Store

    try:
        if store_type == "file":
            store = _create_file_store(params)
        elif store_type == "tcp":
            store = _create_tcp_store(params)
        else:
            raise ValueError(
                "Invalid store type given. Currently only supports file and tcp."
            )

        backend = C10dRendezvousBackend(store, params.run_id)

    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise

    return backend, store
