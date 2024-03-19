# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
from base64 import b64decode, b64encode
from typing import Optional, Tuple, cast

import urllib3.exceptions  # type: ignore[import]
from etcd import Client as EtcdClient  # type: ignore[import]
from etcd import (
    EtcdAlreadyExist,
    EtcdCompareFailed,
    EtcdException,
    EtcdKeyNotFound,
    EtcdResult,
)
from torch.distributed import Store

from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint


class EtcdRendezvousBackend(RendezvousBackend):
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``etcd.Client`` instance to use to communicate with etcd.
        run_id:
            The run id of the rendezvous.
        key_prefix:
            The path under which to store the rendezvous state in etcd.
        ttl:
            The TTL of the rendezvous state. If not specified, defaults to two hours.
    """

    _DEFAULT_TTL = 7200  # 2 hours

    _client: EtcdClient
    _key: str
    _ttl: int

    def __init__(
        self,
        client: EtcdClient,
        run_id: str,
        key_prefix: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._client = client

        if key_prefix:
            self._key = key_prefix + "/" + run_id
        else:
            self._key = run_id

        if ttl and ttl > 0:
            self._ttl = ttl
        else:
            self._ttl = self._DEFAULT_TTL

    @property
    def name(self) -> str:
        """See base class."""
        return "etcd-v2"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        try:
            result = self._client.read(self._key)
        except EtcdKeyNotFound:
            return None
        except (EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        return self._decode_state(result)

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """See base class."""
        base64_state = b64encode(state).decode()

        kwargs = {}

        def get_state():
            result = self.get_state()
            if result is not None:
                tmp = *result, False
                # Python 3.6 does not support tuple unpacking in return
                # statements.
                return tmp
            return None

        if token:
            try:
                token = int(token)
            except ValueError:
                return get_state()

        if token:
            kwargs["prevIndex"] = token
        else:
            kwargs["prevExist"] = False

        try:
            result = self._client.write(self._key, base64_state, self._ttl, **kwargs)
        except (EtcdAlreadyExist, EtcdCompareFailed):
            result = None
        except (EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        if result is None:
            return get_state()

        tmp = *self._decode_state(result), True
        return tmp

    def _decode_state(self, result: EtcdResult) -> Tuple[bytes, Token]:
        base64_state = result.value.encode()

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        return state, result.modifiedIndex


def _create_etcd_client(params: RendezvousParameters) -> EtcdClient:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # The communication protocol
    protocol = params.get("protocol", "http").strip().lower()
    if protocol != "http" and protocol != "https":
        raise ValueError("The protocol must be HTTP or HTTPS.")

    # The SSL client certificate
    ssl_cert = params.get("ssl_cert")
    if ssl_cert:
        ssl_cert_key = params.get("ssl_cert_key")
        if ssl_cert_key:
            # The etcd client expects the certificate key as the second element
            # of the `cert` tuple.
            ssl_cert = (ssl_cert, ssl_cert_key)

    # The root certificate
    ca_cert = params.get("ca_cert")

    try:
        return EtcdClient(
            host,
            port,
            read_timeout=read_timeout,
            protocol=protocol,
            cert=ssl_cert,
            ca_cert=ca_cert,
            allow_reconnect=True,
        )
    except (EtcdException, urllib3.exceptions.TimeoutError) as exc:
        raise RendezvousConnectionError(
            "The connection to etcd has failed. See inner exception for details."
        ) from exc


def create_backend(params: RendezvousParameters) -> Tuple[EtcdRendezvousBackend, Store]:
    """Create a new :py:class:`EtcdRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | read_timeout | The read timeout, in seconds, for etcd operations.        |
    |              | Defaults to 60 seconds.                                   |
    +--------------+-----------------------------------------------------------+
    | protocol     | The protocol to use to communicate with etcd. Valid       |
    |              | values are "http" and "https". Defaults to "http".        |
    +--------------+-----------------------------------------------------------+
    | ssl_cert     | The path to the SSL client certificate to use along with  |
    |              | HTTPS. Defaults to ``None``.                              |
    +--------------+-----------------------------------------------------------+
    | ssl_cert_key | The path to the private key of the SSL client certificate |
    |              | to use along with HTTPS. Defaults to ``None``.            |
    +--------------+-----------------------------------------------------------+
    | ca_cert      | The path to the rool SSL authority certificate. Defaults  |
    |              | to ``None``.                                              |
    +--------------+-----------------------------------------------------------+
    """
    client = _create_etcd_client(params)

    backend = EtcdRendezvousBackend(client, params.run_id, key_prefix="/torch/elastic/rendezvous")

    store = EtcdStore(client, "/torch/elastic/store")

    return backend, store
