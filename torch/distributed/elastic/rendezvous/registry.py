# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import RendezvousHandler, RendezvousParameters
from .api import rendezvous_handler_registry as handler_registry
from .dynamic_rendezvous import create_handler


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend

    backend = create_backend(params)

    return create_handler(backend.store, backend, params)


def _create_expr_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .etcd_rendezvous_backend import create_backend
    from .etcd_store import EtcdStore

    backend = create_backend(params)

    store = EtcdStore(backend.client, "/torch/elastic/store")

    return create_handler(store, backend, params)


def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("c10d-experimental", _create_c10d_handler)
    handler_registry.register("etcd-experimental", _create_expr_etcd_handler)


# The legacy function kept for backwards compatibility.
def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    return handler_registry.create_handler(params)
