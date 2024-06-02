# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import RendezvousHandler, RendezvousParameters
from .api import rendezvous_handler_registry as handler_registry
from .dynamic_rendezvous import create_handler

__all__ = ['get_rendezvous_handler']

def _create_static_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import static_tcp_rendezvous

    return static_tcp_rendezvous.create_rdzv_handler(params)


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _create_etcd_v2_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .etcd_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _create_c10d_handler(params: RendezvousParameters) -> RendezvousHandler:
    from .c10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)

    return create_handler(store, backend, params)


def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("etcd-v2", _create_etcd_v2_handler)
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    """
    Obtain a reference to a :py:class`RendezvousHandler`.

    Custom rendezvous handlers can be registered by

    ::

      from torch.distributed.elastic.rendezvous import rendezvous_handler_registry
      from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler

      def create_my_rdzv(params: RendezvousParameters):
        return MyCustomRdzv(params)

      rendezvous_handler_registry.register("my_rdzv_backend_name", create_my_rdzv)

      my_rdzv_handler = get_rendezvous_handler("my_rdzv_backend_name", RendezvousParameters)
    """
    return handler_registry.create_handler(params)
