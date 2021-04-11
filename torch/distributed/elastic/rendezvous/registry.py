# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import RendezvousHandler, RendezvousParameters
from .api import rendezvous_handler_registry as handler_registry


def _create_etcd_handler(params: RendezvousParameters) -> RendezvousHandler:
    from . import etcd_rendezvous

    return etcd_rendezvous.create_rdzv_handler(params)


def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)


# The legacy function kept for backwards compatibility.
def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    return handler_registry.create_handler(params)
