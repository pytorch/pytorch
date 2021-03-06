# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from . import etcd_rendezvous
from .api import (
    RendezvousHandler,
    RendezvousHandlerFactory,
    RendezvousParameters,
)

_factory = RendezvousHandlerFactory()
_factory.register("etcd", etcd_rendezvous.create_rdzv_handler)


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    return _factory.create_handler(params)
