# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functorch._C import (
    _vmap_add_layers,
    _vmap_remove_layers,
)

from contextlib import contextmanager

_enabled = False
@contextmanager
def _enable_layers(dims):
    global _enabled
    assert not _enabled
    input = list(sorted((d._level, d.size) for d in dims if not isinstance(d, int)))
    n = len(input)
    try:
        _vmap_add_layers(input)
        _enabled = True
        yield
    finally:
        _enabled = False
        _vmap_remove_layers(n)
