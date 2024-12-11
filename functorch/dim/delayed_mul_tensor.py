# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import _Tensor, Tensor
from .reference import _dims, _enable_layers, llist, ltuple


class DelayedMulTensor(_Tensor):
    def __init__(self, lhs, rhs):
        self._lhs, self._rhs = lhs, rhs
        self._data = None
        self._levels_data = None
        self._has_device = lhs._has_device or rhs._has_device
        self._batchtensor_data = None
        self._tensor_data = None

    @property
    def _levels(self):
        if self._levels_data is None:
            levels = llist(self._lhs._levels)
            for l in self._rhs._levels:
                if l not in levels:
                    levels.append(l)
            self._levels_data = ltuple(levels)
        return self._levels_data

    @property
    def _batchtensor(self):
        if self._batchtensor_data is None:
            with _enable_layers(self._levels):
                print("bt multiply fallback")
                self._batchtensor_data = self._lhs._batchtensor * self._rhs._batchtensor
        return self._batchtensor_data

    @property
    def _tensor(self):
        if self._tensor_data is None:
            self._tensor_data = Tensor.from_batched(
                self._batchtensor, self._has_device
            )._tensor
        return self._tensor_data

    @property
    def ndim(self):
        return self._batchtensor.ndim

    @property
    def dims(self):
        return ltuple(super().dims)

    def sum(self, dim):
        dims = _dims(dim, 0, False, False)
        n = ord("a")
        all_levels = self._levels

        def to_char(d):
            return chr(n + all_levels.index(d))

        plhs, levelslhs = self._lhs._tensor, self._lhs._levels
        prhs, levelsrhs = self._rhs._tensor, self._rhs._levels
        new_levels = [l for l in self._levels if l not in dims]
        fmt = "".join(
            [
                *(to_char(d) for d in levelslhs),
                ",",
                *(to_char(d) for d in levelsrhs),
                "->",
                *(to_char(d) for d in new_levels),
            ]
        )
        result_data = torch.einsum(fmt, (plhs, prhs))
        return Tensor.from_positional(result_data, new_levels, True)
