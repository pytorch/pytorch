# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast

import spmd_types as spmd

import torch.nn as nn
from torch.distributed.fsdp._flex_shard import get_global_layout, GlobalLayout
from torch.distributed.fsdp._flex_shard.layout_adapters.spmd_types import (
    spmd_types_to_global_layout,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class _Mesh:
    """dp of size 2 at rank 1, tp of size 4 at rank 2."""

    mesh_dim_names = ("dp", "tp")
    _dims = {"dp": (2, 1), "tp": (4, 2)}

    def size(self, dim: int) -> int:
        return self._dims[self.mesh_dim_names[dim]][0]

    def get_local_rank(self, name: str) -> int:
        return self._dims[name][1]


class TestFlexShardSpmdTypesLayout(TestCase):
    def test_adapter_declares_layouts(self) -> None:
        model = nn.Linear(4, 3)
        spmd_types_to_global_layout(
            model,
            {
                "weight": spmd.SpmdType({"dp": spmd.R, "tp": spmd.S(0)}),
                "bias": spmd.SpmdType({}, spmd.PartitionSpec(("dp", "tp"))),
            },
            cast(Any, _Mesh()),
        )
        self.assertEqual(
            get_global_layout(model.weight),
            GlobalLayout((12, 4), ((6, 0),), ((0, 0),), ((3, 4),)),
        )
        # dp splits the dim first; tp then splits dp's chunk.
        self.assertEqual(
            get_global_layout(model.bias),
            GlobalLayout((24,), ((18,),), ((0,),), ((3,),)),
        )


if __name__ == "__main__":
    run_tests()
