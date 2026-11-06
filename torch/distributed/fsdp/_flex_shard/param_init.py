# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize FlexShard parameters the way one device would."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .placement_contract import GlobalLayout


def init_from_full(
    local: torch.Tensor,
    layout: GlobalLayout,
    init_fn: Callable[[torch.Tensor], object],
) -> None:
    """Fill ``local`` with its regions of a full-shape initialization.

    ``layout`` says where ``local`` sits in the full parameter, as
    ``get_flex_shard_global_layouts`` reports for checkpointing. ``init_fn``
    fills an empty tensor of the full shape in place, like the ``torch.nn.init``
    functions, and ``local`` keeps its regions: a checkpoint load whose source
    is the tensor ``init_fn`` just filled. With the same RNG state on every
    rank, the shards then equal one device's initialization, and every rank's
    RNG advances as that device's would. Each call briefly holds one full-size
    tensor.
    """
    full = local.new_empty(layout.global_shape)
    init_fn(full)
    with torch.no_grad():
        for global_offset, local_offset, size in zip(
            layout.global_offsets,
            layout.local_offsets,
            layout.local_sizes,
            strict=True,
        ):
            local[_box(local_offset, size)].copy_(full[_box(global_offset, size)])


def _box(offset: tuple[int, ...], size: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(start, start + length) for start, length in zip(offset, size, strict=True)
    )
