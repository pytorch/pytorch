# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch import SymInt


torch.library.define(
    "device_mesh::_runtime_get_rank",
    "() -> SymInt",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.register_fake("device_mesh::_runtime_get_rank")
def _runtime_get_rank_fake() -> SymInt:
    ctx = torch._custom_op.impl.get_ctx()
    return ctx._shape_env.create_unbacked_symint()


@torch.library.impl("device_mesh::_runtime_get_rank", "CompositeExplicitAutograd")
def _runtime_get_rank_impl() -> int:
    from torch.distributed.distributed_c10d import _get_rank

    return _get_rank()
