# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP's all-gather extensions for the parameters of ``Shard`` buckets.

A local shard that is a tensor subclass may define how it is all-gathered
(``fsdp_pre_all_gather``) and how its unsharded tensor is built from the
outputs (``fsdp_post_all_gather``), e.g. to all-gather quantized data and
scales. The engine calls them for any frontend's parameters.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, TYPE_CHECKING

from .placements.shard_all_gather import ParamAllGatherInputs


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch
    import torch.nn as nn
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo
    from .placements.shard_all_gather import ParamAllGather


class ParamExtension:
    """A local shard's all-gather extension and its state across one unshard."""

    def __init__(
        self,
        local: torch.Tensor,
        info: ParamInfo,
        world_size: int,
        *,
        reshards_to_smaller_group: bool,
    ) -> None:
        has_pre = hasattr(local, "fsdp_pre_all_gather")
        has_post = hasattr(local, "fsdp_post_all_gather")
        release_fn = getattr(
            local, "fsdp_should_release_all_gather_outputs_after_post_all_gather", None
        )
        if has_pre != has_post:
            raise AssertionError(
                "Both fsdp_pre_all_gather and fsdp_post_all_gather should be defined "
                f"if using all-gather extensions: {local}"
            )
        if release_fn is not None and not has_post:
            raise AssertionError(
                "fsdp_should_release_all_gather_outputs_after_post_all_gather "
                "requires fsdp_pre_all_gather and fsdp_post_all_gather to be "
                f"defined: {local}"
            )
        if has_pre:
            _validate_uneven_shard_pre_all_gather(
                local.fsdp_pre_all_gather,  # type: ignore[attr-defined]
                info.global_shape,
                shard_dim=_shard_dim(info),
                shard_world_size=world_size,
            )
        # Metadata from the latest fsdp_pre_all_gather, for the next
        # fsdp_post_all_gather
        self.metadata: Any = None
        self.releases_outputs = False
        if release_fn is not None:
            # The extension owns whether its post-all-gather representation
            # aliases the raw all-gather outputs, so it must declare when those
            # outputs are no longer needed.
            should_release = release_fn()
            if not isinstance(should_release, bool):
                raise AssertionError(
                    "fsdp_should_release_all_gather_outputs_after_post_all_gather "
                    f"must return a bool, got {type(should_release)}"
                )
            if should_release and reshards_to_smaller_group:
                raise NotImplementedError(
                    "Releasing all-gather outputs after post-all-gather is not "
                    "supported when reshard_after_forward is an int because FSDP "
                    "uses those outputs to construct the post-forward shard"
                )
            self.releases_outputs = should_release

    @staticmethod
    def defined_by(local: torch.Tensor) -> bool:
        """Whether ``local`` defines any part of an all-gather extension (an
        incomplete one raises when the extension is built)."""
        return any(
            hasattr(local, name)
            for name in (
                "fsdp_pre_all_gather",
                "fsdp_post_all_gather",
                "fsdp_should_release_all_gather_outputs_after_post_all_gather",
            )
        )

    def all_gather_inputs(
        self,
        local: torch.Tensor,
        info: ParamInfo,
        mesh: DeviceMesh,
        module: nn.Module,
        mp_policy: Any,
        device: torch.device | None,
    ) -> ParamAllGatherInputs:
        """Call ``fsdp_pre_all_gather``; ``device`` is where to gather a local
        shard stored elsewhere (CPU offload)."""
        if device is not None:
            local = local.to(device, non_blocking=True)
        pre_all_gather = local.fsdp_pre_all_gather  # type: ignore[attr-defined]
        signature = inspect.signature(pre_all_gather)
        num_fn_params = len(signature.parameters)
        # Old signature only passes mesh; keep for BC for now
        if num_fn_params not in (1, 5):
            raise AssertionError(
                f"Invalid fsdp_pre_all_gather: {signature}\n"
                "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                "outer_size: torch.Size, outer_stride: tuple[int, ...], "
                "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
            )
        if num_fn_params == 1:
            inputs, self.metadata = pre_all_gather(mesh)
        else:
            inputs, self.metadata = pre_all_gather(
                mesh, info.global_shape, info.global_stride, module, mp_policy
            )
        # Only the 5-argument hook gets outer_size to pad with (#137005), so
        # only its Tensor inputs are checked for the padded sharded size
        return ParamAllGatherInputs(
            inputs, from_extension=True, check_padding=num_fn_params == 5
        )

    def post_all_gather(
        self,
        local: torch.Tensor,
        info: ParamInfo,
        world_size: int,
        outputs: tuple[torch.Tensor, ...],
        out: nn.Parameter | None,
    ) -> tuple[torch.Tensor, Sequence[torch.Tensor]] | None:
        """Call ``fsdp_post_all_gather``, into ``out`` on a refill."""
        dtype = info.unsharded_dtype
        metadata, self.metadata = self.metadata, None
        post_all_gather = local.fsdp_post_all_gather  # type: ignore[attr-defined]
        if out is not None:
            post_all_gather(outputs, metadata, dtype, out=out)
            return None
        unsharded, inner_tensors = post_all_gather(outputs, metadata, dtype)
        shard_dim = _shard_dim(info)
        orig_size = info.global_shape
        if (
            shard_dim != 0
            and orig_size[shard_dim] % world_size
            and unsharded.numel() != orig_size.numel()
        ):
            # Unlike Shard(0) padding, which a prefix of the output drops,
            # padding inside the rows must be dropped by the extension. Every
            # rank gets the same output, so every rank raises.
            raise ValueError(
                f"{info.fqn} is unevenly sharded on dim {shard_dim}, "
                "so fsdp_post_all_gather must drop the padding in each row of "
                "the all-gather outputs and return the "
                f"{tuple(orig_size)} parameter, but it returned a tensor "
                f"of size {tuple(unsharded.size())}"
            )
        return unsharded, inner_tensors

    @staticmethod
    def check_post_forward_reshard(
        info: ParamInfo,
        all_gather: ParamAllGather,
        world_size: int,
        requires_grad: bool,
    ) -> None:
        """The limits of resharding an extension's outputs to ``world_size``
        ranks after forward: the post-forward parameter views a chunk of the
        extension's output. The largest post-forward shard is the same bound on
        every rank, so all ranks raise together."""
        orig_size = info.global_shape
        output = all_gather.all_gather_outputs[0]
        rows = -(-orig_size[0] // world_size)
        shard_numel = rows * math.prod(orig_size[1:])
        if output.numel() // world_size < shard_numel:
            raise NotImplementedError(
                f"Resharding {info.fqn} to smaller mesh needs post-forward shards "
                f"of up to {shard_numel} elements, but its all-gather extension "
                f"output has {output.numel() // world_size} per rank"
            )
        if requires_grad and not (output.is_floating_point() or output.is_complex()):
            raise NotImplementedError(
                f"Resharding {info.fqn} to smaller mesh needs a floating or complex "
                "all-gather extension output to require gradients, but its output "
                f"has dtype {output.dtype}"
            )


def _shard_dim(info: ParamInfo) -> int:
    return getattr(info.placement, "dim", 0)


def _validate_uneven_shard_pre_all_gather(
    pre_all_gather: Callable[..., Any],
    orig_size: torch.Size,
    *,
    shard_dim: int,
    shard_world_size: int,
) -> None:
    """Raises if the parameter is unevenly sharded on a nonzero dim and its
    ``fsdp_pre_all_gather`` takes only ``mesh``: only the 5-argument hook gets
    ``outer_size`` to pad its shard with (#137005)."""
    uneven = shard_dim != 0 and orig_size[shard_dim] % shard_world_size != 0
    if uneven and len(inspect.signature(pre_all_gather).parameters) == 1:
        raise NotImplementedError(
            f"FSDP does not support uneven sharding on dim {shard_dim} for "
            "all-gather extensions whose fsdp_pre_all_gather takes only "
            f"mesh: {orig_size} (world size: {shard_world_size}). Use "
            "fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, "
            "mp_policy) to get the unsharded size to pad the shard with."
        )
