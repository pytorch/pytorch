# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a ``Shard`` bucket's sharded parameters keep their data.

Each parameter keeps its sharded data at the start of a slot of the bucket's
storage: the padded shard's elements (the rank-0 chunk's) in its dtype, so that
the ranks' slots line up in the collectives, aligned to its element size. A
``Shard(0)`` shard is stored padded, so the all-gather needs no copy to pad it;
a ``Shard(i)``, i > 0, shard is stored as is, since padding would sit between
rows. A parameter whose data is a tensor subclass, or defines an all-gather
extension, keeps storage of its own, and so does every parameter of its bucket.

Modules may replace a sharded parameter, or its data, after sharding:
``to_empty()``, ``load_state_dict(assign=True)``, ``module.to(dtype)``,
``module.cpu()``. ``ParamHoming.refresh`` copies such data back into the
bucket's storage, in new storage when dtypes or the device changed, and points
the parameters at their slots again.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn as nn

from .bucket_runtime import BucketParam
from .param_extensions import ParamExtension


if TYPE_CHECKING:
    import logging

    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_runtime import BucketRuntime
    from .bucket_storage import ParamInfo, ParamMeta


class ParamHoming:
    """Keeps one bucket's sharded parameters' data in its storage (see the
    module docstring), and their all-gather extensions."""

    def __init__(
        self,
        runtime: BucketRuntime,
        device: torch.device,
        extension_mesh: DeviceMesh | Any,
        mp_policies: list[Any],
    ) -> None:
        self.runtime = runtime
        # Where the storage lives when the parameters' devices disagree
        self.device = device
        # The mesh passed to fsdp_pre_all_gather, and each parameter's mixed
        # precision policy, resolved for it
        self.extension_mesh = extension_mesh
        self.mp_policies = mp_policies
        # Each parameter's 1-D slot: the stored data first, then zeros
        self.slots: list[torch.Tensor] = []
        # The local tensor that homing bound each sharded parameter to (None
        # for an alias), which a replaced parameter or local tensor is not
        self.locals: list[torch.Tensor | None] = []
        # The (numel, dtype) of each slot, which storage must match for reuse
        self.slot_keys: list[tuple[int, torch.dtype]] = []
        # (index, first index) of parameters that another entry of the bucket
        # also holds (see home)
        self.aliases: list[tuple[int, int]] = []
        # Whether every parameter keeps storage of its own (see own_storage)
        self.own_storage = False
        self.extensions: list[ParamExtension | None] = []

    @property
    def storage(self):
        return self.runtime.bucket_storage

    @property
    def world_size(self) -> int:
        return self.storage._mesh.size()

    # Slots #
    def _geometry(self, info: ParamInfo) -> tuple[int, torch.Size, int, int]:
        """(slot numel, stored size, shard dim, local length on the dim)."""
        placement = info.placement
        dim = placement.dim  # type: ignore[attr-defined]
        padded = placement.compute_local_shape(info.global_shape, 0, self.world_size)
        stored = padded if dim == 0 else info.local_shape
        return padded.numel(), stored, dim, info.local_shape[dim]

    @torch.no_grad()
    def _bind(
        self,
        index: int,
        slot: torch.Tensor | None,
        source: torch.Tensor,
        *,
        create: bool,
        holds_param: bool = True,
    ) -> torch.Tensor:
        """Keep parameter ``index``'s data at the start of ``slot``, or for
        ``None`` in storage of its own; copy ``source`` (its shard at
        construction, or a replaced local tensor) there, and point its sharded
        parameter at it, unless another entry holds the parameter
        (``holds_param=False``). ``create`` builds the sharded parameter."""
        runtime = self.runtime
        info = runtime.bucket_params[index].param_info
        slot_numel, stored_size, dim, length = self._geometry(info)
        if slot is None:
            slot = source.new_zeros(slot_numel)
            offload = self.storage._cpu_offload
            if offload is not None and not slot.is_meta:
                slot = slot.cpu()
                if offload.pin_memory:
                    slot = slot.pin_memory()
        stored_numel = stored_size.numel()
        stored = slot if slot.numel() == stored_numel else slot[:stored_numel]
        local = stored.view(stored_size).narrow(dim, 0, length)
        if not local.is_meta and local.numel() > 0:
            # A replaced Shard(0) local tensor may hold the padding too
            local.copy_(source.narrow(dim, 0, length))
        representation = runtime.representation
        if create:
            param = representation.new_sharded_param(index, local, info)
            bucket_param = runtime.bucket_params[index]
            for owner in (bucket_param.param_owner, *bucket_param.shared_owners):
                _set_param(owner.module, owner.param_name, param)
            runtime.bucket_params[index] = replace(bucket_param, sharded_param=param)
        elif holds_param:
            representation.set_local_tensor(
                index, runtime.bucket_params[index].sharded_param, local
            )
        return slot.view(-1)

    def home(self, shards: list[torch.Tensor] | None = None) -> None:
        """Keep the parameters' sharded data in one byte buffer, each at the
        start of its slot: ``shards`` (taken at construction, which creates the
        sharded parameters), or the data of sharded parameters replaced since,
        in new storage if their dtypes or device changed. Parameters with
        storage of their own (``own_storage``) keep it."""
        runtime = self.runtime
        representation = runtime.representation
        bucket_params = runtime.bucket_params
        create = shards is not None
        sources = (
            shards
            if shards is not None
            else [representation.local_tensor(bp.sharded_param) for bp in bucket_params]
        )
        if create:
            self.own_storage = any(
                ParamExtension.defined_by(s) or type(s) is not torch.Tensor
                for s in sources
            )
        if self.own_storage:
            self.slots = [
                self._bind(i, None, s, create=create) if create else self.slots[i]
                for i, s in enumerate(sources)
            ]
            self._rebuild_infos()
            return
        geometry = [self._geometry(bp.param_info) for bp in bucket_params]
        keys = [(g[0], s.dtype) for g, s in zip(geometry, sources, strict=True)]
        devices = {s.device for s in sources}
        if torch.device("meta") in devices and len(devices) > 1:
            # Some parameters are materialized and others still on meta, e.g.
            # while to_empty() visits the bucket's modules one by one: each
            # keeps its local tensor until all are materialized
            self.slots = [
                self._bind(i, None, s, create=True) if create else self.slots[i]
                for i, s in enumerate(sources)
            ]
            self.locals = []
            self._rebuild_infos()
            return
        if torch.device("meta") in devices:
            # Meta parameters stay on meta until to_empty()
            device = torch.device("meta")
        elif self.storage._cpu_offload is not None:
            device = torch.device("cpu")
        else:
            # The parameters' device, as sharded parameters stay where
            # Module._apply (e.g. module.cpu()) moved them
            device = devices.pop() if len(devices) == 1 else self.device
        offsets = _aligned_offsets(keys)
        total = offsets[-1] + keys[-1][0] * keys[-1][1].itemsize if keys else 0
        byte_storage = self.storage._byte_storage
        reuse = keys == self.slot_keys and byte_storage.device == device
        if not reuse:
            offload = self.storage._cpu_offload
            byte_storage = torch.zeros(
                total,
                dtype=torch.uint8,
                device=device,
                pin_memory=offload is not None
                and offload.pin_memory
                and device.type == "cpu",
            )
        slots: list[torch.Tensor] = []
        locals_: list[torch.Tensor | None] = []
        first_index: dict[int, int] = {}
        self.aliases = []
        for index, ((numel, dtype), offset, source) in enumerate(
            zip(keys, offsets, sources, strict=True)
        ):
            view = byte_storage[offset : offset + numel * dtype.itemsize].view(dtype)
            sharded_param = bucket_params[index].sharded_param
            if not create:
                first = first_index.setdefault(id(sharded_param), index)
                if first != index:
                    # A parameter tied after sharding (e.g. re-tied after
                    # to_empty) has an entry per module slot, so a slot per
                    # entry. The parameter views the first slot, and each
                    # unshard copies it into the others (sync_aliases).
                    self.aliases.append((index, first))
                    slots.append(
                        self._bind(index, view, source, create=False, holds_param=False)
                    )
                    locals_.append(None)
                    continue
                local = representation.local_tensor(sharded_param)
                if reuse and local is self.locals[index]:
                    slots.append(view)
                    locals_.append(local)
                    continue
            slots.append(self._bind(index, view, source, create=create))
            locals_.append(
                representation.local_tensor(bucket_params[index].sharded_param)
            )
        self.slots = slots
        self.locals = locals_
        self.slot_keys = keys
        self.storage._byte_storage = byte_storage
        self.storage._total_bytes = total
        self._rebuild_infos()

    def homed(self) -> bool:
        """Whether each sharded parameter's local tensor is still the slot that
        ``home`` bound it to, which a replaced sharded parameter or local
        tensor is not."""
        if self.own_storage:
            return True
        runtime = self.runtime
        if len(self.locals) != len(runtime.bucket_params):
            return False
        aliases = {index for index, _ in self.aliases}
        local_tensor = runtime.representation.local_tensor
        return all(
            index in aliases or local_tensor(bp.sharded_param) is local
            for index, (bp, local) in enumerate(
                zip(runtime.bucket_params, self.locals, strict=True)
            )
        )

    @torch.no_grad()
    def sync_aliases(self) -> None:
        """Copy each tied parameter's data into its other slots (see home)."""
        for index, first in self.aliases:
            self.slots[index].copy_(self.slots[first])

    # Re-reading parameters #
    def reset_sharded_params(self) -> None:
        """Re-read the sharded parameters from their modules after ops like
        ``nn.Module._apply`` or ``load_state_dict(assign=True)`` that may replace
        them or their local tensors, keeping their grad dtype policies."""
        runtime = self.runtime
        representation = runtime.representation
        for index, bp in enumerate(runtime.bucket_params):
            info = bp.param_info
            old = bp.sharded_param
            self._capture_grad_dtype(index, old)
            new = bp.param_owner.param
            if new is not old:
                if torch.__future__.get_swap_module_params_on_conversion():
                    raise AssertionError(
                        f"Expects swap_tensors to preserve object but got {new} "
                        f"instead of {old}"
                    )
                self._capture_grad_dtype(index, new)
                runtime.bucket_params[index] = replace(
                    runtime.bucket_params[index], sharded_param=new
                )
            info = runtime.bucket_params[index].param_info
            if info.has_explicit_grad_dtype and not new._has_grad_dtype_override:
                new.grad_dtype = info.grad_dtype_override
            local = representation.local_tensor(new)
            if local.is_meta:
                continue
            _, stored_size, _, _ = self._geometry(info)
            if local.size() not in (info.local_shape, stored_size):
                raise AssertionError(
                    f"{info.fqn} expects a local tensor of size {info.local_shape} "
                    f"but got {local.size()}"
                )
            if self.own_storage and not _same_storage_slot(local, self.slots[index]):
                self.slots[index] = self._bind(index, None, local, create=False)
            # Otherwise refresh copies a replaced local tensor into the storage
            representation.refresh_param(index, new)

    def save_for_apply(self) -> dict[int, torch.Tensor]:
        """Before ``nn.Module._apply`` (e.g. ``module.to(dtype)``), which swaps
        in parameters without grad_dtype: capture the grad dtype policies, and
        take out the grads whose dtype an explicit grad_dtype owns, which
        ``_apply`` would convert with the parameter."""
        saved: dict[int, torch.Tensor] = {}
        for index in range(len(self.runtime.bucket_params)):
            param = self.runtime.bucket_params[index].sharded_param
            self._capture_grad_dtype(index, param)
            grad = param.grad
            info = self.runtime.bucket_params[index].param_info
            if grad is not None and info.has_explicit_grad_dtype:
                saved[index] = grad
                param.grad = None
        return saved

    @torch.no_grad()
    def restore_after_apply(self, saved: dict[int, torch.Tensor]) -> None:
        """After ``nn.Module._apply``: re-read the parameters, give back the
        grads ``save_for_apply`` took out, and re-home the data."""
        self.reset_sharded_params()
        for index, grad in saved.items():
            param = self.runtime.bucket_params[index].sharded_param
            param.grad = grad.to(device=param.device).requires_grad_(grad.requires_grad)
        self.refresh()

    def _capture_grad_dtype(self, index: int, param: nn.Parameter) -> None:
        # Parameters rebuilt by nn.Module._apply or load_state_dict(assign=True)
        # lose grad_dtype, and an override cannot be unset, so only an explicit
        # grad_dtype on the parameter updates the policy.
        if param._has_grad_dtype_override:
            bp = self.runtime.bucket_params[index]
            self.runtime.bucket_params[index] = replace(
                bp,
                param_info=replace(
                    bp.param_info,
                    has_explicit_grad_dtype=True,
                    grad_dtype_override=param.grad_dtype,
                ),
            )

    def init_extensions(self) -> None:
        """Read the local shards' all-gather extensions."""
        runtime = self.runtime
        local_tensor = runtime.representation.local_tensor
        reshards_to_smaller_group = (
            runtime.bucket_storage._post_forward_group is not None
        )
        self.extensions = [
            ParamExtension(
                local,
                bp.param_info,
                self.world_size,
                reshards_to_smaller_group=reshards_to_smaller_group,
            )
            if ParamExtension.defined_by(local := local_tensor(bp.sharded_param))
            else None
            for bp in runtime.bucket_params
        ]

    def _extension(self, index: int, local: torch.Tensor) -> ParamExtension | None:
        """Parameter ``index``'s extension, built if its local shard gained one
        since ``init_extensions``."""
        if not self.extensions:
            self.extensions = [None] * len(self.runtime.bucket_params)
        extension = self.extensions[index]
        if extension is None and ParamExtension.defined_by(local):
            extension = self.extensions[index] = ParamExtension(
                local,
                self.runtime.bucket_params[index].param_info,
                self.world_size,
                reshards_to_smaller_group=self.storage._post_forward_group is not None,
            )
        return extension

    def has_extension(self, index: int) -> bool:
        runtime = self.runtime
        local = runtime.representation.local_tensor(
            runtime.bucket_params[index].sharded_param
        )
        return hasattr(local, "fsdp_pre_all_gather")

    def check_reduce_unused_params(self) -> None:
        # Checks configuration rather than gradient presence so every rank
        # raises together
        for bp in self.runtime.bucket_params:
            if (
                bp.sharded_param.requires_grad
                and bp.param_info.unsharded_grad_dtype is None
            ):
                raise ValueError(
                    "Reducing unused parameters with grad_dtype=None requires an "
                    "explicit MixedPrecisionPolicy.reduce_dtype so every rank can "
                    f"build zero gradients of the same dtype: {bp.param_info.fqn}. "
                    "Set reduce_dtype to the dtype its gradients already have."
                )

    @property
    def has_extensions(self) -> bool:
        local_tensor = self.runtime.representation.local_tensor
        return any(
            hasattr(local_tensor(bp.sharded_param), "fsdp_pre_all_gather")
            for bp in self.runtime.bucket_params
        )

    def refresh(self) -> None:
        """Re-home parameters whose sharded data no longer views the bucket
        (see ``homed``) and re-read their metadata. Runs before an unshard that
        may follow such a change."""
        if self.runtime.is_unsharded:
            return
        if not self.homed():
            self.home()
        else:
            self._rebuild_infos()

    def _rebuild_infos(self) -> None:
        """The parameters' infos from their current data and settings: dtypes
        may have changed (``module.to(dtype)``), and slot offsets with them."""
        runtime = self.runtime
        storage = runtime.bucket_storage
        offsets = _aligned_offsets([(slot.numel(), slot.dtype) for slot in self.slots])
        infos: dict[str, ParamInfo] = {}
        new_params: list[BucketParam] = []
        for index, (bp, slot, offset) in enumerate(
            zip(runtime.bucket_params, self.slots, offsets, strict=True)
        ):
            old = bp.param_info
            dtype = slot.dtype
            policy = self.mp_policies[index]
            requested = getattr(policy, "param_dtype", None)
            # No cast if it is the parameter's dtype or the parameter is not
            # floating point (mixed precision is only meaningful for those)
            param_dtype = (
                None if requested == dtype or not dtype.is_floating_point else requested
            )
            fields = {
                "dtype": dtype,
                "requires_grad": bp.sharded_param.requires_grad,
                "param_dtype": param_dtype,
                "reduce_dtype": getattr(policy, "reduce_dtype", None),
                "byte_offset": offset,
                "storage_nbytes": slot.numel() * slot.element_size(),
            }
            # Explicit unshards refresh every time, so keep unchanged infos
            if any(getattr(old, name) != value for name, value in fields.items()):
                bp = replace(bp, param_info=replace(old, **fields))
            infos[bp.param_info.fqn] = bp.param_info
            new_params.append(bp)
        trainable = [
            info.unsharded_grad_dtype or info.unsharded_dtype
            for info in infos.values()
            if info.requires_grad
        ]
        bucket_reduce_dtype = (
            functools.reduce(torch.promote_types, trainable) if trainable else None
        )
        for info in infos.values():
            info.bucket_reduce_dtype = bucket_reduce_dtype
        storage._param_infos = infos
        runtime.bucket_params = new_params

    def rename(self, fqns: list[str]) -> None:
        """Key the parameters' infos by ``fqns``, e.g. once a root module names
        them."""
        runtime = self.runtime
        runtime.bucket_params = [
            replace(bp, param_info=replace(bp.param_info, fqn=fqn))
            for bp, fqn in zip(runtime.bucket_params, fqns, strict=True)
        ]
        runtime.bucket_storage._param_infos = {
            bp.param_info.fqn: bp.param_info for bp in runtime.bucket_params
        }

    # All-gathers of parameters with storage of their own or an extension #
    def defines_all_gathers(self) -> bool:
        return self.own_storage or self.has_extensions

    def all_gather_inputs(self, index: int):
        from .placements.shard_all_gather import ParamAllGatherInputs

        runtime = self.runtime
        bp = runtime.bucket_params[index]
        info = bp.param_info
        local = runtime.representation.local_tensor(bp.sharded_param)
        storage = runtime.bucket_storage
        device = storage._compute_device if storage._cpu_offload is not None else None
        if hasattr(local, "fsdp_pre_all_gather"):
            extension = cast(ParamExtension, self._extension(index, local))
            return extension.all_gather_inputs(
                local,
                info,
                self.extension_mesh,
                bp.param_owner.module,
                self.mp_policies[index],
                device,
            )
        if not self.own_storage:
            return None
        data = self.slots[index]
        if device is not None:
            data = data.to(device, non_blocking=True)
        slot_numel, _, dim, length = self._geometry(info)
        if dim != 0 and length != _padded_length(self, info):
            padded = data.new_zeros(
                info.placement.compute_local_shape(
                    info.global_shape, 0, self.world_size
                ),
                dtype=info.param_dtype,
            )
            padded.narrow(dim, 0, length).copy_(
                data[: info.local_shape.numel()].view(info.local_shape)
            )
            return ParamAllGatherInputs([padded.view(-1)])
        if info.param_dtype is not None and data.dtype != info.param_dtype:
            data = data.to(info.param_dtype)
        return ParamAllGatherInputs([data])

    def post_all_gather(
        self, index: int, outputs: tuple[torch.Tensor, ...], refill: bool
    ):
        runtime = self.runtime
        bp = runtime.bucket_params[index]
        local = runtime.representation.local_tensor(bp.sharded_param)
        if not hasattr(local, "fsdp_post_all_gather"):
            return None
        extension = cast(ParamExtension, self._extension(index, local))
        out = cast(list, runtime.unsharded_params)[index] if refill else None
        return extension.post_all_gather(
            local, bp.param_info, self.world_size, outputs, out
        )

    def releases_all_gather_outputs(self, index: int) -> bool:
        runtime = self.runtime
        local = runtime.representation.local_tensor(
            runtime.bucket_params[index].sharded_param
        )
        extension = self._extension(index, local)
        return extension is not None and extension.releases_outputs

    def check_post_forward_reshard(self) -> None:
        """The limits of resharding to the smaller post-forward group."""
        runtime = self.runtime
        group = runtime.bucket_storage._post_forward_group
        if group is None:
            raise AssertionError("Expected a post-forward group")
        world_size = group.size()
        all_gather = runtime.placement_state
        for index, bp in enumerate(runtime.bucket_params):
            info = bp.param_info
            param_all_gather = all_gather.params[index]  # type: ignore[attr-defined]
            if (num_outputs := len(param_all_gather.all_gather_outputs)) != 1:
                raise NotImplementedError(
                    f"Resharding {info.fqn} to smaller mesh is not supported for "
                    f"all-gather extensions with {num_outputs} inputs yet"
                )
            dim = info.placement.dim  # type: ignore[attr-defined]
            orig_size = info.global_shape
            if dim > 0 and orig_size.numel() and orig_size[0] % world_size:
                # Flat post-forward shards of unpadded Shard(i>0) parameters
                # match dim-0 chunks only if the world size divides dim 0
                raise NotImplementedError(
                    f"Resharding the Shard({dim}) parameter {info.fqn} to "
                    f"{world_size} ranks after forward requires dim 0 of its size "
                    f"{tuple(orig_size)} to be divisible by {world_size}"
                )
            local = runtime.representation.local_tensor(bp.sharded_param)
            if hasattr(local, "fsdp_pre_all_gather"):
                ParamExtension.check_post_forward_reshard(
                    info,
                    param_all_gather,
                    world_size,
                    bp.sharded_param.requires_grad,
                )

    def check_post_forward_unshard(self) -> None:
        local_tensor = self.runtime.representation.local_tensor
        for bp in self.runtime.bucket_params:
            if hasattr(local_tensor(bp.sharded_param), "fsdp_pre_all_gather"):
                raise NotImplementedError(
                    f"All-gathering the post-forward shard of {bp.param_info.fqn} "
                    "is not supported for all-gather extensions yet"
                )


def _padded_length(homing: ParamHoming, info: ParamInfo) -> int:
    dim = info.placement.dim  # type: ignore[attr-defined]
    return info.placement.compute_local_shape(info.global_shape, 0, homing.world_size)[
        dim
    ]


def _aligned_offsets(slots: list[tuple[int, torch.dtype]]) -> list[int]:
    """Byte offsets of ``(numel, dtype)`` slots packed in one buffer, each
    aligned to its element size so that it can be viewed in its dtype."""
    offsets = []
    offset = 0
    for numel, dtype in slots:
        itemsize = dtype.itemsize
        offset = -(-offset // itemsize) * itemsize
        offsets.append(offset)
        offset += numel * itemsize
    return offsets


def _same_storage_slot(tensor: torch.Tensor, data: torch.Tensor) -> bool:
    """Whether ``tensor`` starts at ``data`` in the same storage, compared by
    identity since empty storages all have data_ptr() 0 (e.g. a rank's shard of
    a (1, ...) parameter over 2 ranks)."""
    return bool(
        tensor.untyped_storage()._cdata == data.untyped_storage()._cdata
        and tensor.storage_offset() * tensor.element_size()
        == data.storage_offset() * data.element_size()
    )


def _set_param(module: nn.Module, param_name: str, param: nn.Parameter) -> None:
    # Bypasses nn.Module.__setattr__'s checks, which cost CPU time, unless the
    # module overrides it
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:
        setattr(module, param_name, param)


@dataclass(frozen=True)
class ParamSource:
    """A parameter of a bucket that ``build_homed_bucket`` builds."""

    # The parameter and the module slots that hold it
    param: nn.Parameter
    owner: Any
    shared_owners: tuple[Any, ...]
    # Its key in the bucket's infos until a root names it
    fqn: str
    # The tensor whose chunks the bucket stores: the parameter, or for an
    # outer-sharded parameter its local tensor
    data: torch.Tensor
    placement: Any
    meta: ParamMeta
    mp_policy: Any
    # The local shape to expose, if the frontend's differs for empty shards
    local_shape: torch.Size | None = None


def build_homed_bucket(
    sources: list[ParamSource],
    *,
    mesh: Any,
    device: torch.device,
    module: nn.Module,
    representation: Any,
    extension_mesh: Any,
    context: Any,
    offload_policy: Any = None,
    replicate_group: Any = None,
    reshard_after_forward: bool = True,
    post_forward_group: Any = None,
    logger: logging.Logger,
    log_name: str,
) -> tuple[BucketRuntime, ParamHoming]:
    """Build a ``Shard``-style bucket for ``sources``: its storage, runtime
    and ``ParamHoming``, which copies each parameter's shard into the storage
    and registers its sharded parameter (``representation.new_sharded_param``)
    in its module slots."""
    from .bucket_runtime import BucketRuntime
    from .bucket_storage import ShardedBucketStorage

    infos, _ = ShardedBucketStorage.create_param_infos(
        # The data tensors stand for their parameters, whose settings the
        # metas carry
        cast(list[tuple[str, nn.Parameter]], [(src.fqn, src.data) for src in sources]),
        mesh,
        {src.fqn: (src.placement,) for src in sources},
        metas={src.fqn: src.meta for src in sources},
    )
    infos = {
        src.fqn: replace(
            info, local_shape=src.local_shape, local_numel=src.local_shape.numel()
        )
        if src.local_shape is not None
        else info
        for src, info in zip(sources, infos.values(), strict=True)
    }
    storage = ShardedBucketStorage(
        torch.empty(0, dtype=torch.uint8, device=device),
        infos,
        mesh,
        0,
        module,
        reshard_after_forward=reshard_after_forward,
        fsdp2_compatible=True,
    )
    storage._post_forward_group = post_forward_group
    if replicate_group is not None:
        storage.set_replicate_group(replicate_group)
    if offload_policy is not None:
        storage.set_cpu_offload(offload_policy, device)
    runtime = BucketRuntime(
        bucket_storage=storage,
        bucket_params=[
            BucketParam(src.owner, info, src.param, src.shared_owners)
            for src, info in zip(sources, infos.values(), strict=True)
        ],
        context=context,
        debug_fqn=None,
        representation=representation,
        logger=logger,
        log_name=log_name,
    )
    homing = representation.homing = ParamHoming(
        runtime, device, extension_mesh, [src.mp_policy for src in sources]
    )
    rank, world_size = mesh.get_local_rank(), mesh.size()
    # The rank's shard of each parameter, which homing copies into the
    # storage, so that the parameters' data can be freed once this returns
    homing.home(
        [
            info.placement.extract_local_shard(src.data, rank, world_size)
            for src, info in zip(sources, infos.values(), strict=True)
        ]
    )
    homing.init_extensions()
    return runtime, homing


def param_owners(
    params: list[nn.Parameter], modules: tuple[nn.Module, ...]
) -> list[tuple[Any, tuple[Any, ...]]]:
    """Each parameter's module slot and the other slots that hold it (shared
    parameter: ``lin1.weight = lin2.weight``; shared module: ``mlp.lin1 =
    mlp.lin2``), walking ``modules`` without removing duplicates."""
    from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates

    from .bucket_runtime import ParamOwnerRef

    params_set = set(params)
    found: dict[nn.Parameter, list[Any]] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param in params_set:
                    found.setdefault(param, []).append(
                        ParamOwnerRef(submodule, param_name)
                    )
    if len(found) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {modules}")
    return [(found[param][0], tuple(found[param][1:])) for param in params]


__all__: list[str] = []
_ = cast
