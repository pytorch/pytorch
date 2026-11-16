# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a bucket's sharded parameters keep their data.

Each parameter keeps its sharded data at the start of its slot of the bucket's
storage, the bytes that its placement reserves for it (e.g. the rank-0 chunk's
for ``Shard``, so that the ranks' slots line up in the collectives). A
parameter whose data is a tensor subclass, or defines an all-gather extension,
keeps storage of its own, and so does every parameter of its bucket.

Modules may replace a sharded parameter, or its data, after sharding:
``to_empty()``, ``load_state_dict(assign=True)``, ``module.to(dtype)``,
``module.cpu()``. ``ParamHoming.refresh`` copies such data back into the
bucket's storage, in new storage when dtypes or the device changed, and points
the parameters at their slots again.
"""

from __future__ import annotations

import functools
import math
import weakref
from dataclasses import dataclass, replace
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn as nn

from .bucket_runtime import BucketParam
from .param_extensions import ParamExtension


if TYPE_CHECKING:
    import logging

    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh

    from .api import CPUOffloadPolicy, MixedPrecisionPolicy
    from .bucket_runtime import (
        BucketCommContext,
        BucketRuntime,
        ParamOwnerRef,
        ParamRepresentation,
    )
    from .bucket_storage import ParamInfo, ParamMeta
    from .placement_contract import Placement


class ParamHoming:
    """Keeps one bucket's sharded parameters' data in its storage (see the
    module docstring), and their all-gather extensions."""

    def __init__(self, runtime: BucketRuntime) -> None:
        self.runtime = runtime
        # Set by install: where the storage lives when the parameters' devices
        # disagree, the mesh passed to fsdp_pre_all_gather, and each
        # parameter's mixed precision policy, resolved for it
        self.device: torch.device | None = None
        self.extension_mesh: DeviceMesh | None = None
        self.mp_policies: list[MixedPrecisionPolicy] | None = None
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
        # Whether modules replaced parameters since the last refresh (e.g.
        # load_state_dict(assign=True)), whose data the next unshard re-homes
        self.params_replaced = False
        # Parameters that were on meta at install and not replaced since
        self.unmaterialized: set[int] = set()

    @property
    def storage(self):
        return self.runtime.bucket_storage

    @property
    def world_size(self) -> int:
        return self.storage._mesh.size()

    # Slots #
    @staticmethod
    def _slot_numel(info: ParamInfo) -> int:
        return info.storage_nbytes // info.dtype.itemsize

    def _storage_device(self, devices: set[torch.device]) -> torch.device:
        if torch.device("meta") in devices:
            # Meta parameters stay on meta until to_empty()
            return torch.device("meta")
        if self.storage._cpu_offload is not None:
            return torch.device("cpu")
        # The parameters' device, as sharded parameters stay where
        # Module._apply (e.g. module.cpu()) moved them
        return devices.pop() if len(devices) == 1 else cast(torch.device, self.device)

    def _new_storage(self, nbytes: int, device: torch.device) -> torch.Tensor:
        offload = self.storage._cpu_offload
        pin = offload is not None and offload.pin_memory and device.type == "cpu"
        return torch.zeros(nbytes, dtype=torch.uint8, device=device, pin_memory=pin)

    def _layout(self, dtypes: list[torch.dtype]) -> tuple[list[int], int]:
        """The slots' byte offsets and the storage's size for data in
        ``dtypes``: the placements' layout, or after ``module.to(dtype)`` the
        slots packed anew."""
        infos = [bp.param_info for bp in self.runtime.bucket_params]
        if all(d == info.dtype for d, info in zip(dtypes, infos, strict=True)):
            return [info.byte_offset for info in infos], self.storage._total_bytes
        keys = [(self._slot_numel(i), d) for i, d in zip(infos, dtypes, strict=True)]
        offsets = _aligned_offsets(keys)
        return offsets, offsets[-1] + keys[-1][0] * keys[-1][1].itemsize

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
        if slot is None:
            slot = source.new_zeros(self._slot_numel(info))
            offload = self.storage._cpu_offload
            if offload is not None and not slot.is_meta:
                slot = slot.cpu()
                if offload.pin_memory:
                    slot = slot.pin_memory()
        local = slot.narrow(0, 0, info.local_numel).view(info.local_shape)
        if not local.is_meta and local.numel() > 0:
            # A replaced local tensor may hold the padding rows too
            local.copy_(source[tuple(slice(0, n) for n in info.local_shape)])
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

    def install(
        self,
        sources: list[torch.Tensor],
        *,
        device: torch.device,
        extension_mesh: DeviceMesh,
        mp_policies: list[MixedPrecisionPolicy],
    ) -> None:
        """Keep each parameter's shard of ``sources``, the tensors whose
        chunks the bucket stores, in the bucket's storage, as its placement
        lays it out, and register a sharded parameter that views it in the
        parameter's module slots. Parameters on meta while others are not
        leave their slots zero and count as unmaterialized until replaced
        (see ``check_materialized``)."""
        self.device, self.extension_mesh = device, extension_mesh
        self.mp_policies = mp_policies
        self.aliases = []
        runtime, storage = self.runtime, self.storage
        infos = [bp.param_info for bp in runtime.bucket_params]
        rank = storage._mesh.get_local_rank()
        sources = [source.detach() for source in sources]
        self.own_storage = any(
            ParamExtension.defined_by(s) or type(s) is not torch.Tensor for s in sources
        )
        if self.own_storage:
            if unsupported := next(
                (i for i in infos if not i.placement.supports_own_param_storage), None
            ):
                raise ValueError(
                    f"{unsupported.fqn} keeps storage of its own (a tensor subclass "
                    "or a parameter with an all-gather extension), which "
                    f"{unsupported.placement!r} does not support; use Shard."
                )
            self.slots = [
                self._bind(
                    i,
                    None,
                    info.placement.extract_local_shard(source, rank, self.world_size),
                    create=True,
                )
                for i, (info, source) in enumerate(zip(infos, sources, strict=True))
            ]
            self._rebuild_infos()
            return
        devices = {s.device for s in sources}
        materialized = devices - {torch.device("meta")}
        self.unmaterialized = (
            {i for i, s in enumerate(sources) if s.is_meta} if materialized else set()
        )
        byte_storage = self._new_storage(
            storage._total_bytes, self._storage_device(materialized or devices)
        )
        if not byte_storage.is_meta:
            for info, source in zip(infos, sources, strict=True):
                info.placement.copy_param_to_storage(
                    byte_storage, info, source, rank, self.world_size
                )
        storage._byte_storage = byte_storage
        self.slots = [
            self._slot(byte_storage, info.byte_offset, info) for info in infos
        ]
        for index, (info, slot) in enumerate(zip(infos, self.slots, strict=True)):
            param = runtime.representation.new_sharded_param(
                index, slot.narrow(0, 0, info.local_numel).view(info.local_shape), info
            )
            bp = runtime.bucket_params[index]
            for owner in (bp.param_owner, *bp.shared_owners):
                _set_param(owner.module, owner.param_name, param)
            runtime.bucket_params[index] = replace(bp, sharded_param=param)
        self.locals = [
            runtime.representation.local_tensor(bp.sharded_param)
            for bp in runtime.bucket_params
        ]
        self.slot_keys = [(self._slot_numel(info), info.dtype) for info in infos]
        self._rebuild_infos()

    def install_state_dict_hooks(self) -> None:
        """As FSDP2: state dicts read and load the sharded parameters, so
        resharding first, and a load (``assign=True`` replaces parameters) has
        the parameters re-read, which the next unshard re-homes. The hooks
        hold the homing weakly, so that a submodule kept alive does not keep
        the whole model alive."""
        homing = weakref.ref(self)
        owners = {
            owner.module
            for bp in self.runtime.bucket_params
            for owner in (bp.param_owner, *bp.shared_owners)
        }
        for module in owners:
            reshard = functools.partial(_reshard_before_state_dict, homing)
            module.register_state_dict_pre_hook(reshard)
            module._register_load_state_dict_pre_hook(reshard)
            module.register_load_state_dict_post_hook(
                functools.partial(_reread_after_load, homing)
            )

    def check_materialized(self, name: str) -> None:
        """Raise if parameters are still on meta, e.g. before the first
        forward."""
        runtime = self.runtime
        meta = [
            bp.param_info.fqn
            for index, bp in enumerate(runtime.bucket_params)
            if index in self.unmaterialized or bp.sharded_param.device.type == "meta"
        ]
        if meta:
            raise RuntimeError(
                f"{name} parameters should be materialized from meta device before "
                f"training, but the following were still on meta device: {meta}\n"
                "For example, call module.to_empty(device) to materialize to "
                "device and call module.reset_parameters() on each module to "
                "initialize values."
            )

    def before_unshard(self) -> None:
        """Called before an unshard reads the storage: re-home parameters that
        modules replaced since (``params_replaced``), and copy each tied
        parameter's data into its other slots."""
        if self.params_replaced:
            self.refresh()
        self.sync_aliases()

    def _slot(
        self,
        byte_storage: torch.Tensor,
        offset: int,
        info: ParamInfo,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        dtype = dtype or info.dtype
        nbytes = self._slot_numel(info) * dtype.itemsize
        return byte_storage[offset : offset + nbytes].view(dtype)

    def home(self) -> None:
        """Keep the data of sharded parameters replaced since construction in
        the bucket's storage again, at the start of their slots, in new storage
        if their dtypes or the device changed. Parameters with storage of their
        own (``own_storage``) keep it."""
        runtime = self.runtime
        representation = runtime.representation
        bucket_params = runtime.bucket_params
        sources = [
            representation.local_tensor(bp.sharded_param) for bp in bucket_params
        ]
        if self.own_storage:
            self._rebuild_infos()
            return
        devices = {s.device for s in sources}
        if torch.device("meta") in devices and len(devices) > 1:
            # Some parameters are materialized and others still on meta: each
            # keeps its local tensor until all are materialized (see install)
            self.locals = []
            self._rebuild_infos()
            return
        device = self._storage_device(devices)
        dtypes = [s.dtype for s in sources]
        offsets, total = self._layout(dtypes)
        keys = [
            (self._slot_numel(bp.param_info), d)
            for bp, d in zip(bucket_params, dtypes, strict=True)
        ]
        byte_storage = self.storage._byte_storage
        reuse = keys == self.slot_keys and byte_storage.device == device
        if not reuse:
            byte_storage = self._new_storage(total, device)
        slots: list[torch.Tensor] = []
        locals_: list[torch.Tensor | None] = []
        first_index: dict[int, int] = {}
        self.aliases = []
        for index, (offset, source) in enumerate(zip(offsets, sources, strict=True)):
            view = self._slot(
                byte_storage, offset, bucket_params[index].param_info, source.dtype
            )
            sharded_param = bucket_params[index].sharded_param
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
            # The local tensor may be the parameter itself, whose data an op
            # like to_empty() replaced in place
            if (
                reuse
                and local is self.locals[index]
                and _same_storage_slot(local, view)
            ):
                slots.append(view)
                locals_.append(local)
                continue
            slots.append(self._bind(index, view, source, create=False))
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
            index in aliases
            or (
                local is not None
                and local_tensor(bp.sharded_param) is local
                and _starts_at(local, slot)
            )
            for index, (bp, local, slot) in enumerate(
                zip(runtime.bucket_params, self.locals, self.slots, strict=True)
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
                # Slots that shared the parameter (e.g. tied weights) got
                # parameters of their own too, e.g. from to_empty(); tie again
                for owner in bp.shared_owners:
                    if owner.param is not new:
                        _set_param(owner.module, owner.param_name, new)
            info = runtime.bucket_params[index].param_info
            if info.has_explicit_grad_dtype and not new._has_grad_dtype_override:
                new.grad_dtype = info.grad_dtype_override
            local = representation.local_tensor(new)
            if local.is_meta:
                continue
            rows = math.prod(info.local_shape[1:])
            padded = torch.Size(
                (self._slot_numel(info) // rows if rows else 0, *info.local_shape[1:])
            )
            if local.size() not in (info.local_shape, padded):
                raise AssertionError(
                    f"{info.fqn} expects a local tensor of size {info.local_shape} "
                    f"but got {local.size()}"
                )
            if new is not old or (
                index < len(self.locals) and local is not self.locals[index]
            ):
                self.unmaterialized.discard(index)
            if self.own_storage and not _same_storage_slot(local, self.slots[index]):
                self.slots[index] = self._bind(index, None, local, create=False)
            # Otherwise refresh copies a replaced local tensor into the storage
            representation.refresh_param(index, new, info)
        self.params_replaced = True

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
        runtime = self.runtime
        if runtime.is_unsharded:
            return
        dtypes = [info.unsharded_dtype for info in runtime.infos]
        if not self.homed():
            self.home()
        else:
            self._rebuild_infos()
        self.params_replaced = False
        if dtypes != [info.unsharded_dtype for info in runtime.infos]:
            # module.to(dtype): the unsharded parameters and the all-gather's
            # outputs, kept across unshards, are rebuilt in the new dtypes
            runtime.forget_unsharded_params()

    def _rebuild_infos(self) -> None:
        """The parameters' infos from their current data and settings: dtypes
        may have changed (``module.to(dtype)``), and slot offsets with them."""
        runtime = self.runtime
        storage = runtime.bucket_storage
        offsets, _ = self._layout([slot.dtype for slot in self.slots])
        infos: dict[str, ParamInfo] = {}
        new_params: list[BucketParam] = []
        for index, (bp, slot, offset) in enumerate(
            zip(runtime.bucket_params, self.slots, offsets, strict=True)
        ):
            old = bp.param_info
            dtype = slot.dtype
            fields: dict[str, Any] = {
                "dtype": dtype,
                "requires_grad": bp.sharded_param.requires_grad,
                "byte_offset": offset,
                "storage_nbytes": slot.numel() * slot.element_size(),
            }
            policy = cast(list, self.mp_policies)[index]
            requested = getattr(policy, "param_dtype", None)
            # No cast if it is the parameter's dtype or the parameter is
            # not floating point (mixed precision is only meaningful there)
            fields["param_dtype"] = (
                None if requested == dtype or not dtype.is_floating_point else requested
            )
            fields["reduce_dtype"] = getattr(policy, "reduce_dtype", None)
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
                cast("DeviceMesh", self.extension_mesh),
                bp.param_owner.module,
                cast(list, self.mp_policies)[index],
                device,
            )
        if not self.own_storage:
            return None
        data = self.slots[index]
        if device is not None:
            data = data.to(device, non_blocking=True)
        dim = info.placement.dim  # type: ignore[attr-defined]
        length = info.local_shape[dim]
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


def _reshard_before_state_dict(
    homing_ref: weakref.ref[ParamHoming], *args: Any
) -> None:
    if (homing := homing_ref()) is not None and homing.runtime.is_unsharded:
        homing.runtime.reshard()


def _reread_after_load(homing_ref: weakref.ref[ParamHoming], *args: Any) -> None:
    if (homing := homing_ref()) is not None:
        homing.reset_sharded_params()


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


def _starts_at(local: torch.Tensor, slot: torch.Tensor) -> bool:
    """Whether a local tensor still starts at its slot: it may be the
    parameter itself, whose data an op like ``module.to(dtype)`` replaced in
    place. Cheaper than ``_same_storage_slot``, for every explicit unshard. An
    empty local tensor reports data_ptr() 0 even when its slot does not (a
    rank whose shard is empty), so it is compared by storage."""
    if local.numel() == 0:
        return _same_storage_slot(local, slot)
    return (
        local.data_ptr() == slot.data_ptr()
        and local.dtype == slot.dtype
        and local.device == slot.device
    )


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
    """A parameter of a bucket that ``build_bucket`` builds."""

    # The parameter and the module slots that hold it
    param: nn.Parameter
    owner: ParamOwnerRef
    shared_owners: tuple[ParamOwnerRef, ...]
    # Its key in the bucket's infos until a root names it
    fqn: str
    # The tensor whose chunks the bucket stores: the parameter, or for a
    # DTensor parameter its local tensor, whose settings ``meta`` gives
    data: torch.Tensor
    placement: Placement
    meta: ParamMeta | None = None
    # The names of the other module slots, relative to the module
    shared_fqns: tuple[str, ...] = ()


def build_bucket(
    sources: list[ParamSource],
    *,
    mesh: DeviceMesh,
    module: nn.Module,
    device: torch.device,
    context: BucketCommContext,
    representation: ParamRepresentation,
    mp_policy: MixedPrecisionPolicy,
    extension_mesh: DeviceMesh,
    storage_options: dict[str, Any],
    offload_policy: CPUOffloadPolicy | None = None,
    replicate_group: dist.ProcessGroup | None = None,
    post_forward_group: dist.ProcessGroup | None = None,
    debug_fqn: str | None = None,
    logger: logging.Logger | None = None,
    log_name: str = "FlexShard",
    runtime: BucketRuntime | None = None,
) -> BucketRuntime:
    """Build a bucket for ``sources``, for either frontend: its storage, with
    ``storage_options`` (``ShardedBucketStorage``'s settings), and its runtime,
    whose ``ParamHoming`` copies each parameter's shard into the storage and
    registers its sharded parameter (``representation.new_sharded_param``) in
    its module slots. With ``runtime``, the new storage replaces that bucket's
    (``relayout_bucket``), keeping its hooks and context."""
    from .bucket_runtime import BucketRuntime, logger as default_logger
    from .bucket_storage import ShardedBucketStorage

    infos, total_bytes = ShardedBucketStorage.create_param_infos(
        # The data tensors stand for their parameters, whose settings the
        # metas carry when they differ
        cast(list[tuple[str, nn.Parameter]], [(src.fqn, src.data) for src in sources]),
        mesh,
        {src.fqn: (src.placement,) for src in sources},
        mp_policy,
        {src.fqn: src.meta for src in sources if src.meta is not None} or None,
    )
    for src in sources:
        infos[src.fqn].shared_fqns = src.shared_fqns
    by_fqn = {src.fqn: src for src in sources}
    sources = [by_fqn[fqn] for fqn in infos]
    storage = ShardedBucketStorage(
        torch.empty(0, dtype=torch.uint8, device=device),
        infos,
        mesh,
        total_bytes,
        module,
        **storage_options,
    )
    storage._post_forward_group = post_forward_group
    if replicate_group is not None:
        storage.set_replicate_group(replicate_group)
    if offload_policy is not None:
        storage.set_cpu_offload(offload_policy, device)
    bucket_params = [
        BucketParam(src.owner, infos[src.fqn], src.param, src.shared_owners)
        for src in sources
    ]
    if runtime is None:
        runtime = BucketRuntime(
            bucket_storage=storage,
            bucket_params=bucket_params,
            context=context,
            debug_fqn=debug_fqn,
            representation=representation,
            log_name=log_name,
            logger=logger or default_logger,
        )
        context.buckets.append(runtime)
        first_install = True
    else:
        runtime.rebind_storage(storage, bucket_params)
        first_install = False
    # Copies each parameter's shard into the storage, so that the parameters'
    # data can be freed once this returns
    runtime.homing.install(
        [src.data for src in sources],
        device=device,
        extension_mesh=extension_mesh,
        mp_policies=[mp_policy._resolve_for_param(src.param) for src in sources],
    )
    runtime.homing.init_extensions()
    if first_install:
        runtime.homing.install_state_dict_hooks()
    return runtime


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
