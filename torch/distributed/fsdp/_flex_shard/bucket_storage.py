# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for

from .placement_contract import (
    get_global_layout,
    get_partial_grad_group,
    GradientReduceOp,
    GradientReduction,
)
from .sharded_param import set_sharding_info
from .utils import _get_single_placement, _set_param_on_module


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_runtime import BucketCommContext, BucketRuntime
    from .placement_contract import (
        BucketStorageLayout,
        GlobalLayout,
        LocalStorageLayout,
        Placement,
    )


BucketParamFQNsByIndex = list[list[str]]

PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], "DeviceMesh"],
    dict[str, tuple["Placement", ...]],
]

# Called with a bucket's ``(fqn, unsharded param)`` pairs.
BucketHook = Callable[[list[tuple[str, nn.Parameter]]], None]


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Mixed precision policy for FlexShard buckets.

    Args:
        param_dtype: Dtype for forward compute. Placements should materialize
            unsharded parameters in this dtype. If None, use storage dtype.
        reduce_dtype: Dtype for autograd accumulation and gradient
            reduction. Eager casts each full-parameter gradient to this dtype
            before it accumulates on the unsharded parameter; torch.compile
            casts the accumulated gradient in the reduce-scatter copy-in. If
            None, each parameter accumulates in its ``grad_dtype`` (its
            dtype unless explicitly set), independent of param_dtype, and
            each bucket reduces in the promoted accumulation dtype of its
            parameters trainable at wrap time, widened in each backward by
            those trainable then. A parameter with ``grad_dtype`` explicitly
            None accumulates in whatever dtype its gradients arrive in, which
            can widen its bucket's reduce dtype. Every rank must produce the
            same gradient dtypes, or ranks reduce in different dtypes and the
            collective can hang. Where the parameter is unused, its zero
            gradient has the forward dtype (param_dtype, else its dtype); set
            reduce_dtype if its gradients can arrive in another dtype. This only sets the
            accumulation and communication dtype: the reduced sharded
            gradient is then cast to the parameter's ``grad_dtype`` when
            stored in ``.grad``, and ``grad_dtype=None`` skips that cast.
            flex_shard() keeps an explicit ``grad_dtype`` setting.
        param_dtype_overrides: Sparse mapping from root-relative parameter FQN
            to a parameter dtype override. Parameters absent from the mapping
            use param_dtype. Override values must be param_dtype or the
            parameter's original dtype. Every key must name a parameter assigned
            to a bucket that uses this policy (or an equal one).
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    param_dtype_overrides: Mapping[str, torch.dtype] | None = field(
        default=None,
        kw_only=True,
    )

    def _resolve_param_dtype(
        self,
        fqn: str,
        param: nn.Parameter,
    ) -> torch.dtype | None:
        if self.param_dtype_overrides is None or fqn not in self.param_dtype_overrides:
            return self.param_dtype
        param_dtype = self.param_dtype_overrides[fqn]
        if not isinstance(param_dtype, torch.dtype):
            raise ValueError(
                "MixedPrecisionPolicy.param_dtype_overrides values must be "
                f"torch.dtype, but {fqn!r} maps to {type(param_dtype)}."
            )
        if param_dtype not in (self.param_dtype, param.dtype):
            raise ValueError(
                "MixedPrecisionPolicy.param_dtype_overrides values must be "
                "param_dtype or the parameter's original dtype, but "
                f"{fqn!r} maps to {param_dtype} for a parameter with dtype "
                f"{param.dtype} and param_dtype {self.param_dtype}."
            )
        return param_dtype

    def _group_by_unsharded_dtype(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> list[tuple[str, nn.Parameter]]:
        if not self.param_dtype_overrides:
            return named_params
        groups: dict[torch.dtype, list[tuple[str, nn.Parameter]]] = {}
        for fqn, param in named_params:
            dtype = self._resolve_param_dtype(fqn, param) or param.dtype
            groups.setdefault(dtype, []).append((fqn, param))
        return [named_param for group in groups.values() for named_param in group]


@dataclass(frozen=True)
class OffloadPolicy:
    """CPU offload policy for FlexShard buckets.

    This is a placeholder for future CPU offload support. The minimal eager
    FlexShard path currently rejects non-None offload policies before storage
    materialization.

    Args:
        pin_memory: Whether to pin CPU memory for faster H2D/D2H
            transfers via DMA. Set to False if insufficient CPU memory.
            Default True.
    """

    pin_memory: bool = True


@dataclass(frozen=True)
class BucketSpec:
    """Specification for a parameter communication bucket.

    Args:
        patterns: fnmatch glob patterns matched against parameter FQNs, or
            module FQNs, which match all of the module's parameters. A
            parameter matches this bucket if its FQN matches any pattern. A
            parameter registered under several names (e.g. tied embedding and
            output weights) must match one bucket with every name, as FSDP2
            requires one FSDP group per parameter; the bucket's hooks then sit
            on a module that contains every use: the deepest module containing
            its params. A bucket whose patterns all name modules hooks those
            modules instead, as one group, as FSDP2's ``fully_shard`` does for a
            list of modules: it unshards when the first of them starts its
            forward, runs its post-forward once all of them have finished, and
            reduce-scatters once the first one's input grads are computed. For
            a bucket of sibling modules, e.g. consecutive transformer layers,
            whose parent would gather it for its whole forward and
            reduce-scatter it only after its whole backward. As with FSDP2, a
            forward may skip some of them, and the root module's post-forward
            then completes the group, e.g. a decoder that returns its norm's
            output and leaves the output projection in its bucket to the
            loss. One of them may also run on its own, e.g. that projection
            on each chunk of the hidden states with a backward per chunk:
            such a call does not complete the group, so its outputs get no
            pre-backward hook, and its post-backward runs as soon as its input
            grads are computed. Eager only.
        placement_fn: Required callable that maps this bucket's
            ``(named_params, mesh)`` to per-parameter placements.
            The minimal eager path expects one ``Placement`` per parameter.
        mesh: The 1D CUDA ``DeviceMesh`` this bucket's collective runs on. A
            bucket is one collective over one process group, so the mesh is a
            per-bucket property; ``placement_fn`` receives it. Different buckets
            may use different meshes (e.g. expert params on an expert-parallel
            mesh vs dense params on the data-parallel mesh), but all bucket
            meshes in one ``flex_shard()`` call must share a device type.
        mp_policy: Mixed precision policy for this bucket. This currently
            covers parameter and gradient-reduction dtypes only.
            TODO: add module-boundary input/output casting separately from
            BucketSpec. FSDP2 exposes cast_forward_inputs and output_dtype, but
            those apply to a module's forward args/outputs, not to an individual
            parameter bucket. Keeping them out of BucketSpec avoids ambiguous
            behavior when multiple buckets share the same hooked module.
        offload_policy: CPU offload policy for this bucket. TODO: implement
            and test CPU offload before allowing this in flex_shard().
        gradient_reduce_op: Gradient reduction semantics. ``dist.ReduceOp.AVG``
            preserves FlexShard's historical average-gradient behavior.
            ``dist.ReduceOp.SUM`` matches FSDP2's no-gradient-division mode,
            where the training loop owns global gradient scaling.
        gradient_divide_factor: With ``gradient_reduce_op=AVG``, the reduced
            gradient is the sum over ``mesh`` divided by this factor, which
            defaults to the mesh size; like FSDP2's
            ``set_gradient_divide_factor``. For example, with expert parallelism
            an expert's gradient already includes tokens routed from its
            expert-parallel peers, so an expert bucket on the expert
            data-parallel mesh divides by the dense data-parallel size. The
            placement implements both settings: ``reduce_prepared_grad``
            receives them as a ``GradientReduction``.
        reshard_after_forward: Whether to free this bucket's unsharded
            parameters after forward and re-gather them before backward. This
            defaults to True. Under ``torch.compile`` the traced graph owns
            buffer lifetimes instead.
        pre_backward_hook: Optional callable run with this bucket's
            ``(fqn, unsharded param)`` pairs once they are unsharded and before
            its backward runs: in its pre-backward hook, and when activation
            checkpointing unshards it for the recomputed forward; it may run
            more than once per backward. For example, kernels that add weight grads
            into a buffer in place and give autograd none (TransformerEngine's
            ``fuse_wgrad_accumulation`` reads ``param.main_grad``) need the
            grads allocated and exposed there first. Grads allocated as views
            of one zeroed buffer per storage the unsharded params view, at
            their offsets, in the reduce dtype, let a placement whose gradient
            reduction layout is its parameter layout (``BucketedBlockShard``
            with equal rank ranges, whole-parameter ``BucketedOwned``)
            reduce-scatter that buffer without copying the grads in
            (``alloc_grads_in_param_layout``). A whole-parameter
            ``BucketedOwned`` bucket with no hook gets its grads allocated so
            when its params are in the dtype their grads accumulate and reduce
            in (``Placement.reduces_grads_in_param_layout``). Eager only.
        post_reduce_hook: Optional callable run with the same pairs once a
            syncing backward has taken their grads for the reduce-scatter, e.g.
            to release references to those grads. Eager only.
        defer_post_backward: Whether this bucket's post-backward (the
            reduce-scatter with gradient sync on, then reshard) waits for
            ``FlexShardModule.finish_deferred_backward`` instead of running once
            its module's backward is done. For kernels that compute weight
            grads after that backward, e.g. TransformerEngine's
            ``delay_wgrad_compute``, whose ``backward_dw()`` runs later. A
            syncing backward that runs the module's backward but ends with the
            bucket unfinished raises. Eager only.
        fsdp2_compatible: Whether this bucket's collectives match those of the
            FSDP2 ``fully_shard`` group it replaces, byte for byte, so training
            is bitwise identical to FSDP2. The parameters take FSDP2's order: a
            post-order walk over the modules the patterns name, in pattern
            order, or over the whole module for parameter patterns, keeping
            each module's own parameters. The reduce-scatter groups them by
            sharded grad dtype, as FSDP2 does, and includes only the parameters
            that got a grad, as FSDP2 does by default: every rank must then get
            grads for the same parameters. Requires ``Shard`` placements. Eager
            only.
    """

    patterns: list[str]
    placement_fn: PlacementFn
    mesh: DeviceMesh
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: OffloadPolicy | None = None
    gradient_reduce_op: GradientReduceOp = dist.ReduceOp.AVG
    gradient_divide_factor: float | None = None
    reshard_after_forward: bool = True
    pre_backward_hook: BucketHook | None = None
    post_reduce_hook: BucketHook | None = None
    defer_post_backward: bool = False
    fsdp2_compatible: bool = False


@dataclass(frozen=True)
class BucketParamLayout:
    """Bucket-global layout metadata for one parameter."""

    param_offset: int
    local_global_offset: int


@dataclass(frozen=True)
class BucketLayout:
    """Bucket-global storage layout shared by parameters in one bucket."""

    global_numel: int
    local_numel: int
    rank_offsets: tuple[int, ...]
    rank_numels: tuple[int, ...]
    param_layouts: dict[str, BucketParamLayout]


@dataclass
class ParamInfo:
    """Metadata for a parameter in chunked storage."""

    fqn: str
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    placements: tuple[Placement, ...]
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    local_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    local_numel: int = 0
    byte_offset: int = 0  # byte offset into the sharded storage
    storage_nbytes: int = 0  # bytes reserved for this param's local storage
    global_numel: int = 0  # total elements in unsharded param
    bucket_layout: BucketLayout | None = None
    # Outer (TP/EP) layout declared by the input param; ``global_shape`` is then
    # the outer local shape.
    outer_layout: GlobalLayout | None = None
    # Group over which each rank computes only part of the grad; FlexShard sums
    # the unsharded grad over it before the reduce-scatter.
    partial_grad_group: dist.ProcessGroup | None = None
    # Python attributes of the original parameter (e.g. framework tags), copied
    # onto its persistent unsharded parameter.
    param_attrs: dict[str, Any] = field(default_factory=dict)
    has_explicit_grad_dtype: bool = False
    grad_dtype_override: torch.dtype | None = None
    # Other names the parameter is registered under (e.g. tied weights), all in
    # this bucket; each module slot holds the same sharded or unsharded param.
    shared_fqns: tuple[str, ...] = ()
    bucket_reduce_dtype: torch.dtype | None = None  # promoted over the bucket

    @property
    def placement(self) -> Placement:
        """The single placement supported by the minimal eager path."""
        return _get_single_placement(self.placements)

    @property
    def unsharded_dtype(self) -> torch.dtype:
        """Dtype exposed to module forward for the full parameter."""
        return self.param_dtype or self.dtype

    @property
    def unsharded_grad_dtype(self) -> torch.dtype | None:
        """Dtype for autograd accumulation of the full-parameter gradient.

        None accepts gradients in any dtype.
        """
        if self.reduce_dtype is not None:
            return self.reduce_dtype
        if self.has_explicit_grad_dtype:
            return self.grad_dtype_override
        return self.dtype

    @property
    def grad_reduce_dtype(self) -> torch.dtype:
        """Dtype used to communicate this parameter's gradient."""
        return (
            self.bucket_reduce_dtype
            or self.unsharded_grad_dtype
            or self.unsharded_dtype
        )


class ShardedBucketStorage:
    """
    Manages a byte buffer that backs one bucket of sharded parameters.

    All parameters in a bucket storage must share a dtype and placement-compatible
    local layout. Each placement owns its parameter's local storage layout and
    exposed tensor view; ShardedBucketStorage only places those layouts
    sequentially in one byte buffer.

    Communication is delegated to the bucket runtime's forward hooks; this
    bucket storage object owns the byte buffer and metadata.
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
        module: nn.Module,
        reshard_after_forward: bool = True,
        gradient_reduce_op: GradientReduceOp = dist.ReduceOp.AVG,
        pre_backward_hook: BucketHook | None = None,
        post_reduce_hook: BucketHook | None = None,
        gradient_divide_factor: float | None = None,
        defer_post_backward: bool = False,
        hook_module_fqns: list[str] | None = None,
        fsdp2_compatible: bool = False,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._module = module
        self._reshard_after_forward = reshard_after_forward
        self._pre_backward_hook = pre_backward_hook
        self._post_reduce_hook = post_reduce_hook
        self._defer_post_backward = defer_post_backward
        self._fsdp2_compatible = fsdp2_compatible
        # The modules a bucket of module patterns hooks (BucketSpec.patterns).
        self._hook_module_fqns = hook_module_fqns
        # See set_requires_gradient_sync and set_reshard_after_backward.
        self._requires_gradient_sync = True
        self._reshard_after_backward = True
        # See set_separate_reduce_scatter_group.
        self._reduce_mesh: DeviceMesh | None = None
        # See set_buckets_to_forward_prefetch and set_buckets_to_backward_prefetch.
        self._forward_prefetch: list[ShardedBucketStorage] | None = None
        self._backward_prefetch: list[ShardedBucketStorage] | None = None
        self._gradient_reduction = GradientReduction(
            gradient_reduce_op, gradient_divide_factor
        )

    @classmethod
    def from_bucket(
        cls,
        module: nn.Module,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
        device: torch.device,
        bucket_spec: BucketSpec,
        shared_names: dict[str, list[str]] | None = None,
    ) -> ShardedBucketStorage:
        """Create storage metadata for one bucket and install sharded params."""
        param_infos, total_bytes = cls.create_param_infos(
            named_params,
            mesh,
            param_placements,
            bucket_spec.mp_policy,
        )
        for fqn, other_fqns in (shared_names or {}).items():
            if fqn in param_infos:
                param_infos[fqn].shared_fqns = tuple(other_fqns)
        if bucket_spec.offload_policy is not None:
            byte_storage = torch.empty(
                total_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=bucket_spec.offload_policy.pin_memory,
            )
            expected_param_device = torch.device("cpu")
        else:
            byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            expected_param_device = torch.device(device)

        bucket_storage = cls(
            byte_storage,
            param_infos,
            mesh,
            total_bytes,
            module,
            reshard_after_forward=bucket_spec.reshard_after_forward,
            gradient_reduce_op=bucket_spec.gradient_reduce_op,
            pre_backward_hook=bucket_spec.pre_backward_hook,
            post_reduce_hook=bucket_spec.post_reduce_hook,
            gradient_divide_factor=bucket_spec.gradient_divide_factor,
            defer_post_backward=bucket_spec.defer_post_backward,
            hook_module_fqns=_hook_module_fqns(module, bucket_spec.patterns),
            fsdp2_compatible=bucket_spec.fsdp2_compatible,
        )
        bucket_storage.copy_params_from(named_params)
        bucket_storage.install_sharded_params(expected_param_device)
        return bucket_storage

    @classmethod
    def create_param_infos(
        cls,
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
        param_placements: dict[str, tuple[Placement, ...]],
        mp_policy: MixedPrecisionPolicy | None = None,
    ) -> tuple[dict[str, ParamInfo], int]:
        """
        Create ParamInfo for each parameter, computing local layout and byte offsets.

        The caller validates that each bucket uses compatible placements and a
        uniform dtype. Each placement owns its per-parameter local storage layout;
        bucket storage only places those layouts sequentially in the byte buffer.
        """
        if not named_params:
            return {}, 0

        # Mixed buckets alias one contiguous storage span per (placement, dtype)
        # subgroup, so equal unsharded dtypes must be adjacent in the layout.
        # ParamInfo order must match layout order: FP8 pack plans index infos
        # by position.
        if mp_policy is not None:
            named_params = mp_policy._group_by_unsharded_dtype(named_params)
        first_fqn = named_params[0][0]
        placement = _get_single_placement(param_placements[first_fqn])
        bucket_layout = placement.bucket_storage_layout(
            named_params,
            param_placements,
            mesh,
        )
        if bucket_layout is not None:
            param_infos, total_bytes = cls._create_param_infos_from_bucket_layout(
                named_params,
                param_placements,
                bucket_layout,
                mp_policy,
            )
        else:
            param_infos, total_bytes = cls._create_param_infos_from_local_layouts(
                named_params,
                mesh,
                param_placements,
                mp_policy,
            )

        # One collective per bucket needs one dtype. Like FSDP2, promote over
        # trainable grads only. Grads with no fixed dtype normally arrive in
        # the unsharded dtype; backward only widens this over the grads that
        # actually arrive.
        trainable_grad_dtypes = [
            info.unsharded_grad_dtype or info.unsharded_dtype
            for info in param_infos.values()
            if info.requires_grad
        ]
        if trainable_grad_dtypes:
            bucket_reduce_dtype = functools.reduce(
                torch.promote_types,
                trainable_grad_dtypes,
            )
            for info in param_infos.values():
                info.bucket_reduce_dtype = bucket_reduce_dtype
        return param_infos, total_bytes

    @classmethod
    def _create_param_infos_from_local_layouts(
        cls,
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
        param_placements: dict[str, tuple[Placement, ...]],
        mp_policy: MixedPrecisionPolicy | None,
    ) -> tuple[dict[str, ParamInfo], int]:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        param_infos: dict[str, ParamInfo] = {}
        current_byte_offset = 0
        for fqn, param in named_params:
            placements = param_placements[fqn]
            placement = _get_single_placement(placements)
            local_storage_layout = cls._compute_local_storage_layout(
                fqn,
                param,
                placement,
                rank,
                world_size,
            )
            if local_storage_layout.storage_nbytes > 0:
                byte_offset = current_byte_offset
                current_byte_offset += local_storage_layout.storage_nbytes
            else:
                byte_offset = 0

            param_infos[fqn] = cls._create_param_info(
                fqn=fqn,
                param=param,
                placements=placements,
                local_shape=local_storage_layout.local_shape,
                local_numel=local_storage_layout.local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_storage_layout.storage_nbytes,
                mp_policy=mp_policy,
            )

        return param_infos, current_byte_offset

    @staticmethod
    def _compute_local_storage_layout(
        fqn: str,
        param: nn.Parameter,
        placement: Placement,
        rank: int,
        world_size: int,
    ) -> LocalStorageLayout:
        local_storage_layout = placement.local_storage_layout(
            param.shape,
            param.dtype,
            rank,
            world_size,
        )

        if local_storage_layout.local_numel < 0:
            raise ValueError(
                f"Placement {placement!r} returned negative local_numel "
                f"for parameter {fqn!r}."
            )
        if local_storage_layout.storage_nbytes < 0:
            raise ValueError(
                f"Placement {placement!r} returned negative storage_nbytes "
                f"for parameter {fqn!r}."
            )
        return local_storage_layout

    @classmethod
    def _create_param_infos_from_bucket_layout(
        cls,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        bucket_layout: BucketStorageLayout,
        mp_policy: MixedPrecisionPolicy | None,
    ) -> tuple[dict[str, ParamInfo], int]:
        expected_fqns = {fqn for fqn, _ in named_params}
        actual_fqns = set(bucket_layout.param_layouts)
        if actual_fqns != expected_fqns:
            msg_parts = []
            missing_fqns = expected_fqns - actual_fqns
            extra_fqns = actual_fqns - expected_fqns
            if missing_fqns:
                msg_parts.append(f"missing layouts for {sorted(missing_fqns)}")
            if extra_fqns:
                msg_parts.append(f"unexpected layouts for {sorted(extra_fqns)}")
            raise ValueError(
                "Placement bucket_storage_layout() must return layouts for "
                f"exactly the provided parameters; {', '.join(msg_parts)}."
            )
        if bucket_layout.total_bytes < 0:
            raise ValueError(
                "Placement bucket_storage_layout() returned negative total_bytes."
            )

        param_infos: dict[str, ParamInfo] = {}
        for fqn, param in named_params:
            placements = param_placements[fqn]
            layout = bucket_layout.param_layouts[fqn]
            if layout.local_numel < 0:
                raise ValueError(
                    "Placement bucket_storage_layout() returned negative "
                    f"local_numel for parameter {fqn!r}."
                )
            if layout.byte_offset < 0:
                raise ValueError(
                    "Placement bucket_storage_layout() returned negative "
                    f"byte_offset for parameter {fqn!r}."
                )
            if layout.storage_nbytes < 0:
                raise ValueError(
                    "Placement bucket_storage_layout() returned negative "
                    f"storage_nbytes for parameter {fqn!r}."
                )
            if (
                layout.storage_nbytes > 0
                and layout.byte_offset + layout.storage_nbytes
                > bucket_layout.total_bytes
            ):
                raise ValueError(
                    "Placement bucket_storage_layout() returned storage for "
                    f"parameter {fqn!r} outside the bucket byte range."
                )

            param_infos[fqn] = cls._create_param_info(
                fqn=fqn,
                param=param,
                placements=placements,
                local_shape=layout.local_shape,
                local_numel=layout.local_numel,
                byte_offset=layout.byte_offset,
                storage_nbytes=layout.storage_nbytes,
                bucket_layout=layout.bucket_layout,
                mp_policy=mp_policy,
            )

        return param_infos, bucket_layout.total_bytes

    @staticmethod
    def _create_param_info(
        *,
        fqn: str,
        param: nn.Parameter,
        placements: tuple[Placement, ...],
        local_shape: torch.Size,
        local_numel: int,
        byte_offset: int,
        storage_nbytes: int,
        bucket_layout: BucketLayout | None = None,
        mp_policy: MixedPrecisionPolicy | None = None,
    ) -> ParamInfo:
        has_explicit_grad_dtype = param._has_grad_dtype_override
        return ParamInfo(
            fqn=fqn,
            global_shape=param.shape,
            global_stride=tuple(make_contiguous_strides_for(param.shape)),
            dtype=param.dtype,
            param_dtype=(
                mp_policy._resolve_param_dtype(fqn, param)
                if mp_policy is not None
                else None
            ),
            reduce_dtype=mp_policy.reduce_dtype if mp_policy is not None else None,
            requires_grad=param.requires_grad,
            placements=placements,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            storage_nbytes=storage_nbytes,
            global_numel=param.numel(),
            bucket_layout=bucket_layout,
            outer_layout=get_global_layout(param),
            partial_grad_group=get_partial_grad_group(param),
            param_attrs=dict(vars(param)),
            has_explicit_grad_dtype=has_explicit_grad_dtype,
            grad_dtype_override=param.grad_dtype if has_explicit_grad_dtype else None,
        )

    def copy_params_from(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> None:
        """Pack original parameter data into byte storage."""
        my_rank = self._mesh.get_local_rank()
        world_size = self._mesh.size()

        for fqn, param in named_params:
            info = self._param_infos[fqn]
            info.placement.copy_param_to_storage(
                self._byte_storage,
                info,
                param,
                my_rank,
                world_size,
            )

    def install_sharded_params(
        self,
        expected_device: torch.device,
    ) -> None:
        """Replace original module parameters with local storage views."""
        for fqn, info in self._param_infos.items():
            typed_view = info.placement.make_local_storage_view(
                self._byte_storage,
                info,
            )
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            if info.has_explicit_grad_dtype:
                new_param.grad_dtype = info.grad_dtype_override
            if new_param.device != expected_device:
                raise AssertionError(
                    f"Expected sharded parameter {fqn!r} on "
                    f"{expected_device}, but got {new_param.device}"
                )
            set_sharding_info(
                new_param,
                placements=info.placements,
                global_shape=info.global_shape,
                global_stride=info.global_stride,
                mesh=self._mesh,
                outer_layout=info.outer_layout,
            )
            for name in (fqn, *info.shared_fqns):
                _set_param_on_module(self._module, name, new_param)

    @property
    def byte_storage(self) -> torch.Tensor:
        """The underlying unified byte storage tensor (sharded)."""
        return self._byte_storage

    @property
    def flat_storage(self) -> torch.Tensor:
        """Alias for byte_storage for backwards compatibility."""
        return self._byte_storage

    @property
    def total_bytes(self) -> int:
        """Total bytes in the sharded storage."""
        return self._total_bytes

    @property
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

    @property
    def gradient_reduction(self) -> GradientReduction:
        """What this bucket's gradient reduction must produce, which its
        placement implements."""
        return self._gradient_reduction

    @property
    def gradient_reduce_op(self) -> GradientReduceOp:
        """Gradient reduction semantics for this bucket."""
        return self._gradient_reduction.op

    def set_gradient_reduce_op(self, op: GradientReduceOp) -> None:
        """Set gradient reduction semantics for this bucket."""
        self._gradient_reduction = replace(self._gradient_reduction, op=op)

    @property
    def gradient_divide_factor(self) -> float | None:
        """What ``AVG`` divides this bucket's summed gradients by; None means the
        mesh size."""
        return self._gradient_reduction.divide_factor

    def set_gradient_divide_factor(self, factor: float | None) -> None:
        """Set what ``AVG`` divides this bucket's summed gradients by."""
        self._gradient_reduction = replace(
            self._gradient_reduction, divide_factor=factor
        )

    @property
    def reduce_mesh(self) -> DeviceMesh:
        """The mesh this bucket reduce-scatters its gradients over: its mesh,
        unless ``set_separate_reduce_scatter_group`` gave the reduce-scatters a
        process group of their own over the same ranks."""
        return self._mesh if self._reduce_mesh is None else self._reduce_mesh

    def set_separate_reduce_scatter_group(
        self,
        enable: bool,
        reduce_meshes: dict[tuple[str, tuple[int, ...]], DeviceMesh] | None = None,
    ) -> None:
        """Run this bucket's reduce-scatters on a process group of their own,
        over its mesh's ranks, or with ``enable=False`` on its mesh's group.

        ``FlexShardModule.set_separate_reduce_scatter_group`` documents the
        behavior and sets every bucket. ``reduce_meshes`` holds the reduce
        meshes made so far, by device type and ranks, so buckets over the same
        ranks share one process group.
        """
        if not enable or self._mesh.size() == 1:
            # A 1-rank mesh has no reduce-scatter to separate.
            self._reduce_mesh = None
            return
        if self._mesh.ndim != 1:
            raise NotImplementedError(
                "A separate reduce-scatter group needs a 1-D bucket mesh, got "
                f"{self._mesh.ndim} dims."
            )
        from torch.distributed.device_mesh import DeviceMesh

        ranks = tuple(self._mesh.mesh.flatten().tolist())
        key = (self._mesh.device_type, ranks)
        if reduce_meshes is None:
            reduce_meshes = {}
        if key not in reduce_meshes:
            if self._reduce_mesh is not None:
                # Already enabled: keep its group.
                reduce_meshes[key] = self._reduce_mesh
            else:
                # Local synchronization: ranks outside this mesh don't wait in
                # new_group's barrier, as for FSDP2's group.
                group = dist.new_group(
                    list(ranks),
                    use_local_synchronization=True,
                    group_desc="flex_shard_reduce_scatter",
                )
                if group == dist.GroupMember.NON_GROUP_MEMBER:
                    raise AssertionError(
                        f"This rank is not in its bucket mesh's ranks {ranks}."
                    )
                reduce_meshes[key] = DeviceMesh.from_group(
                    group,
                    self._mesh.device_type,
                    mesh=self._mesh.mesh,
                    mesh_dim_names=self._mesh.mesh_dim_names,
                )
        self._reduce_mesh = reduce_meshes[key]

    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        """Set whether backward reduce-scatters this bucket's gradients.

        ``FlexShardModule.set_requires_gradient_sync`` documents the behavior
        and sets every bucket.
        """
        self._requires_gradient_sync = requires_gradient_sync

    def set_reshard_after_backward(self, reshard_after_backward: bool) -> None:
        """Set whether a backward without gradient sync reshards this bucket.

        ``FlexShardModule.set_reshard_after_backward`` documents the behavior
        and sets every bucket.
        """
        self._reshard_after_backward = reshard_after_backward

    @property
    def reshard_after_backward(self) -> bool:
        """Whether a backward without gradient sync reshards this bucket; see
        ``set_reshard_after_backward``."""
        return self._reshard_after_backward

    @property
    def reshard_after_forward(self) -> bool:
        """Whether this bucket frees its unsharded parameters after forward;
        see ``set_reshard_after_forward``."""
        return self._reshard_after_forward

    def set_reshard_after_forward(self, reshard_after_forward: bool) -> None:
        """Set whether this bucket reshards after forward, like FSDP2's
        ``set_reshard_after_forward`` on its group.

        ``BucketSpec.reshard_after_forward`` sets the initial value; this one
        applies from the next forward on, e.g. to keep the bucket gathered
        across several calls of its module.
        """
        self._reshard_after_forward = reshard_after_forward

    def unshard(self) -> None:
        """Unshard this bucket now and keep it unsharded, like FSDP2's
        ``FSDPModule.unshard()`` on its group, synchronously.

        ``FlexShardModule.unshard`` documents the behavior and unshards every
        bucket. Eager only.
        """
        context, bucket = self._runtime()
        context.check_outside_backward("unshard")
        bucket.unshard()

    def reshard(self) -> None:
        """Reshard this bucket now, like FSDP2's ``FSDPModule.reshard()`` on
        its group, and release its prefetched unshard, if any.

        Grads accumulated without sync stay. Unlike
        ``FlexShardModule.reshard``, this keeps the bucket's backward state, so
        a backward still to run re-gathers the bucket in its pre-backward hook.
        Eager only.
        """
        context, bucket = self._runtime()
        context.check_outside_backward("reshard")
        context.release_pending_unshard(bucket)
        if bucket.is_unsharded:
            bucket.reshard()

    def _runtime(self) -> tuple[BucketCommContext, BucketRuntime]:
        """This bucket's communication context and runtime, which exist once
        its storage is materialized."""
        from .bucket_runtime import BucketCommContext

        context = BucketCommContext.get(self._module, self.byte_storage.device)
        if context is None:
            raise RuntimeError(
                "FlexShard: the bucket has no runtime until its storage is "
                "materialized, e.g. by to_empty()."
            )
        return context, context.bucket_runtime(self)

    def set_buckets_to_forward_prefetch(
        self, buckets: list[ShardedBucketStorage]
    ) -> None:
        """Set the buckets this bucket prefetches in forward, like FSDP2's
        ``set_modules_to_forward_prefetch``.

        Right after this bucket unshards in its pre-forward hook, the listed
        buckets start their all-gathers, in order, unless they are unsharded
        already. Each stays in flight until its own hook takes it, or the
        backward or ``FlexShardModule.reshard()`` releases it, so listing more
        buckets overlaps more communication and holds more unsharded buckets
        at once. Without a list, a bucket prefetches the next bucket in the
        learned forward order, one prefetch at a time; a list, even an empty
        one, replaces that. The buckets must come from the same ``flex_shard``
        call. Eager only.
        """
        self._forward_prefetch = _check_prefetch_buckets(buckets)

    def set_buckets_to_backward_prefetch(
        self, buckets: list[ShardedBucketStorage]
    ) -> None:
        """Set the buckets this bucket prefetches in backward, like FSDP2's
        ``set_modules_to_backward_prefetch``.

        Right after this bucket re-gathers in its pre-backward hook, the listed
        buckets start their all-gathers, as ``set_buckets_to_forward_prefetch``
        describes. Without a list, a bucket prefetches the next bucket in the
        reverse post-forward order; a list, even an empty one, replaces that.
        """
        self._backward_prefetch = _check_prefetch_buckets(buckets)

    @property
    def world_size(self) -> int:
        """World size of the mesh."""
        return self._mesh.size()

    def get_local_view(self, fqn: str) -> torch.Tensor:
        """Get the local tensor view for a parameter by FQN (from sharded storage)."""
        info = self._param_infos[fqn]
        return info.placement.make_local_storage_view(self._byte_storage, info)


def _check_prefetch_buckets(
    buckets: list[ShardedBucketStorage],
) -> list[ShardedBucketStorage]:
    buckets = list(buckets)
    for bucket in buckets:
        if not isinstance(bucket, ShardedBucketStorage):
            raise TypeError(
                "Prefetch targets must be entries of "
                f"FlexShardModule.sharded_bucket_storages, got {type(bucket).__name__}."
            )
    return buckets


def _assign_params_to_buckets(
    param_fqns: list[str],
    buckets: list[BucketSpec],
    shared_names: dict[str, list[str]] | None = None,
) -> BucketParamFQNsByIndex:
    """Assign each param FQN to exactly one bucket via fnmatch.

    Returns:
        A list aligned with ``buckets``: result[bucket_idx] is the ordered list
        of parameter FQNs assigned to ``buckets[bucket_idx]``.

    Raises:
        ValueError: if any param matches zero or multiple buckets, or if the
            names of a shared parameter (``shared_names``: first name -> other
            names) match different buckets.
    """
    param_to_buckets: dict[str, list[int]] = {
        fqn: _matching_buckets(fqn, buckets) for fqn in param_fqns
    }

    # Check for orphans
    orphans = [fqn for fqn, idxs in param_to_buckets.items() if len(idxs) == 0]
    if orphans:
        orphan_list = "\n  ".join(orphans)
        raise ValueError(
            f"flex_shard: {len(orphans)} parameters not covered by any bucket:\n"
            f"  {orphan_list}\n"
            'Add these to an existing bucket or add a catch-all bucket: ["*"]'
        )

    # Check for overlaps
    overlaps = {fqn: idxs for fqn, idxs in param_to_buckets.items() if len(idxs) > 1}
    if overlaps:
        lines = []
        for fqn, idxs in overlaps.items():
            bucket_descs = ", ".join(f"bucket {i} {buckets[i].patterns}" for i in idxs)
            lines.append(f"  {fqn} -> {bucket_descs}")
        overlap_list = "\n".join(lines)
        raise ValueError(
            f"flex_shard: {len(overlaps)} parameters matched multiple buckets:\n"
            f"{overlap_list}\n"
            "Ensure each parameter matches exactly one bucket."
        )

    # Like FSDP2, which requires one FSDP group per parameter, every name of a
    # shared parameter must match its bucket: the bucket's hooks then sit on a
    # module that contains every use of the parameter.
    for fqn, other_fqns in (shared_names or {}).items():
        names = {name: _matching_buckets(name, buckets) for name in (fqn, *other_fqns)}
        if any(idxs != param_to_buckets[fqn] for idxs in names.values()):
            lines = []
            for name, idxs in names.items():
                descs = ", ".join(f"bucket {i} {buckets[i].patterns}" for i in idxs)
                lines.append(f"  {name} -> {descs or 'no bucket'}")
            matches = "\n".join(lines)
            raise ValueError(
                f"flex_shard: parameter {fqn!r} is shared with "
                f"{', '.join(map(repr, other_fqns))}, but its names match "
                f"different buckets:\n{matches}\n"
                "For shared/tied parameters, put every name that refers to the "
                "parameter in the same BucketSpec."
            )

    # Build assignments
    assignments: BucketParamFQNsByIndex = [[] for _ in buckets]
    for fqn, idxs in param_to_buckets.items():
        assignments[idxs[0]].append(fqn)

    return assignments


def _matching_buckets(fqn: str, buckets: list[BucketSpec]) -> list[int]:
    # A pattern naming a module matches the params under it.
    return [
        bucket_idx
        for bucket_idx, bucket in enumerate(buckets)
        if any(
            fnmatch.fnmatch(fqn, pattern) or fnmatch.fnmatch(fqn, f"{pattern}.*")
            for pattern in bucket.patterns
        )
    ]


def _hook_module_fqns(module: nn.Module, patterns: list[str]) -> list[str] | None:
    """The modules a bucket hooks when all its patterns name modules, else None
    (see ``BucketSpec.patterns``)."""
    named = []
    for pattern in patterns:
        try:
            named.append(module.get_submodule(pattern))
        except AttributeError:
            pass
    if not named:
        return None
    if len(named) != len(patterns):
        raise ValueError(
            f"BucketSpec patterns {patterns} mix module FQNs with parameter "
            "patterns; a bucket's patterns must all name modules, whose forwards "
            "then carry its hooks, or all match parameters."
        )
    for pattern, submodule in zip(patterns, named, strict=True):
        if type(submodule).forward is nn.Module.forward:
            raise ValueError(
                f"BucketSpec pattern {pattern!r} names a "
                f"{type(submodule).__name__}, which has no forward to hook; name "
                "the modules it contains instead."
            )
    return list(patterns)


def _fsdp2_param_order(
    module: nn.Module,
    patterns: list[str],
    named_params: list[tuple[str, nn.Parameter]],
) -> list[tuple[str, nn.Parameter]]:
    """Order a bucket's params as ``fully_shard`` orders its group's params.

    FSDP2 walks the modules it is given in post-order and takes each module's
    own params when it first meets them. A bucket of module patterns walks
    those modules, in pattern order. For parameter patterns, any module that
    contains the bucket gives the same relative order, so walk the root.
    """
    hook_fqns = _hook_module_fqns(module, patterns)
    roots = [module.get_submodule(fqn) for fqn in hook_fqns] if hook_fqns else [module]
    bucket_param_ids = {id(param) for _, param in named_params}
    positions: dict[int, int] = {}
    visited: set[nn.Module] = set()

    def visit(submodule: nn.Module) -> None:
        visited.add(submodule)
        for child in submodule.children():
            if child not in visited:
                visit(child)
        for param in submodule.parameters(recurse=False):
            if id(param) in bucket_param_ids:
                positions.setdefault(id(param), len(positions))

    for root in roots:
        if root not in visited:
            visit(root)
    return sorted(named_params, key=lambda named_param: positions[id(named_param[1])])
