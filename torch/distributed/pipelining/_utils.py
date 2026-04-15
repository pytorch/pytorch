# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import cast, Literal, overload, Protocol, TYPE_CHECKING, TypeAlias

import torch
from torch import fx
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import Placement


logger = logging.getLogger(__name__)


class GetMeshCallback(Protocol):
    """Callback to create/retrieve a DeviceMesh from its cache key components."""

    def __call__(
        self,
        mesh_dim_names: tuple[str, ...],
        mesh_layout: _MeshLayout | None,
    ) -> DeviceMesh: ...


# Key for mesh cache: (mesh_dim_names, mesh_layout)
# mesh_layout is the _MeshLayout object containing shape and stride (not actual ranks).
# This uniquely identifies a mesh within the same "universe" where all stages share
# the same rank tensor.
MeshCacheKey: TypeAlias = tuple[tuple[str, ...], _MeshLayout | None]


class PipeliningMetadataError(RuntimeError):
    """Raised on metadata mismatches during pipeline communication."""


@dataclass(frozen=True, slots=True)
class _TensorMeta:
    """Tensor metadata for recv buffer allocation and validation.

    For plain tensors, these are the tensor's actual attributes.
    For DTensors, these are LOCAL shard attributes; global attributes
    are stored in :class:`_DTensorMeta`.
    """

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> _TensorMeta:
        """Create metadata from a plain tensor.

        Args:
            tensor: A plain ``torch.Tensor`` (not DTensor).

        Returns:
            Metadata capturing shape, stride, dtype, and requires_grad.

        Raises:
            TypeError: If ``tensor`` is a DTensor.
        """
        if isinstance(tensor, DTensor):
            raise PipeliningMetadataError(
                "Expected plain tensor, got DTensor. Use _DTensorMeta.from_dtensor instead."
            )
        return _TensorMeta(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

    def to_tensor(self, device: torch.device | str) -> torch.Tensor:
        """Reconstruct a tensor on ``device`` from this metadata.

        Args:
            device: Target device for the tensor.

        Returns:
            An empty strided tensor on ``device``.
        """
        t = _make_tensor_from_meta(self, device)
        t.requires_grad_(self.requires_grad)
        return t

    def get_diff(self, other: _TensorMeta) -> list[str]:
        """Return field-by-field differences with ``other``.

        Args:
            other: Metadata to compare against.

        Returns:
            List of human-readable difference strings (empty if equal).
        """
        if self == other:
            return []

        diffs = []
        if self.shape != other.shape:
            diffs.append(f"shape mismatch: {self.shape} vs {other.shape}")
        if self.stride != other.stride:
            diffs.append(f"stride mismatch: {self.stride} vs {other.stride}")
        if self.dtype != other.dtype:
            diffs.append(f"dtype mismatch: {self.dtype} vs {other.dtype}")
        # requires_grad is intentionally excluded: it is a runtime concern
        # determined by has_backward and grad context, not a metadata invariant.
        return diffs


@dataclass(frozen=True, slots=True)
class _DTensorMeta(_TensorMeta):
    """DTensor metadata extending :class:`_TensorMeta` with distribution info.

    Inherited fields (shape, stride, etc.) are LOCAL shard attributes.
    Additional fields capture global shape and placement information
    needed to reconstruct a :class:`DTensor` via ``DTensor.from_local()``.

    The :class:`DeviceMesh` is **not** stored (not serializable for P2P);
    it is looked up from :class:`_MeshCache` using
    ``(mesh_dim_names, mesh_layout)`` as the key.
    """

    # Global DTensor properties (for reconstruction)
    global_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    global_stride: tuple[int, ...] = field(default=())

    # DTensor distribution properties
    placements: tuple[Placement, ...] = field(
        default=()
    )  # e.g., (Shard(0), Replicate())

    # Mesh identification - used to look up the correct DeviceMesh from cache
    mesh_dim_names: tuple[str, ...] = field(default=())  # e.g., ("tp",) or ("dp", "tp")
    mesh_layout: _MeshLayout | None = field(
        default=None
    )  # _MeshLayout with shape/stride - uniquely identifies mesh within the same universe

    @staticmethod
    def from_dtensor(dtensor: DTensor) -> _DTensorMeta:
        """Create metadata from a DTensor.

        Args:
            dtensor: The DTensor to extract metadata from.

        Returns:
            Metadata capturing both local and global attributes.
        """
        device_mesh = dtensor.device_mesh

        return _DTensorMeta(
            # Local tensor attributes (for recv buffer allocation)
            shape=dtensor._local_tensor.shape,
            stride=dtensor._local_tensor.stride(),
            dtype=dtensor.dtype,
            requires_grad=dtensor.requires_grad,
            # Global DTensor attributes (for reconstruction)
            global_shape=dtensor.shape,
            global_stride=dtensor.stride(),
            # Distribution info
            placements=dtensor._spec.placements,
            mesh_dim_names=(
                tuple(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names else ()
            ),
            mesh_layout=device_mesh._layout,
        )

    @property
    def mesh_cache_key(self) -> MeshCacheKey:
        """Cache key ``(mesh_dim_names, mesh_layout)`` for mesh lookup."""
        return (self.mesh_dim_names, self.mesh_layout)

    def to_dtensor(self, device: torch.device | str, mesh: DeviceMesh) -> DTensor:
        """Reconstruct a DTensor on ``device`` with placements.

        Args:
            device: Target device for the local tensor.
            mesh: The ``DeviceMesh`` to attach.

        Returns:
            A DTensor on ``device``.
        """
        local_tensor = _make_tensor_from_meta(self, device)
        # Set requires_grad after from_local() so that the from_local
        # operation itself is not recorded in the autograd graph.
        return cast(
            DTensor,
            DTensor.from_local(
                local_tensor,
                device_mesh=mesh,
                placements=self.placements,
                shape=self.global_shape,
                stride=self.global_stride,
                run_check=False,
            ).requires_grad_(self.requires_grad),
        )

    def get_diff(self, other: _TensorMeta) -> list[str]:
        """Return field-by-field differences, including DTensor-specific fields.

        Args:
            other: Metadata to compare against.

        Returns:
            List of human-readable difference strings (empty if equal).
        """
        if self == other:
            return []

        # Get base class differences (compares local shape/stride/dtype/requires_grad)
        # NOTE: Use explicit class call instead of super() because
        # @dataclass(slots=True) on both parent and child can break super().
        diffs = _TensorMeta.get_diff(self, other)

        # Add DTensor-specific comparisons if other is also _DTensorMeta
        if isinstance(other, _DTensorMeta):
            if self.global_shape != other.global_shape:
                diffs.append(
                    f"global_shape mismatch: {self.global_shape} vs {other.global_shape}"
                )
            if self.global_stride != other.global_stride:
                diffs.append(
                    f"global_stride mismatch: {self.global_stride} vs {other.global_stride}"
                )
            if self.placements != other.placements:
                diffs.append(
                    f"placements mismatch: {self.placements} vs {other.placements}"
                )
            if self.mesh_dim_names != other.mesh_dim_names:
                diffs.append(
                    f"mesh_dim_names mismatch: {self.mesh_dim_names} vs {other.mesh_dim_names}"
                )
            if self.mesh_layout != other.mesh_layout:
                diffs.append(
                    f"mesh_layout mismatch: {self.mesh_layout} vs {other.mesh_layout}"
                )
        else:
            diffs.append("type: _DTensorMeta vs _TensorMeta")

        return diffs


# Type alias for union of tensor metadata types
TensorMeta: TypeAlias = _TensorMeta | _DTensorMeta


# Not frozen: fields are populated incrementally during forward and
# backward metadata inference or from user provided static metadata
@dataclass(slots=True)
class _StageMeta:
    """Consolidated tensor metadata for a pipeline stage's forward and backward passes."""

    inputs: tuple[TensorMeta, ...] | None = None
    outputs: tuple[TensorMeta, ...] | None = None
    input_grads: tuple[TensorMeta | None, ...] | None = None
    output_grads: tuple[TensorMeta | None, ...] | None = None

    def has_any(self) -> bool:
        """Check if any metadata field is populated."""
        return any(
            v is not None
            for v in [self.inputs, self.outputs, self.input_grads, self.output_grads]
        )

    def has_dtensors(self) -> bool:
        """Check if any input/output metadata is DTensor type."""
        for metas in [self.inputs, self.outputs]:
            if metas and any(isinstance(m, _DTensorMeta) for m in metas if m):
                return True
        return False

    def is_complete_for_forward(self) -> bool:
        """Check if forward metadata is fully populated."""
        return self.inputs is not None and self.outputs is not None


@dataclass(frozen=True, slots=True)
class _StageForwardMeta:
    """Forward metadata transmitted from stage *i* to stage *i+1* during inference."""

    forward_metas: tuple[TensorMeta, ...]  # Stage i's outputs → Stage i+1's inputs


@dataclass(frozen=True, slots=True)
class _StageBackwardMeta:
    """Backward metadata transmitted from stage *i* to stage *i-1* during inference.

    Gradient placements may differ from forward activations
    (e.g., ``Replicate`` → ``Partial``).
    """

    backward_metas: tuple[
        TensorMeta | None, ...
    ]  # Stage i's input_grads → Stage i-1's output_grads


def _make_tensor_from_meta(
    meta: _TensorMeta,
    device: torch.device | str,
) -> torch.Tensor:
    """Create a tensor from metadata.

    Args:
        meta: Metadata with shape, stride, and dtype.
        device: Target device for the tensor.

    Returns:
        Empty tensor preserving the exact memory layout.
    """
    return torch.empty_strided(
        size=meta.shape,
        stride=meta.stride,
        dtype=meta.dtype,
        device=device,
    )


def _derive_grad_metas(
    tensor_metas: tuple[TensorMeta, ...],
) -> tuple[_TensorMeta | None, ...]:
    """Derive gradient metadata from tensor metadata.

    Returns metadata with the same shape/stride/dtype but ``requires_grad=False``.
    Entries where the source has ``requires_grad=False`` become ``None``.
    """
    return tuple(
        _TensorMeta(shape=m.shape, stride=m.stride, dtype=m.dtype, requires_grad=False)
        if m.requires_grad
        else None
        for m in tensor_metas
    )


class _MeshCache:
    """Cache for :class:`DeviceMesh` objects keyed by ``(mesh_dim_names, mesh_layout)``.

    Assumes all pipeline stages share the same rank tensor (true for
    TorchTitan-style frameworks where meshes derive from a common world).
    """

    def __init__(self, get_mesh_cb: GetMeshCallback | None = None) -> None:
        self._cache: dict[MeshCacheKey, DeviceMesh] = {}
        self._get_mesh_cb = get_mesh_cb

    def get_mesh(self, key: MeshCacheKey) -> DeviceMesh:
        """Return a cached mesh, or create one via the callback.

        Args:
            key: Cache key ``(mesh_dim_names, mesh_layout)``.

        Returns:
            The ``DeviceMesh``.

        Raises:
            PipeliningMetadataError: If not cached and no callback provided.
        """
        if key in self._cache:
            return self._cache[key]

        mesh_dim_names, mesh_layout = key

        if self._get_mesh_cb is None:
            raise PipeliningMetadataError(
                f"Mesh not found in cache for mesh_dim_names={mesh_dim_names}, "
                f"mesh_layout={mesh_layout}, and no get_mesh callback provided. "
                f"Provide a get_mesh callback or use DTensors in static mode."
            )

        mesh = self._get_mesh_cb(mesh_dim_names, mesh_layout)
        if mesh is None:
            raise PipeliningMetadataError(
                f"Mesh lookup failed: callback returned None for "
                f"mesh_dim_names={mesh_dim_names}, mesh_layout={mesh_layout}. "
                f"Ensure all stages use meshes from the same universe."
            )
        self._cache[key] = mesh
        return mesh

    def put(self, key: MeshCacheKey, mesh: DeviceMesh) -> None:
        """Add a mesh to the cache."""
        self._cache[key] = mesh

    def update_from_tensors(self, tensors: tuple[torch.Tensor | None, ...]) -> None:
        """Extract and cache meshes from any :class:`DTensor` instances in *tensors*."""
        for tensor in tensors:
            if isinstance(tensor, DTensor):
                mesh = tensor.device_mesh
                dim_names = tuple(mesh.mesh_dim_names) if mesh.mesh_dim_names else ()
                mesh_layout = mesh._layout
                key = (dim_names, mesh_layout)
                if key not in self._cache:
                    self._cache[key] = mesh

    def __contains__(self, key: MeshCacheKey) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ============================================================================
# Inference mode enum
# ============================================================================


class InferenceMode(Enum):
    """Pipeline-level metadata inference mode, determined collectively across all PP ranks.

    The mode is set by the schedule (not individual stages) because
    ``has_backward`` is only known at schedule creation time and all
    stages must agree to avoid P2P hangs.

    .. attribute:: STATIC

        All stages have sufficient metadata; runtime inference is skipped.

    .. attribute:: DYNAMIC

        At least one stage requires runtime metadata inference.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"

    @classmethod
    def needs_dynamic(cls, meta: _StageMeta, stage_has_backward: bool) -> bool:
        """Determine whether dynamic metadata inference is needed for a stage.

        Args:
            meta: Stage metadata from user-provided args.
            stage_has_backward: Whether a backward pass will be performed.

        Returns:
            ``True`` if dynamic inference is needed.
        """
        # Case 1: Forward metadata incomplete → needs DYNAMIC
        if not meta.is_complete_for_forward():
            return True

        # Case 2: No DTensors → STATIC is fine (bwd metadata derivable from fwd metadata)
        if not meta.has_dtensors():
            return False

        # Case 3: No backward needed → STATIC is fine (don't need grad metadata)
        if not stage_has_backward:
            return False

        # Case 4: DTensors with backward but missing ANY grad metadata → needs DYNAMIC
        # Both input_grads AND output_grads are required for static mode with DTensors
        if meta.input_grads is None or meta.output_grads is None:
            return True

        # Case 5: DTensors with complete grads → STATIC is fine
        return False


# ============================================================================
# Utility functions
# ============================================================================


def flatten_args(args, *, detach: bool = False):
    """Flatten ``args`` into a list, optionally detaching tensors.

    Args:
        args: Nested arguments to flatten.
        detach: If ``True``, detach tensors while preserving ``requires_grad``.

    Returns:
        ``(new_args, flat_detached_args)`` when ``detach=True``;
        ``flat_args`` list otherwise.
    """
    flat_args, treespec = tree_flatten(args)

    if detach:
        flat_detached = [
            a.detach().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor)
            else a
            for a in flat_args
        ]
        new_args = tree_unflatten(flat_detached, treespec)
        return new_args, flat_detached

    return flat_args


# Backward compatibility alias
def flatten_args_detach(args):
    """Flatten and detach. Deprecated: use ``flatten_args(args, detach=True)``."""
    return flatten_args(args, detach=True)


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # dont change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, list[int]]:
    """
    Compute the rank to stage id mapping for either a looped or V-style schedule.

    This function inverts the stage_to_rank_mapping to get which stages are assigned to each rank.

    Returns a dictionary mapping rank -> list of stage indices assigned to that rank.
    """
    stage_to_rank = generate_stage_to_rank_mapping(pp_size, num_stages, style)

    # Invert the mapping: rank -> list of stages
    rank_to_stages: dict[int, list[int]] = {}
    for stage_id, rank in stage_to_rank.items():
        if rank not in rank_to_stages:
            rank_to_stages[rank] = []
        rank_to_stages[rank].append(stage_id)

    # Sort the stage lists for each rank to ensure consistent ordering
    for stages in rank_to_stages.values():
        stages.sort()

    return rank_to_stages


@dataclass(slots=True)
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool


# ============================================================================
# Metadata extraction helpers
# ============================================================================


def extract_tensor_meta(tensor: torch.Tensor) -> TensorMeta:
    """Extract metadata from a tensor.

    Handles both plain Tensor and DTensor correctly: DTensors are
    dispatched to ``_DTensorMeta.from_dtensor`` which captures local
    shard attributes plus global shape/placement info, while plain
    tensors use ``_TensorMeta.from_tensor``.

    Args:
        tensor: A plain tensor or DTensor.

    Returns:
        ``_TensorMeta`` for plain tensors, ``_DTensorMeta`` for DTensors.
    """
    if isinstance(tensor, DTensor):
        return _DTensorMeta.from_dtensor(tensor)
    else:
        return _TensorMeta.from_tensor(tensor)


@overload
def extract_tensor_metas(
    tensors: tuple[torch.Tensor, ...] | None,
    *,
    allow_none: Literal[False] = ...,
) -> tuple[TensorMeta, ...] | None: ...


@overload
def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...] | None,
    *,
    allow_none: Literal[True],
) -> tuple[TensorMeta | None, ...] | None: ...


def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...] | tuple[torch.Tensor, ...] | None,
    *,
    allow_none: bool = False,
) -> tuple[TensorMeta | None, ...] | None:
    """Extract metadata from a tuple of tensors.

    Args:
        tensors: Tuple of tensors (may include ``None`` when ``allow_none=True``).
        allow_none: If ``True``, preserve ``None`` elements (for gradients).

    Returns:
        Tuple of ``TensorMeta``, or ``None`` if ``tensors`` is ``None``.

    Raises:
        PipeliningMetadataError: If ``None`` found and ``allow_none=False``.
    """
    if tensors is None:
        return None

    metas_with_none: list[TensorMeta | None] = []
    has_none = False
    for t in tensors:
        if isinstance(t, torch.Tensor):
            metas_with_none.append(extract_tensor_meta(t))
        else:
            has_none = True
            metas_with_none.append(None)
    if not allow_none and has_none:
        raise PipeliningMetadataError(
            "None values are not allowed in tensor metadata tuples. "
            "Use allow_none=True for optional values."
        )
    return tuple(metas_with_none)


def to_local_if_dtensor(tensor: torch.Tensor, detach: bool = False) -> torch.Tensor:
    """Convert a DTensor to its local shard, or return a plain tensor as-is.

    When ``detach=True``, the tensor is detached before conversion —
    this applies to both DTensors and plain tensors.

    Args:
        tensor: A tensor that may be a DTensor.
        detach: If ``True``, detach before ``to_local()`` to avoid
            redistribution during backward.

    Returns:
        The local tensor component.
    """
    maybe_detached_tensor = tensor.detach() if detach else tensor
    if isinstance(maybe_detached_tensor, DTensor):
        return maybe_detached_tensor.to_local()
    return maybe_detached_tensor


@overload
def validate_and_normalize_to_tuple(
    args: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
    allow_none: Literal[False] = ...,
) -> tuple[torch.Tensor, ...] | None: ...


@overload
def validate_and_normalize_to_tuple(
    args: torch.Tensor
    | tuple[torch.Tensor | None, ...]
    | list[torch.Tensor | None]
    | None,
    allow_none: Literal[True] = ...,
) -> tuple[torch.Tensor | None, ...] | None: ...


def validate_and_normalize_to_tuple(
    args: torch.Tensor
    | tuple[torch.Tensor, ...]
    | tuple[torch.Tensor | None, ...]
    | list[torch.Tensor]
    | list[torch.Tensor | None]
    | None,
    allow_none: bool = False,
) -> tuple[torch.Tensor | None, ...] | tuple[torch.Tensor, ...] | None:
    """Normalize ``args`` to a tuple and validate that all elements are tensors.

    Args:
        args: A single tensor, tuple/list of tensors, or ``None``.
        allow_none: If ``True``, permit ``None`` elements (for gradients).

    Returns:
        Tuple of tensors, or ``None`` if ``args`` is ``None``.

    Raises:
        PipeliningMetadataError: On non-tensor values
            (or ``None`` when ``allow_none=False``).
    """
    if args is None:
        return None
    elif isinstance(args, torch.Tensor):
        return (args,)
    elif isinstance(args, (tuple, list)):
        for i, arg in enumerate(args):
            if arg is None:
                if not allow_none:
                    raise PipeliningMetadataError(
                        f"Stage arg[{i}] is None. "
                        f"Stage args must be tensors. Use kwargs for optional values."
                    )
                continue
            if not isinstance(arg, torch.Tensor):
                raise PipeliningMetadataError(
                    f"Stage arg[{i}] has type {type(arg).__name__}. "
                    f"All stage args must be tensors. Use kwargs for non-tensor inputs."
                )
        # Normalize list to tuple
        return tuple(args) if isinstance(args, list) else args
    else:
        raise PipeliningMetadataError(
            f"Stage args must be a tensor, tuple, or list of tensors, got {type(args).__name__}."
        )


# ============================================================================
# Validation functions
# ============================================================================


def validate_metadata(
    desc: str,
    expected: TensorMeta,
    actual: torch.Tensor | TensorMeta,
    *,
    raise_on_mismatch: bool = False,
    warn_on_mismatch: bool = False,
) -> list[str]:
    """
    Compare expected metadata against actual tensor or metadata.

    This is the unified validation/comparison function that uses get_diff() from
    metadata classes. Works with both plain tensors and DTensors.

    For plain tensors: compares shape/stride/dtype/requires_grad.
    For DTensors: compares all properties including global shape and placements.

    Args:
        desc: Description for error/warning messages.
        expected: Expected tensor metadata (_TensorMeta or _DTensorMeta).
        actual: Actual tensor or metadata to compare against.
        raise_on_mismatch: If True, raise PipeliningMetadataError on mismatch.
        warn_on_mismatch: If True, issue a warning on mismatch.

    Returns:
        List of differences (empty if metadata matches).

    Raises:
        PipeliningMetadataError: If raise_on_mismatch=True and differences exist.
    """
    # Extract metadata if actual is a tensor
    if isinstance(actual, torch.Tensor):
        actual_meta = extract_tensor_meta(actual)
    else:
        actual_meta = actual

    # Type check: ensure both are same type for meaningful comparison
    if type(expected) is not type(actual_meta):
        type_diff = [
            f"type: expected {type(expected).__name__}, got {type(actual_meta).__name__}"
        ]
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {type_diff[0]}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata type mismatch. {type_diff[0]}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
                stacklevel=2,
            )
        return type_diff

    # Use get_diff() from the metadata class
    diffs = expected.get_diff(actual_meta)

    if diffs:
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {'; '.join(diffs)}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata mismatch. {'; '.join(diffs)}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
                stacklevel=2,
            )

    return diffs


def validate_tensors_metadata(
    desc: str,
    expected: tuple[TensorMeta | None, ...],
    actual: tuple[torch.Tensor | TensorMeta | None, ...],
    *,
    raise_on_mismatch: bool = True,
    warn_on_mismatch: bool = False,
) -> list[str]:
    """Validate metadata for a tuple of tensors element-wise.

    Args:
        desc: Description prefix for error/warning messages.
        expected: Tuple of expected metadata (may include ``None`` for grads).
        actual: Tuple of actual tensors or metadata to compare against.
        raise_on_mismatch: If ``True``, raise on the first mismatch.
        warn_on_mismatch: If ``True``, issue warnings for mismatches.

    Returns:
        Aggregated list of difference strings.

    Raises:
        PipeliningMetadataError: If lengths differ or on mismatch.
    """
    if len(expected) != len(actual):
        msg = f"{desc}: expected {len(expected)} tensors, got {len(actual)}"
        if raise_on_mismatch:
            raise PipeliningMetadataError(msg)
        if warn_on_mismatch:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return [msg]

    all_diffs: list[str] = []
    for i, (exp, act) in enumerate(zip(expected, actual, strict=True)):
        if exp is None and act is None:
            continue
        if exp is None or act is None:
            msg = (
                f"{desc}[{i}]: expected {'None' if exp is None else 'metadata'}, "
                f"got {'None' if act is None else 'metadata'}"
            )
            if raise_on_mismatch:
                raise PipeliningMetadataError(msg)
            if warn_on_mismatch:
                warnings.warn(msg, UserWarning, stacklevel=2)
            all_diffs.append(msg)
            continue
        diffs = validate_metadata(
            f"{desc}[{i}]",
            exp,
            act,
            raise_on_mismatch=raise_on_mismatch,
            warn_on_mismatch=warn_on_mismatch,
        )
        all_diffs.extend(diffs)
    return all_diffs


def validate_static_arg_grad_correspondence(
    stage_index: int,
    args: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
    is_input: bool,
) -> None:
    """
    Validate the args↔grads contract for static mode.

    Enforces four rules for each (arg, grad) pair:
      1. len(args) must equal len(grads).
      2. If arg.requires_grad is False, grad must be None.
      3. If arg.requires_grad is True and grad is None, emit a warning
         (this is legal at pipeline boundaries but may indicate a bug).
      4. If arg is a DTensor with requires_grad=True and grad is not None,
         grad must also be a DTensor.

    Args:
        stage_index: The stage index for error messages.
        args: Tuple of forward tensors.
        grads: Tuple of gradient tensors (can include None).
        is_input: True for input_args/input_grads, False for output_args/output_grads.

    Raises:
        PipeliningMetadataError: If any hard rule (1, 2, or 4) is violated.
    """
    kind = "input" if is_input else "output"
    args_name = f"{kind}_args"
    grads_name = f"{kind}_grads"

    # Rule 1: lengths must match
    if len(args) != len(grads):
        raise PipeliningMetadataError(
            f"Stage {stage_index}: {grads_name} length ({len(grads)}) does not match "
            f"{args_name} length ({len(args)}). Each forward tensor must have a "
            f"corresponding gradient entry (use None for tensors that don't require grad)."
        )

    for i, (arg, grad) in enumerate(zip(args, grads, strict=True)):
        # Rule 2: no grad for a non-differentiable arg
        if not arg.requires_grad and grad is not None:
            raise PipeliningMetadataError(
                f"Stage {stage_index}: {args_name}[{i}] has requires_grad=False, "
                f"but {grads_name}[{i}] is not None ({type(grad).__name__}). "
                f"Non-differentiable tensors must have None as their gradient entry."
            )

        # Rule 3: missing grad for a differentiable arg (warn, don't raise)
        if arg.requires_grad and grad is None:
            warnings.warn(
                f"Stage {stage_index}: {args_name}[{i}] has requires_grad=True, "
                f"but {grads_name}[{i}] is None. This is legal at pipeline boundaries "
                f"but may indicate a missing gradient.",
                UserWarning,
                stacklevel=2,
            )

        # Rule 4: DTensor arg must have DTensor grad
        if (
            isinstance(arg, DTensor)
            and arg.requires_grad
            and grad is not None
            and not isinstance(grad, DTensor)
        ):
            raise PipeliningMetadataError(
                f"Stage {stage_index}: {args_name}[{i}] is a DTensor with requires_grad=True, "
                f"but {grads_name}[{i}] is {type(grad).__name__}, expected DTensor or None. "
                f"DTensor gradients may have different placements than forward tensors."
            )
