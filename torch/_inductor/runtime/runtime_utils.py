from __future__ import annotations

import functools
import math
import operator
from typing import Any, TYPE_CHECKING

import sympy

import torch

# NOTE: other files rely on the imports below
from torch._dynamo import callback as compilation_callback  # noqa: F401
from torch._inductor.runtime.cache_dir_utils import (  # noqa: F401
    cache_dir,
    default_cache_dir,
    triton_cache_dir,
)


if TYPE_CHECKING:
    from collections.abc import Hashable

    from .triton_compat import Config


def conditional_product(*args: int) -> int:
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(number: int, denom: int) -> int:
    return -(number // -denom)


def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    if isinstance(n, sympy.Integer):
        n = int(n)
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def last_power_of_2(n: int) -> int:
    """Return the largest power of 2 less than or equal to n"""
    next_pow2 = next_power_of_2(n)
    return next_pow2 // 2 if next_pow2 > n else next_pow2


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg: Config) -> Hashable:
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    # pyrefly: ignore [missing-attribute]
    items = sorted(cfg.kwargs.items())
    # pyrefly: ignore [missing-attribute]
    items.append(("num_warps", cfg.num_warps))
    # pyrefly: ignore [missing-attribute]
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def validate_triton_config(cfg: Config) -> None:
    # [Note: Triton pre_hook in inductor]
    # pre-hook is a lambda function, which we don't attempt to serialize.
    # right now, if a pre-hook is attached to the config, it will not be saved;
    # and then it won't be used when the config is loaded from cache.
    # So we assert - if we do get a pre_hook, it might get ignored after caching.
    assert getattr(cfg, "pre_hook", None) is None, (
        "triton configs with pre_hooks not supported"
    )


def create_bandwidth_info_str(
    ms: float,
    num_gb: float,
    gb_per_s: float,
    prefix: str = "",
    suffix: str = "",
    color: bool = True,
) -> str:
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid() -> int:
    return 65535


try:
    import colorama

    HAS_COLORAMA = True
except ModuleNotFoundError:
    HAS_COLORAMA = False
    colorama = None  # type: ignore[assignment]


if HAS_COLORAMA:

    def _color_text(msg: str, color: str) -> str:
        # pyrefly: ignore [missing-attribute]
        return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET

else:

    def _color_text(msg: str, color: str) -> str:
        return msg


def green_text(msg: str) -> str:
    return _color_text(msg, "green")


def yellow_text(msg: str) -> str:
    return _color_text(msg, "yellow")


def red_text(msg: str) -> str:
    return _color_text(msg, "red")


def blue_text(msg: str) -> str:
    return _color_text(msg, "blue")


def get_first_attr(obj: Any, *attrs: str) -> Any:
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


dynamo_timed = torch._dynamo.utils.dynamo_timed  # type: ignore[has-type]


def triton_hash_to_path_key(key: str) -> str:
    # In early versions of Triton, the hash is directly used in the path name.
    # Later, the hash is converted to base64 before being used in the path name.
    # Later, the base64 conversion was replaced to the base32
    #
    # This code tries to import _base64 and falls back to _base32 if _base64 is unavailable.
    #
    # To handle this, try to import the to-base64-conversion function.
    # If it exists, use it; otherwise, try using _base32; if both are unavailable, use the hash directly.
    try:
        from triton.runtime.cache import _base64

        return _base64(key)
    except Exception:
        try:
            from triton.runtime.cache import _base32

            return _base32(key)
        except Exception:
            return key


def compile_mps_shader(source: str) -> Any:
    """
    Compiles shader source but raise more actionable error message when needed
    """
    try:
        return torch.mps.compile_shader(source)
    except SyntaxError as err:
        raise SyntaxError(f"failed to compile {source} with {err.msg}") from err


def torch_dtype_to_jax_runtime(dtype: torch.dtype) -> Any:
    """
    Map PyTorch dtype to actual JAX dtype object at runtime.

    This helper is used in generated Pallas kernels at runtime to convert
    PyTorch dtypes to JAX dtype objects (not string representations).

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        JAX dtype object (e.g., jnp.float32 object itself)
    """
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    dtype_map = {
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        torch.float16: jnp.float16,
        torch.bfloat16: jnp.bfloat16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.int16: jnp.int16,
        torch.int8: jnp.int8,
        torch.uint8: jnp.uint8,
        torch.bool: jnp.bool_,
        torch.complex64: jnp.complex64,
        torch.complex128: jnp.complex128,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype for JAX conversion: {dtype}")
    return dtype_map[dtype]


def torch_dtype_to_jax(dtype: torch.dtype) -> str:
    """
    Map PyTorch dtype to JAX dtype expression string.

    This helper is used at compile time in codegen to generate
    JAX dtype expressions for Pallas kernels.

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        JAX dtype expression as string (e.g., "jnp.float32")
    """
    jax_dtype = torch_dtype_to_jax_runtime(dtype)
    dtype_name = jax_dtype.__name__
    if dtype_name == "bool":
        dtype_name = "bool_"
    return f"jnp.{dtype_name}"


def pallas_partial_reduce(reduce_fn: Any, v: Any, pw_numel: int, red_numel: int) -> Any:
    """
    Helper for partial reductions in Pallas kernels.

    Reduces over contiguous axes whose product matches *red_numel* in the
    original (un-tiled) tensor, returning the result with keepdims-style
    shape for proper in-kernel broadcasting.

    When running inside a tiled pallas_call the actual tile shape may differ
    from ``(pw_numel, red_numel)``, so we reduce directly over the discovered
    axes instead of reshaping to exact sizes.

    Args:
        reduce_fn: The reduction function to apply (e.g., jnp.sum, jnp.max)
        v: The input array to reduce
        pw_numel: The number of pointwise elements (used for axis detection)
        red_numel: The number of reduction elements (used for axis detection)

    Returns:
        Reduced array with keepdims-style shape
    """
    shape = tuple(v.shape)
    # Find contiguous axes whose product = red_numel (search from right)
    red_axes = None
    for i in range(len(shape) - 1, -1, -1):
        prod = 1
        for j in range(i, -1, -1):
            prod *= shape[j]
            if prod == red_numel:
                red_axes = list(range(j, i + 1))
                break
        if red_axes is not None:
            break
    if red_axes is None:
        red_axes = [len(shape) - 1]
    result = reduce_fn(v, axis=tuple(red_axes), keepdims=True)
    return result


def pallas_gpu_pad_inputs(inputs: list[Any], alignment: int = 128) -> list[Any]:
    """Flatten and pad each input JAX array to a multiple of alignment."""
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    padded = []
    for inp in inputs:
        flat = inp.flatten()
        orig_size = flat.size
        aligned_size = ((orig_size + alignment - 1) // alignment) * alignment
        if orig_size != aligned_size:
            padded.append(jnp.pad(flat, (0, aligned_size - orig_size)))
        else:
            padded.append(flat)
    return padded


def pallas_gpu_align_output_specs(
    out_shapes: tuple[Any, ...],
    out_dtypes: tuple[Any, ...],
    alignment: int = 128,
) -> tuple[tuple[Any, ...], list[bool]]:
    """Build aligned output ShapeDtypeStruct specs for GPU kernels.

    Returns (aligned_specs, is_scalar_output) where is_scalar_output[i] is True
    when the i-th output is scalar and should not be padded/unpadded.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]

    aligned_specs = []
    is_scalar = []
    for shape, dtype in zip(out_shapes, out_dtypes):
        numel = math.prod(shape)
        if numel <= 1:
            aligned_specs.append(jax.ShapeDtypeStruct(shape, dtype))
            is_scalar.append(True)
        else:
            aligned_numel = ((numel + alignment - 1) // alignment) * alignment
            aligned_specs.append(jax.ShapeDtypeStruct((aligned_numel,), dtype))
            is_scalar.append(False)
    return tuple(aligned_specs), is_scalar


def pallas_gpu_unpad_results(
    results: Any,
    orig_shapes: tuple[Any, ...],
    is_scalar_output: list[bool] | None = None,
) -> Any:
    """Remove padding from GPU kernel results and reshape to original shapes.

    If is_scalar_output is None, all outputs are treated as non-scalar.
    """
    if not isinstance(results, tuple):
        results = (results,)
    unpadded = []
    for i, (res, shape) in enumerate(zip(results, orig_shapes)):
        if is_scalar_output is not None and is_scalar_output[i]:
            unpadded.append(res)
        else:
            orig_numel = math.prod(shape)
            unpadded.append(res[:orig_numel].reshape(shape))
    return unpadded[0] if len(unpadded) == 1 else tuple(unpadded)


# ---------------------------------------------------------------------------
# Pallas CPU / TPU tiling helpers
# ---------------------------------------------------------------------------
# TPU alignment: last dim must be full or a multiple of 128,
#                second-to-last dim must be full or a multiple of 8.
_TPU_ALIGN_LAST = 128
_TPU_ALIGN_SECOND_LAST = 8


def _pallas_tile_size(dim: int, alignment: int, max_tile: int = 1024) -> int:
    """Pick the largest aligned tile size <= max_tile for *dim*.

    If *dim* is already <= alignment the full dimension is used (no tiling
    on this axis).
    """
    if dim <= alignment:
        return dim
    t = min(max_tile, dim)
    t = (t // alignment) * alignment
    return max(alignment, t)


def pallas_compute_tiling(
    ref_shape: tuple[int, ...],
    transpose: bool = False,
    skip_last_n: int = 0,
    exact_only: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, int]]:
    """Compute tile shape, grid and axis→grid-dim mapping for CPU/TPU.

    Always uses TPU-compatible alignment (last dim multiple of 128,
    second-to-last multiple of 8) so that the same generated kernel works
    on both CPU-interpret and real TPU.

    When *transpose* is True and nd >= 2, both last-2 dims use the same
    square tile size based on the smaller alignment so that transposed
    buffers can use the same tile for both dims.

    *skip_last_n* prevents tiling the last N dimensions (used when those
    dims correspond to internal reduction ranges that must remain full).

    *exact_only* restricts tiling to dimensions that divide evenly by the
    tile size (no remainder blocks).  Required on TPU where Mosaic needs
    block shapes to match the XLA memory layout.

    Returns ``(tile_shape, grid, axis_to_grid)`` where *axis_to_grid*
    maps each tiled reference-shape axis index to its position in the
    grid tuple.

    When no dimension benefits from tiling the grid is ``(1,)`` and the
    tile covers the full tensor.
    """
    nd = len(ref_shape)
    if nd == 0:
        return (), (1,), {}

    # Effective number of dims eligible for tiling
    tileable_nd = nd - skip_last_n

    tile = list(ref_shape)
    grid_parts: list[int] = []
    axis_to_grid: dict[int, int] = {}  # ref axis → grid dim

    # Pick alignment based on the physical position of the axis in the
    # tensor, not its position in the tileable subset.  The TPU requires
    # the physical last dim to be a multiple of 128 and the physical
    # second-to-last dim to be a multiple of 8.
    def _align(ax: int) -> int:
        return _TPU_ALIGN_LAST if ax == nd - 1 else _TPU_ALIGN_SECOND_LAST

    def _can_tile_ax(dim: int, t: int) -> bool:
        """Check if tiling dim to t is valid."""
        if t >= dim:
            return False
        if exact_only and dim % t != 0:
            return False
        return True

    if transpose and tileable_nd >= 2:
        # Square tile for both last-2 tileable dims
        ax_last = tileable_nd - 1
        ax_second = tileable_nd - 2
        min_dim = min(ref_shape[ax_last], ref_shape[ax_second])
        t = _pallas_tile_size(min_dim, max(_align(ax_last), _align(ax_second)))

        if _can_tile_ax(ref_shape[ax_second], t):
            tile[ax_second] = t
            axis_to_grid[ax_second] = len(grid_parts)
            grid_parts.append(ref_shape[ax_second] // t)

        if _can_tile_ax(ref_shape[ax_last], t):
            tile[ax_last] = t
            axis_to_grid[ax_last] = len(grid_parts)
            grid_parts.append(ref_shape[ax_last] // t)
    else:
        # Second-to-last tileable dim (added first so it becomes grid dim 0)
        if tileable_nd >= 2:
            ax = tileable_nd - 2
            t = _pallas_tile_size(ref_shape[ax], _align(ax))
            if _can_tile_ax(ref_shape[ax], t):
                tile[ax] = t
                axis_to_grid[ax] = len(grid_parts)
                grid_parts.append((ref_shape[ax] + t - 1) // t)

        # Last tileable dim
        if tileable_nd >= 1:
            ax = tileable_nd - 1
            t = _pallas_tile_size(ref_shape[ax], _align(ax))
            if _can_tile_ax(ref_shape[ax], t):
                tile[ax] = t
                axis_to_grid[ax] = len(grid_parts)
                grid_parts.append((ref_shape[ax] + t - 1) // t)

    grid = tuple(grid_parts) if grid_parts else (1,)
    return tuple(tile), grid, axis_to_grid


def pallas_make_block_spec(
    buf_shape: tuple[int, ...],
    ref_shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    axis_to_grid: dict[int, int],
    n_grid: int,
    swap_last_two: bool = False,
    is_output: bool = False,
) -> Any:
    """Build a ``pl.BlockSpec`` for *buf_shape* given tiling of *ref_shape*.

    Lower-ndim buffers are right-aligned with the reference shape (numpy
    broadcast rules).  Dimensions that match a tiled reference dimension
    are tiled; broadcast dimensions (size 1 or absent) are kept full.

    When *buf_nd > ref_nd* (reduction inputs), we find an alignment offset
    so the ref dims map into the buffer.  Extra dims are kept at full size
    with index 0 in the index_map.

    When *swap_last_two* is True, the last two buffer dims are swapped
    relative to the reference: ref axis -2 maps to buf axis -1 and vice versa.

    When *is_output* is True and *buf_nd < ref_nd*, left-alignment is used
    as a fallback (for reduction outputs whose trailing dims were reduced).
    """
    from jax.experimental import (  # pyrefly: ignore [import-error, missing-import]
        pallas as pl,
    )

    buf_nd = len(buf_shape)
    ref_nd = len(ref_shape)

    if buf_nd == 0:
        # Scalar — untouched regardless of grid shape.
        return pl.BlockSpec((), _make_index_map([], buf_nd, n_grid))

    bs = list(buf_shape)
    tiled_pairs: list[tuple[int, int]] = []

    if buf_nd > ref_nd:
        # Reduction input: find alignment offset k where ref dims map into buf.
        align_k = 0
        for k in range(buf_nd - ref_nd + 1):
            ok = True
            for i in range(ref_nd):
                if ref_shape[i] == 1:
                    continue
                if buf_shape[k + i] != ref_shape[i]:
                    ok = False
                    break
            if ok:
                align_k = k
                break

        for ref_ax, grid_dim in axis_to_grid.items():
            buf_ax = align_k + ref_ax
            if 0 <= buf_ax < buf_nd and buf_shape[buf_ax] == ref_shape[ref_ax]:
                bs[buf_ax] = tile_shape[ref_ax]
                tiled_pairs.append((buf_ax, grid_dim))

    elif swap_last_two and buf_nd >= 2 and ref_nd >= 2:
        # Transposed buffer: last-2 dims of buf are swapped vs ref.
        for ref_ax, grid_dim in axis_to_grid.items():
            # Map ref axis to buf axis with swap on the last two
            if ref_ax == ref_nd - 2:
                buf_ax = buf_nd - 1
            elif ref_ax == ref_nd - 1:
                buf_ax = buf_nd - 2
            else:
                buf_ax = ref_ax - (ref_nd - buf_nd)

            if 0 <= buf_ax < buf_nd and buf_shape[buf_ax] == ref_shape[ref_ax]:
                bs[buf_ax] = tile_shape[ref_ax]
                tiled_pairs.append((buf_ax, grid_dim))

    else:
        # Standard right-alignment, with left-alignment fallback for
        # reduction outputs (e.g. sum(dim=-1) on (10,10) → (10,)).
        for ref_ax, grid_dim in axis_to_grid.items():
            buf_ax = ref_ax - (ref_nd - buf_nd)
            if 0 <= buf_ax < buf_nd and buf_shape[buf_ax] == ref_shape[ref_ax]:
                bs[buf_ax] = tile_shape[ref_ax]
                tiled_pairs.append((buf_ax, grid_dim))
            elif (
                is_output
                and buf_nd < ref_nd
                and 0 <= ref_ax < buf_nd
                and buf_shape[ref_ax] == ref_shape[ref_ax]
            ):
                # Left-aligned match for output: buf dim i matches ref dim i
                # (reduction output whose trailing dims were reduced away)
                bs[ref_ax] = tile_shape[ref_ax]
                tiled_pairs.append((ref_ax, grid_dim))

    return pl.BlockSpec(
        tuple(bs),
        _make_index_map(tiled_pairs, buf_nd, n_grid, swap_last_two=swap_last_two),
    )


def _make_index_map(
    tiled_pairs: list[tuple[int, int]],
    buf_nd: int,
    n_grid: int,
    swap_last_two: bool = False,
) -> Any:
    """Return an index_map callable for ``pl.BlockSpec``.

    *tiled_pairs* is a list of ``(buf_axis, grid_dim)`` indicating which
    buffer axes receive a grid index.  All other axes return 0 (full block).

    When *swap_last_two* is True the grid args for the last two buffer dims
    are swapped so that the tile iteration follows the transposed layout.

    All returned values are explicitly ``jnp.int32`` so that TPU Mosaic
    lowering (which rejects 64-bit types) works when ``jax_enable_x64`` is
    active.  The casts are created inside the function body (not captured)
    to satisfy JAX's "index_map must not capture constants" rule.
    """
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    # Pre-build the mapping so the returned lambda is a plain lookup.
    mapping = dict(tiled_pairs)

    if n_grid == 0 or (n_grid == 1 and not mapping):
        return lambda _i: tuple(jnp.int32(0) for _ in range(buf_nd))

    def index_map(*grid_args):
        return tuple(
            jnp.int32(grid_args[mapping[d]]) if d in mapping else jnp.int32(0)
            for d in range(buf_nd)
        )

    return index_map
