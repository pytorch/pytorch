# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import functools
import typing
from enum import auto, Enum

import torch
from torch.utils._triton import has_triton_package


# The following maximums only apply to runtime autotuning, when using FixedTritonConfig one may see larger values
# NOTE: if these fail asserts submit a PR to increase them
TRITON_MAX_BLOCK = {
    "X": 8192 if torch.version.hip else 4096,
    "Y": 1024,
    "Z": 1024,
    "R0_": 4096 * 16,  # * 16 is multi-kernel only
    "R1_": 2048 * 16,  # * 16 is multi-kernel only
}
TRITON_MAX_RSPLIT = 64
TRITON_MAX_TENSOR_NUMEL = 1 << 20
TRITON_DOT_MIN_BLOCK = 16
TRITON_DEFAULT_BLOCK_SIZES = {
    "XBLOCK": 128,
    "YBLOCK": 1,
    "ZBLOCK": 1,
    "R0_BLOCK": 1,
}
TRITON_DEFAULT_RSPLIT = 1
TRITON_DEFAULT_RSPLIT_SIZE = 1


def native_matmul_block_numel(
    kwargs: typing.Mapping[str, int], r0_block: int | None = None
) -> int:
    return (
        kwargs.get("XBLOCK", 1)
        * kwargs.get("YBLOCK", 1)
        * kwargs.get("ZBLOCK", 1)
        * (kwargs.get("R0_BLOCK", 1) if r0_block is None else r0_block)
    )


def native_matmul_persistent_rblock(r0_block: int) -> int:
    return max(r0_block, TRITON_DOT_MIN_BLOCK)


class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3


class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1


# Define `AttrsDescriptorWrapper` function with clear conditional handling
if has_triton_package():
    import triton
    import triton.backends.compiler
    import triton.compiler.compiler

    if hasattr(triton.backends.compiler, "AttrsDescriptor"):
        # Triton 3.2.0 - the second implementation
        from triton.backends.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "tt.divisibility": divisible_by_16,
                "tt.equal_to": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            res = AttrsDescriptor.from_dict(
                {"arg_properties": kwargs, "cls": AttrsDescriptor.__name__}
            )
            if res.property_values["tt.divisibility"] != 16:
                raise AssertionError(
                    f"Expected tt.divisibility == 16, got {res.property_values['tt.divisibility']}"
                )
            if res.property_values["tt.equal_to"] != 1:
                raise AssertionError(
                    f"Expected tt.equal_to == 1, got {res.property_values['tt.equal_to']}"
                )
            return res

    elif hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        # Triton 3.0.0 - the original implementation
        from triton.compiler.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            # Prepare the arguments for AttrsDescriptor
            kwargs = {
                "divisible_by_16": divisible_by_16,
                "equal_to_1": equal_to_1,
            }

            # Instantiate AttrsDescriptor with the prepared arguments
            return AttrsDescriptor(**kwargs)

    else:
        # Triton in 2025:
        # note: there's also a range of triton commits not currently supported
        # from ~Dec 9, 2024 to Jan 1 2025, in which AttrsDescriptors are still
        # used, but the contents are different.

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            # pyrefly: ignore [not-iterable]
            # Build attr dict merging divisibility and pointer_range per arg index,
            # since a single arg can carry both attributes.
            result = {(x,): [["tt.divisibility", 16]] for x in (divisible_by_16 or ())}
            for x in pointer_range_32 or ():
                key = (x,)
                if key in result:
                    result[key].append(["tt.pointer_range", 32])
                else:
                    result[key] = [["tt.pointer_range", 32]]
            return result

else:
    # Define a namedtuple as a fallback when AttrsDescriptor is not available
    AttrsDescriptorWrapper = collections.namedtuple(  # type: ignore[no-redef, name-match]
        # pyrefly: ignore [invalid-argument]
        "AttrsDescriptor",
        ["divisible_by_16", "equal_to_1", "pointer_range_32"],
        defaults=[(), (), ()],
    )


class HeuristicType(Enum):
    PERSISTENT_REDUCTION = auto()
    POINTWISE = auto()
    REDUCTION = auto()
    SPLIT_SCAN = auto()
    TEMPLATE = auto()
    USER_AUTOTUNE = auto()
    FIXED = auto()


class AutotuneHint(Enum):
    ONE_ELEMENT_PER_THREAD = 0

    # Triton codegen tries to codegen set of AutotuneHints.
    # Enum.__repr__ looks like "<AutotuneHint.ELEMENTS_PER_WARP_32: 0>""
    # which isn't valid python.
    # Enum.__str__ will just return "AutotuneHint.ELEMENTS_PER_WARP_32".
    __repr__ = Enum.__str__


class DeviceProperties(typing.NamedTuple):
    """Copy device properties into a data structure not requiring torch to be imported"""

    type: str  # type: ignore[assignment]
    index: int  # type: ignore[assignment]
    multi_processor_count: int
    cc: int
    major: int | None = None
    regs_per_multiprocessor: int | None = None
    max_threads_per_multi_processor: int | None = None
    max_threads_per_block: int | None = None
    warp_size: int | None = None

    @property
    def warp_size_or_default(self) -> int:
        if self.warp_size is not None:
            return self.warp_size
        if self.type in ("cuda", "hip"):
            raise RuntimeError(f"{self.type} device properties must report warp_size")
        return 32

    @classmethod
    @functools.cache
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.get_device_properties(device)
        try:
            multi_processor_count = props.multi_processor_count
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            elif device_type == "mtia":
                multi_processor_count = 64
            else:
                raise
        return cls(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=device_interface.get_compute_capability(device),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
            warp_size=getattr(props, "warp_size", None),
        )


def get_warp_size(device) -> int:
    """Return the wave/warp size in threads for the given device.

    Reads from torch.cuda.get_device_properties(device).warp_size via the cached
    DeviceProperties.create(). Correct on both AMD (64 for CDNA/gfx9, 32 for
    RDNA/gfx10+) and NVIDIA (always 32). Missing cuda/hip warp_size metadata is
    treated as an error rather than silently falling back.
    """
    return DeviceProperties.create(device).warp_size_or_default


class TritonMeta(typing.TypedDict, total=False):
    """Metadata bag threaded from Triton codegen into the runtime launcher.

    total=False because the key set is populated incrementally across codegen
    (signature/device/constants/configs first, then backend/ROCm/tlx extras)
    and the whole bag is forwarded verbatim to external Triton APIs, which
    tolerate and ignore keys they do not recognize. `device` is typed as the
    codegen-time DeviceProperties; CachingAutotuner rewrites it to the integer
    device index before reaching Triton, and the runtime read sites that expect
    that int narrow it with an explicit cast.
    """

    signature: dict[str, typing.Any]
    device: DeviceProperties
    device_type: str
    constants: dict[str, typing.Any]
    configs: list[typing.Any]
    native_matmul: bool
    launch_cooperative_grid: bool
    enable_fp_fusion: bool
    launch_pdl: bool
    disable_ftz: bool
    matrix_instr_nonkdim: int
    waves_per_eu: int
    kpack: int
    restore_value: tuple[str, ...]
    reset_to_zero: tuple[str, ...]
    backend_options: dict[str, typing.Any]


class InductorMeta(typing.TypedDict, total=False):
    """The inductor kernel-config / heuristics metadata bag.

    Produced on the codegen side (TritonKernel.inductor_meta_common /
    inductor_meta_per_kernel plus the codegen_kernel literal) and consumed by
    the runtime autotuning machinery in triton_heuristics.py,
    coordinate_descent_tuner.py, and autotune_cache.py. Every key is optional
    (total=False): consumers read via .get(...) with defaults, and several keys
    are injected only on specific paths (reductions, combo kernels, fixed grids).
    Dynamically keyed nested bags (e.g. combo_grid_meta) stay typed dict[str,
    Any] because their keys are computed at runtime. The codegen-side producers
    still build this bag as a plain dict[str, Any], so the write side is not yet
    checked against this TypedDict; typing the producers is left as a follow-up.
    """

    # Global inductor config snapshot (inductor_meta_common / inductor_meta_from_config)
    backend_hash: str | None
    assert_indirect_indexing: bool
    autotune_local_cache: bool
    autotune_pointwise: bool
    autotune_remote_cache: bool | None
    bundled_autotune_remote_cache: bool | None
    force_disable_caches: bool
    dynamic_scale_rblock: bool
    incremental_autotune: bool
    max_autotune: bool
    max_autotune_pointwise: bool
    min_split_scan_rblock: int
    spill_threshold: int
    store_cubin: bool
    deterministic: bool
    batch_invariant: bool
    force_filter_reduction_configs: bool
    mix_order_reduction_allow_multi_stages: bool
    dynamic_disable_pipelining: bool
    are_deterministic_algorithms_enabled: bool
    is_hip: bool | None
    is_fbcode: bool
    profile_bandwidth: bool
    profile_bandwidth_regex: str
    profile_bandwidth_output: str
    profile_bandwidth_with_do_bench_using_profiling: bool
    coordinate_descent_tuning: bool
    coordinate_descent_search_radius: int
    coordinate_descent_check_all_directions: bool

    # Per-kernel metadata (inductor_meta_per_kernel)
    no_x_dim: bool
    atomic_add_found: bool
    num_load: int
    num_store: int
    num_reduction: int
    autotune_hints: typing.Any
    RSPLIT_SIZE: int
    has_loadstore_with_contiguous_rdim: bool
    tma_min_block_sizes: dict[str, int]
    host_tma_descriptor_args: dict[str, dict[str, typing.Any]]
    tiling_scores: typing.Any
    min_xblock: int
    min_rblock: int
    persistent_reduction: bool
    native_matmul_persistent_rblock: int
    add_persistent_rblock: bool
    max_persistent_rblock: int
    kernel_num_gb: float
    kernel_flop: int

    # Kernel identity / launch (codegen_kernel literal)
    grid_type: str
    kernel_name: str
    mutated_arg_names: typing.Any
    optimize_mem: bool

    # Injected at runtime (CachingAutotuner, reduction heuristics, combo kernels)
    warp_size: int | None
    max_threads_per_block: int | None
    reduction_hint: typing.Any
    combo_grid_meta: dict[str, typing.Any]
    combo_tuning_groups: typing.Any
    combo_coordesc_field_order: list[str]
    combo_coordesc_field_limits: dict[str, int]
    combo_warp_stage_candidates: typing.Any
    extra_launcher_args: typing.Any
    fixed_grid: typing.Any
    precomputed_grids: typing.Any
    config_args: typing.Any
    use_fast_triton_launcher: bool


class HalideInputSpec(typing.NamedTuple):
    ctype: str
    name: str
    shape: list[str] | None = None
    stride: list[str] | None = None
    offset: str | None = None
    alias_of: str | None = None

    def bindings_type(self) -> str:
        if self.ctype in ("at::Half*", "at::BFloat16*"):
            return "uint16_t*"  # half not defined
        return self.ctype

    def halide_type(self) -> str:
        if self.ctype == "at::Half*":
            return "halide_type_t(halide_type_float, 16)"  # half not defined
        if self.ctype == "at::BFloat16*":
            return "halide_type_t(halide_type_bfloat, 16)"  # half not defined
        return f"halide_type_of<{self.ctype.replace('*', '')}>()"

    def is_scalar(self) -> bool:
        return self.shape is None

    def is_buffer(self) -> bool:
        return self.shape is not None


class HalideMeta(typing.NamedTuple):
    argtypes: list[HalideInputSpec]
    target: str
    scheduler: str | None = None
    scheduler_flags: dict[str, int | str] | None = None
    cuda_device: int | None = None

    def args(self) -> list[str]:
        """Command line args to pass to halide generator"""
        args = [f"target={self.target}"]
        if self.scheduler:
            args.append(f"autoscheduler={self.scheduler}")
        if self.scheduler_flags:
            if not self.scheduler:
                raise AssertionError("scheduler_flags requires scheduler to be set")
            for k, v in self.scheduler_flags.items():
                args.append(f"autoscheduler.{k}={v}")
        return args

    def is_cuda(self) -> bool:
        return self.cuda_device is not None
