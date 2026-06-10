# mypy: allow-untyped-defs
import atexit
import functools
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch._dynamo.utils import counters, identity
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    context_add_strides,
    context_add_using_tf32,
    mm_operations,
)
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
from torch._inductor.remote_gemm_autotune_cache import gen_best_config
from torch._inductor.virtualized import ops, V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.functional import ScalingType  # type: ignore[attr-defined]
from torch.torch_version import TorchVersion
from torch.utils._ordered_set import OrderedSet

from .. import config as inductor_config, distributed_autotune
from ..codegen.common import WorkspaceArg, WorkspaceZeroMode
from ..codegen.cutlass.gemm_template import CUTLASS2xGemmTemplate, CUTLASS3xGemmTemplate
from ..codegen.rocm.ck_tile_universal_gemm_template import CKTileGemmTemplate
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from ..codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from ..ir import Buffer, ChoiceCaller, is_triton, Layout
from ..kernel_inputs import MMKernelInputs
from ..lowering import (
    empty_strided,
    lowerings,
    make_pointwise,
    make_reduction,
    register_lowering,
    transform_args,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    KernelTemplate,
    realize_inputs,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    ceildiv,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_ck_tile_gemm_template,
    use_cpp_gemm_template,
    use_cutlass_template,
    use_decompose_k_choice,
    use_nv_universal_gemm_template,
    use_triton_blackwell_tma_template,
    use_triton_scaling_template,
    use_triton_template,
    use_triton_tma_template,
)
from .mm_common import (
    addmm_epilogue,
    _is_static_problem,
    load_kernel_template,
    mm_args,
    mm_grid,
    persistent_mm_grid,
    use_native_matmul,
)


try:
    import triton

    triton_version = TorchVersion(triton.__version__)
    has_triton = True
except ImportError:
    triton_version = TorchVersion("0.0.0")
    has_triton = False

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# StreamK configuration
ENABLE_STREAMK = os.environ.get("TORCHINDUCTOR_ENABLE_STREAMK", "0") == "1"
STREAMK_AUTOTUNE = os.environ.get("TORCHINDUCTOR_STREAMK_AUTOTUNE", "0") == "1"
STREAMK_ONLY = os.environ.get("TORCHINDUCTOR_STREAMK_ONLY", "0") == "1"
STREAMK_DEBUG = os.environ.get("TORCHINDUCTOR_STREAMK_DEBUG", "0") == "1"
STREAMK_GRID_POLICY = os.environ.get(
    "TORCHINDUCTOR_STREAMK_GRID_POLICY", "autotune"
).lower()
STREAMK_REFERENCE_CONFIG_PATH = os.environ.get(
    "TORCHINDUCTOR_STREAMK_REFERENCE_CONFIG_PATH", ""
)


def _normalize_streamk_grid_policy(policy: str) -> str:
    if policy in {"auto", "autotune", "auto_tune"}:
        return "autotune"
    if policy in {"manual", "adapter", "full"}:
        return "manual"
    if policy != "origami":
        streamk_log_info(
            f"Unknown TORCHINDUCTOR_STREAMK_GRID_POLICY={policy!r}; using origami"
        )
    return "origami"


def refresh_streamk_env():
    global ENABLE_STREAMK, STREAMK_AUTOTUNE, STREAMK_ONLY, STREAMK_DEBUG, STREAMK_GRID_POLICY, STREAMK_REFERENCE_CONFIG_PATH
    ENABLE_STREAMK = os.environ.get("TORCHINDUCTOR_ENABLE_STREAMK", "0") == "1"
    STREAMK_AUTOTUNE = os.environ.get("TORCHINDUCTOR_STREAMK_AUTOTUNE", "0") == "1"
    STREAMK_ONLY = os.environ.get("TORCHINDUCTOR_STREAMK_ONLY", "0") == "1"
    STREAMK_DEBUG = os.environ.get("TORCHINDUCTOR_STREAMK_DEBUG", "0") == "1"
    STREAMK_GRID_POLICY = os.environ.get(
        "TORCHINDUCTOR_STREAMK_GRID_POLICY", "autotune"
    ).lower()
    STREAMK_REFERENCE_CONFIG_PATH = os.environ.get(
        "TORCHINDUCTOR_STREAMK_REFERENCE_CONFIG_PATH", ""
    )
    _clear_streamk_caches()

def streamk_log_info(msg):
    log.info("[StreamKInfo] %s", msg)
def streamk_log_debug(msg):
    if STREAMK_DEBUG:
        log.info("[StreamKDebug] %s", msg)


def _streamk_dtype_key(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _streamk_reference_shape_keys(M, N, K, dtype: Any) -> tuple[str, ...]:
    dtype_key = _streamk_dtype_key(dtype)
    return (
        f"{M},{N},{K},{dtype_key}",
        f"{M}x{N}x{K}:{dtype_key}",
        f"{M},{N},{K}",
        f"{M}x{N}x{K}",
    )


@functools.lru_cache(maxsize=4)
def _load_streamk_reference_configs(path: str) -> dict[str, dict[str, Any]]:
    """Load debug-only Stream-K reference configs exported from old benchmark logs."""
    if not path:
        return {}

    try:
        with open(path) as f:
            payload = json.load(f)
    except OSError as exc:
        streamk_log_info(f"Failed to read StreamK reference configs {path!r}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        streamk_log_info(f"Failed to parse StreamK reference configs {path!r}: {exc}")
        return {}

    reference_configs: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict) and isinstance(payload.get("configs"), list):
        for entry in payload["configs"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
                continue
            dtype_key = entry.get("dtype", "")
            keys = [
                f"{entry.get('M')},{entry.get('N')},{entry.get('K')},{dtype_key}",
                f"{entry.get('M')}x{entry.get('N')}x{entry.get('K')}:{dtype_key}",
                f"{entry.get('M')},{entry.get('N')},{entry.get('K')}",
                f"{entry.get('M')}x{entry.get('N')}x{entry.get('K')}",
            ]
            for key in keys:
                if "None" not in key:
                    reference_configs[key] = dict(entry["config"])
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                reference_configs[key] = dict(value.get("config", value))

    streamk_log_debug(
        f"loaded {len(reference_configs)} StreamK reference config keys from {path}"
    )
    return reference_configs


if ENABLE_STREAMK or STREAMK_AUTOTUNE or STREAMK_DEBUG or STREAMK_ONLY:
    log.info(
        "[StreamK] module loaded: enabled=%s autotune=%s streamk_only=%s debug=%s grid_policy=%s reference_config=%s",
        ENABLE_STREAMK,
        STREAMK_AUTOTUNE,
        STREAMK_ONLY,
        STREAMK_DEBUG,
        STREAMK_GRID_POLICY,
        bool(STREAMK_REFERENCE_CONFIG_PATH),
    )

def log_choices_summary(choices, problem_desc):
    """Log summary of all choices for debugging"""
    if STREAMK_DEBUG:
        streamk_log_debug(f"Choice summary for {problem_desc}:")
        choice_types = {}
        for choice in choices:
            choice_name = getattr(choice, 'name', str(type(choice).__name__))
            choice_types[choice_name] = choice_types.get(choice_name, 0) + 1

        for choice_type, count in sorted(choice_types.items()):
            streamk_log_debug(f"  - {choice_type}: {count} configs")
        streamk_log_debug(f"Total choices: {len(choices)}")


def _safe_even_k_check(k, block_k):
    """Safely check if K is evenly divisible by block_k, handling symbolic variables"""
    try:
        return k % block_k == 0
    except (TypeError, AttributeError):
        # Symbolic variable - assume it might not be even
        return False


@dataclass(frozen=True)
class OrigamiHardwareInfo:
    n_cu: int
    num_xcd: int


@functools.lru_cache(maxsize=8)
def _get_origami_hardware_info(device_index: int = 0) -> OrigamiHardwareInfo:
    import origami
    hardware = origami.get_hardware_for_device(device_index)
    return OrigamiHardwareInfo(
        n_cu=hardware.N_CU,
        num_xcd=max(1, getattr(hardware, "NUM_XCD", 1)),
    )


@functools.lru_cache(maxsize=8)
def _get_origami_hardware(device_index: int = 0):
    import origami
    return origami.get_hardware_for_device(device_index)


def _get_hardware_chiplet_count():
    """Get actual hardware chiplet count using cached Origami metadata."""
    try:
        info = _get_origami_hardware_info(0)
        streamk_log_debug(f"hardware detection: num_xcd={info.num_xcd}")
        return info.num_xcd
    except (ImportError, AttributeError) as e:
        streamk_log_debug(f"hardware detection failed: {e}; defaulting to 1 chiplet")
        return 1


class StreamKOrigamiSelector:
    """Origami-based selector for StreamK configuration."""

    _MODERN_ORIGAMI_API = (
        "dim3_t",
        "config_t",
        "problem_t",
        "select_config",
        "select_grid_size",
        "select_reduction",
        "select_workgroup_mapping",
        "grid_selection_t",
        "transpose_t",
        "string_to_datatype",
    )
    _LEGACY_ORIGAMI_API = (
        "select_best_macro_tile_size",
        "select_best_wgm",
        "string_to_datatype",
    )

    # Dtype to string mapping (from tritonBLAS)
    dtype_to_str = {
        torch.float32: "f32",
        torch.complex64: "c32",
        torch.complex128: "c64",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.int32: "i32",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
        torch.float8_e5m2: "f8",
        torch.float8_e4m3fn: "f8",
    }
    # Add FP8 FNUZ variants if available
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"

    def __init__(self, M, N, K, a_dtype, b_dtype, c_dtype, device):
        import origami

        self._log_origami_api_support(origami)
        self.origami_api = self._detect_origami_api(origami)

        streamk_log_debug(f"initializing origami selector for {M}x{N}x{K}")
        self.M = M
        self.N = N
        self.K = K
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.device = device

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        self.hardware = _get_origami_hardware(device_index)
        self.num_sms = self.hardware.N_CU
        self._arch_name = self._get_arch_name(device)
        streamk_log_debug(
            f"using {self.origami_api} origami hardware: device={device_index} "
            f"arch={self._arch_name} num_sms={self.num_sms} "
            f"num_xcd={getattr(self.hardware, 'NUM_XCD', 'unknown')}"
        )

        # Initialize configuration ranges (from tritonBLAS)
        self.block_mn_range = [16, 32, 64, 128, 256]
        self.block_k_range = [16, 32, 64, 128, 256, 512]

        # Get element sizes and infer MI dimensions
        self.element_size_A = self._get_dtype_bits(a_dtype)
        self.element_size_B = self._get_dtype_bits(b_dtype)
        self.element_size_out = self._get_dtype_bits(c_dtype)

        # Set MI dtype - use input dtype for matrix instruction type
        input_dtype_for_mi = a_dtype if self._get_dtype_bits(a_dtype) <= self._get_dtype_bits(b_dtype) else b_dtype
        self.mi_dtype = self.dtype_to_str.get(input_dtype_for_mi, self.dtype_to_str.get(c_dtype))

        # Infer Matrix Instruction Dimensions (modern Origami config_t API)
        self.MI_dim = self._infer_matrix_instruction_dimensions(origami)

        # StreamK grid constants (from tritonBLAS)
        self.split_factors = [8, 6, 4, 3, 2, 1]
        self.tile_fractions = [0.0, 1.0/2.0, 1.0/8.0, 1.0/5.0, 1.0/4.0, 1.0/3.0]
        self.max_workspace = 128 * 1024 * 1024

        if self.origami_api == "modern":
            self.problem = self._make_problem(origami)
            self.configs = self._generate_default_configs(origami)
            self.result = origami.select_config(self.problem, self.hardware, self.configs)
            selected = self.result.config.mt
            self.reduction_strategy = origami.select_reduction(
                self.problem,
                self.hardware,
                self.result.config,
                self.result.config.grid_selection,
            )
            streamk_log_debug(
                f"modern origami selected: block_m={selected.m} "
                f"block_n={selected.n} block_k={selected.k} "
                f"occupancy={self.result.config.occupancy} "
                f"reduction={self.reduction_strategy}"
            )
            self.grid = origami.select_grid_size(
                self.problem,
                self.hardware,
                self.result.config,
                self.result.config.grid_selection,
                self.hardware.N_CU,
            )
            self.xcc_workgroup_mapping, self.workgroup_mapping = (
                self._select_workgroup_mapping(origami)
            )
            self.config = (selected.m, selected.n, selected.k, self.workgroup_mapping)
        else:
            self.reduction_strategy = None
            self.config = self._select_legacy_config(origami)
            self.grid = self._compute_streamk_grid()
            self.xcc_workgroup_mapping = getattr(self.hardware, "NUM_XCD", 1)
            self.workgroup_mapping = self.config[3]

        streamk_log_debug(
            f"final selector config: block_m={self.config[0]} "
            f"block_n={self.config[1]} block_k={self.config[2]} "
            f"group_m={self.config[3]} grid={self.grid}"
        )

    @classmethod
    def _log_origami_api_support(cls, origami_module):
        module_path = getattr(origami_module, "__file__", "<builtin>")
        names = cls._MODERN_ORIGAMI_API + tuple(
            name for name in cls._LEGACY_ORIGAMI_API if name not in cls._MODERN_ORIGAMI_API
        )
        support = {name: hasattr(origami_module, name) for name in names}
        streamk_log_info(
            "Origami API support: "
            f"module={module_path}, "
            + ", ".join(f"{name}={value}" for name, value in support.items())
        )

    @classmethod
    def _detect_origami_api(cls, origami_module):
        if all(hasattr(origami_module, name) for name in cls._MODERN_ORIGAMI_API):
            streamk_log_info("Using Origami API: modern")
            return "modern"
        if all(hasattr(origami_module, name) for name in cls._LEGACY_ORIGAMI_API):
            streamk_log_info("Using Origami API: legacy")
            return "legacy"

        required = set(cls._MODERN_ORIGAMI_API) | set(cls._LEGACY_ORIGAMI_API)
        missing = sorted(name for name in required if not hasattr(origami_module, name))
        raise RuntimeError(
            "Stream-K requires either the modern or legacy Origami Python API; "
            f"missing symbols: {missing}."
        )

    def _get_arch_name(self, device):
        if hasattr(self.hardware, "arch") and hasattr(self.hardware.arch, "name"):
            return self.hardware.arch.name
        gcn = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        return gcn.split(":")[0] if gcn else "unknown"

    def _get_dtype_bits(self, dtype):
        """Get bits for torch dtypes"""
        try:
            return torch.finfo(dtype).bits
        except TypeError:
            return torch.iinfo(dtype).bits

    def _infer_matrix_instruction_dimensions(self, origami):
        """Infer MI dimensions based on hardware and data types."""
        mi_dim = None
        largest_bitsize = max(self.element_size_A, self.element_size_B)

        def make_mi(m, n, k):
            if self.origami_api == "modern":
                return origami.dim3_t(m, n, k)
            return (m, n, k)

        # gfx950
        if self._arch_name == "gfx950":
            if largest_bitsize == 32:
                mi_dim = make_mi(16, 16, 4)
            elif largest_bitsize == 16:
                mi_dim = make_mi(16, 16, 32)
            elif largest_bitsize <= 8:
                if self.K % 256 == 0:
                    self.block_k_range = [256]
                else:
                    self.block_k_range = [128]
                self.block_mn_range = [32, 64, 128, 256]
                mi_dim = make_mi(16, 16, 128)

        elif self._arch_name == "gfx942" or self.hardware.N_CU in [304, 80, 64, 228]:
            if largest_bitsize == 32:
                mi_dim = make_mi(16, 16, 4)
            elif largest_bitsize == 16:
                mi_dim = make_mi(16, 16, 16)
            elif largest_bitsize == 8:
                self.block_mn_range = self.block_mn_range + [512]
                self.block_k_range = self.block_k_range + [128, 256]
                mi_dim = make_mi(16, 16, 32)
            elif largest_bitsize < 8:
                raise ValueError("gfx942 doesn't support F4/F6")

        elif self._arch_name == "gfx90a" or self.hardware.N_CU == 104:
            if largest_bitsize == 32:
                mi_dim = make_mi(16, 16, 4)
            elif largest_bitsize == 16:
                mi_dim = make_mi(16, 16, 16)
            elif largest_bitsize == 8:
                raise ValueError("gfx90s doesn't support F8")
            elif largest_bitsize < 8:
                raise ValueError("gfx90s doesn't support F4/F6")

        if mi_dim is None:
            raise ValueError(
                f"No valid matrix instruction for {largest_bitsize}-bit inputs "
                f"on {self._arch_name} with N_CU={self.hardware.N_CU}"
            )

        return mi_dim

    def _select_legacy_config(self, origami):
        mi_m, mi_n, mi_k = self.MI_dim
        valid_tiles = [
            (blk_m, blk_n, blk_k, mi_m, mi_n, mi_k, 1)
            for blk_m in self.block_mn_range
            for blk_n in self.block_mn_range
            for blk_k in self.block_k_range
        ]
        results = origami.select_best_macro_tile_size(
            self.M,
            self.N,
            self.K,
            1,  # batch
            True,  # transA, matching the existing Stream-K integration convention
            False,  # transB
            self.hardware,
            valid_tiles,
            self.element_size_A,
            self.element_size_B,
            self.element_size_out,
            origami.string_to_datatype(self.mi_dtype),
            0,  # mx_block_size
            0.8,  # H_L2
            False,
            False,
            6,  # WGM
        )

        best_result = results[0]
        if self.hardware.N_CU in [304, 80, 64]:
            if best_result[1] == 256 and best_result[2] == 256:
                if len(results) > 1 and results[0][0] * 1.00 > results[1][0]:
                    best_result = results[1]

        block_m, block_n, block_k = best_result[1], best_result[2], best_result[3]
        group_m_results = origami.select_best_wgm(
            self.M,
            self.N,
            self.K,
            1,
            self.hardware,
            block_m,
            block_n,
            block_k,
            mi_m,
            mi_n,
            mi_k,
            [1, 2, 4, 6, 8],
            self.element_size_A,
            0.8,
            False,
            False,
        )
        group_m = group_m_results[1]
        streamk_log_debug(
            f"legacy origami selected: block_m={block_m} "
            f"block_n={block_n} block_k={block_k} group_m={group_m}"
        )
        return (block_m, block_n, block_k, group_m)

    def _generate_default_configs(self, origami):
        config_list = []
        for blk_m, blk_n, blk_k, occupancy in itertools.product(
            self.block_mn_range,
            self.block_mn_range,
            self.block_k_range,
            [1],
        ):
            new_config = origami.config_t()
            new_config.mt = origami.dim3_t(blk_m, blk_n, blk_k)
            new_config.mi = self.MI_dim
            new_config.occupancy = occupancy
            new_config.workspace_size = self.max_workspace
            # Stream-K partial sums are accumulated in fp32 workspace.
            new_config.workspace_size_per_elem_c = 4
            new_config.grid_selection = origami.grid_selection_t.k_split_aware
            config_list.append(new_config)
        return config_list

    def _make_problem(self, origami):
        problem = origami.problem_t()
        problem.size = origami.dim3_t(self.M, self.N, self.K)
        problem.batch = 1
        problem.a_transpose = origami.transpose_t.T
        problem.b_transpose = origami.transpose_t.N
        problem.a_dtype = origami.string_to_datatype(self.dtype_to_str[self.a_dtype])
        problem.b_dtype = origami.string_to_datatype(self.dtype_to_str[self.b_dtype])
        problem.c_dtype = origami.string_to_datatype(self.dtype_to_str[self.c_dtype])
        problem.d_dtype = problem.c_dtype
        problem.mi_dtype = origami.string_to_datatype(self.mi_dtype)
        problem.a_mx_block_size = 0
        problem.b_mx_block_size = 0
        return problem

    def _select_workgroup_mapping(self, origami):
        wg_result = origami.select_workgroup_mapping(
            self.problem,
            self.hardware,
            self.result.config,
            self.grid,
        )
        if isinstance(wg_result, tuple):
            if len(wg_result) == 3:
                _, xcc_mapping, workgroup_mapping = wg_result
            else:
                xcc_mapping, workgroup_mapping = wg_result
        else:
            xcc_mapping = wg_result.wgmxcc
            workgroup_mapping = wg_result.wgm
        return xcc_mapping, workgroup_mapping

    def _compute_streamk_grid(self):
        """Compute StreamK grid size following tritonBLAS logic"""
        BLK_M, BLK_N, BLK_K, _ = self.config

        # Calculate total tiles
        tiles_m = math.ceil(self.M / BLK_M) if hasattr(self.M, '__truediv__') else (self.M + BLK_M - 1) // BLK_M
        tiles_n = math.ceil(self.N / BLK_N) if hasattr(self.N, '__truediv__') else (self.N + BLK_N - 1) // BLK_N
        total_tiles = tiles_m * tiles_n

        # StreamK grid computation (from tritonBLAS origami.py:301)
        sk_grid = total_tiles
        iters_per_tile = max(1, math.ceil(self.K / BLK_K) if hasattr(self.K, '__truediv__') else (self.K + BLK_K - 1) // BLK_K)

        # More tiles than CUs: try fractional splits to distribute work
        if total_tiles > self.num_sms:
            virt_cu_count = self.num_sms
            min_even_tiles = total_tiles / virt_cu_count

            for frac in self.tile_fractions:
                # Compute candidate grid with rounding
                frac_grid = int((total_tiles / (min_even_tiles + frac)) + 0.5)

                # Skip if this split leaves a remainder AND workspace is too large
                if (total_tiles % frac_grid != 0 and
                    self._partial_tile_size(frac_grid) > self.max_workspace):
                    continue

                # Accept the first grid no larger than the virtual CU count
                if frac_grid <= virt_cu_count:
                    sk_grid = frac_grid
                    break

        # Fewer tiles than CUs
        elif total_tiles < self.num_sms:
            total_iters = total_tiles * iters_per_tile
            min_iters_per_cu = 4
            if total_iters >= self.num_sms * min_iters_per_cu:
                # Enough K-work: use all CUs (same as 'after' baseline)
                sk_grid = self.num_sms
            else:
                # Tiny problem: 1 WG per tile to avoid serializing tiles
                # within WGs (reference: TritonBLAS uses total_tiles)
                sk_grid = total_tiles

        # Only revert if workspace would exceed budget
        if total_tiles % sk_grid != 0:
            partial_ws = self._partial_tile_size(sk_grid)
            if partial_ws > self.max_workspace:
                sk_grid = total_tiles

        # Last wave optimization for gfx942
        if total_tiles >= self.hardware.N_CU:
            last_wave_remainder = total_tiles % self.hardware.N_CU

            if (last_wave_remainder < 128 and last_wave_remainder > 0 and
                self.hardware.N_CU in [304, 80, 64]):  # gfx942
                sk_grid = 256 if self.hardware.N_CU == 304 else 64

        return sk_grid

    def _partial_tile_size(self, sk_grid):
        """Compute partial tile size for workspace calculation"""
        BLK_M, BLK_N, _, _ = self.config
        bytes_per_elem = self.element_size_out // 8
        tile_size = BLK_M * BLK_N * bytes_per_elem
        return tile_size * sk_grid

    def get_config(self):
        """Return (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)"""
        return self.config

    def get_grid(self):
        """Return optimal StreamK grid size"""
        return self.grid


# LRU cache for origami selector following tritonBLAS pattern
@functools.lru_cache(maxsize=1024)
def _make_streamk_selector(M, N, K, a_dtype, b_dtype, c_dtype, device_type):
    """Create cached origami selector following tritonBLAS pattern"""
    selector_start = time.perf_counter()
    streamk_log_debug(f"creating selector cache entry for {M}x{N}x{K}")

    # Create a dummy device object for the selector
    device = torch.device(device_type)
    # Use the origami-based selector instead of the simple heuristic one
    origami_selector = StreamKOrigamiSelector(M, N, K, a_dtype, b_dtype, c_dtype, device)

    wrapper_start = time.perf_counter()

    # Normalize selector output into template kwargs.
    class OrigamiSelectorWrapper:
        def __init__(self, origami_selector):
            self.origami_selector = origami_selector
            self.grid = None
            self._config_candidates = None
            self._extra_autotune_labels = []
            self._reference_autotune_labels = []

        def _build_config(
            self,
            *,
            block_m,
            block_n,
            block_k,
            group_m,
            grid,
            streamk_tiles,
            num_xcds,
        ):
            return {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                "STREAMK_TILES": streamk_tiles,
                "NUM_SMS": grid,
                "EVEN_K": _safe_even_k_check(self.origami_selector.K, block_k),
                "ACC_TYPE": "tl.float32",
                "ALLOW_TF32": True,
                "CACHE_MODIFIER_A": None,  # TritonBLAS-aligned
                "CACHE_MODIFIER_B": None,  # TritonBLAS-aligned
                "CHUNK_SIZE": max(1, min(4 * 4, grid // num_xcds)),  # TritonBLAS-aligned, min 1 to avoid div-by-zero
                "NUM_XCDS": num_xcds,
                "BIAS": False,
                "INPUT_PRECISION": None,
                "OUTPUT_DTYPE_IS_INT8": False,
                "QUANTIZED": False,
                "USE_FAST_ACCUM": True,
                # Use tritonBLAS fixed settings to avoid shared memory issues
                "num_warps": 8,        # Fixed like tritonBLAS (not dynamic)
                "num_stages": 2,       # Fixed like tritonBLAS (always 2, never 3)
                "waves_per_eu": 0,     # Fixed like tritonBLAS
                "matrix_instr_nonkdim": 16,  # Fixed like tritonBLAS (mfmaInstrSize)
                "kpack": 1,            # Fixed like tritonBLAS
            }

        def _compute_config_candidates(self):
            if self._config_candidates is not None:
                return self._config_candidates

            block_m, block_n, block_k, group_m = self.origami_selector.get_config()
            origami_grid = self.origami_selector.get_grid()

            import math
            total_tiles_m = math.ceil(self.origami_selector.M / block_m)
            total_tiles_n = math.ceil(self.origami_selector.N / block_n)
            total_tiles = total_tiles_m * total_tiles_n

            iters_per_tile = math.ceil(self.origami_selector.K / block_k)
            cu_count = self.origami_selector.num_sms
            tile_work_per_sms = (total_tiles * iters_per_tile) / max(1, cu_count)
            if origami_grid > total_tiles and tile_work_per_sms >= 1:
                origami_streamk_tiles = total_tiles
            else:
                origami_streamk_tiles = 0

            manual_grid = cu_count
            min_iters_per_cu = 8
            remainder_tiles = total_tiles % cu_count
            if (
                total_tiles < cu_count
                and total_tiles * iters_per_tile >= cu_count * min_iters_per_cu
            ):
                manual_streamk_tiles = total_tiles
            elif total_tiles > cu_count:
                if (
                    remainder_tiles
                    and remainder_tiles * iters_per_tile >= cu_count * min_iters_per_cu
                ):
                    manual_streamk_tiles = remainder_tiles
                else:
                    manual_streamk_tiles = 0
            else:
                manual_streamk_tiles = 0

            streamk_log_debug("k-split policy")
            streamk_log_debug(
                f"  total_tiles={total_tiles} origami_grid={origami_grid} manual_grid={manual_grid} "
                f"iters_per_tile={iters_per_tile} tile_work_per_sms={tile_work_per_sms}"
            )
            streamk_log_debug(f"  min_iters_per_cu={min_iters_per_cu}")
            streamk_log_debug(f"  block_m={block_m} block_n={block_n} block_k={block_k}")
            streamk_log_debug(
                f"  origami_streamk_tiles={origami_streamk_tiles} "
                f"manual_streamk_tiles={manual_streamk_tiles}"
            )
            num_xcds = max(
                1,
                int(
                    getattr(
                        self.origami_selector,
                        "xcc_workgroup_mapping",
                        _get_hardware_chiplet_count(),
                    )
                ),
            )

            self._config_candidates = {
                "origami": self._build_config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    group_m=group_m,
                    grid=origami_grid,
                    streamk_tiles=origami_streamk_tiles,
                    num_xcds=num_xcds,
                ),
                "manual": self._build_config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    group_m=group_m,
                    grid=manual_grid,
                    streamk_tiles=manual_streamk_tiles,
                    num_xcds=num_xcds,
                ),
                "cu_no_split": self._build_config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    group_m=group_m,
                    grid=cu_count,
                    streamk_tiles=0,
                    num_xcds=num_xcds,
                ),
                "cu_remainder_split": self._build_config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    group_m=group_m,
                    grid=cu_count,
                    streamk_tiles=remainder_tiles if total_tiles > cu_count else 0,
                    num_xcds=num_xcds,
                ),
                "cu_all_tiles_split": self._build_config(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    group_m=group_m,
                    grid=cu_count,
                    streamk_tiles=total_tiles,
                    num_xcds=num_xcds,
                ),
            }

            if STREAMK_AUTOTUNE:
                selected_shape = (block_m, block_n, block_k, group_m)
                curated_tile_shapes = [
                    (128, 128, 64, 8),
                    (128, 256, 64, 8),
                    (64, 128, 64, 8),
                    (128, 64, 64, 8),
                    # Frequent winners from pyt-29 reference configs.
                    (256, 256, 64, 4),
                    (256, 256, 64, 6),
                    (128, 64, 128, 4),
                    (128, 256, 64, 6),
                    (128, 128, 128, 6),
                ]
                if min(self.origami_selector.M, self.origami_selector.N) <= 32:
                    curated_tile_shapes.append((16, 16, 256, 1))

                for shape_idx, (cur_block_m, cur_block_n, cur_block_k, cur_group_m) in enumerate(
                    curated_tile_shapes
                ):
                    if (cur_block_m, cur_block_n, cur_block_k, cur_group_m) == selected_shape:
                        continue

                    cur_total_tiles = math.ceil(
                        self.origami_selector.M / cur_block_m
                    ) * math.ceil(self.origami_selector.N / cur_block_n)
                    cur_grid = (
                        min(cu_count, cur_total_tiles)
                        if (cur_block_m, cur_block_n, cur_block_k, cur_group_m)
                        == (16, 16, 256, 1)
                        else cu_count
                    )
                    no_split_label = f"tile_shape_{shape_idx}_cu_no_split"
                    self._config_candidates[no_split_label] = self._build_config(
                        block_m=cur_block_m,
                        block_n=cur_block_n,
                        block_k=cur_block_k,
                        group_m=cur_group_m,
                        grid=cur_grid,
                        streamk_tiles=0,
                        num_xcds=num_xcds,
                    )
                    self._extra_autotune_labels.append(no_split_label)

                if self._extra_autotune_labels:
                    streamk_log_debug(
                        "tile-shape autotune candidates: "
                        + ", ".join(
                            f"{self._config_candidates[label]['BLOCK_M']}x"
                            f"{self._config_candidates[label]['BLOCK_N']}x"
                            f"{self._config_candidates[label]['BLOCK_K']} "
                            f"GROUP_M={self._config_candidates[label]['GROUP_M']} "
                            f"NUM_SMS={self._config_candidates[label]['NUM_SMS']} "
                            f"STREAMK_TILES={self._config_candidates[label]['STREAMK_TILES']}"
                            for label in self._extra_autotune_labels
                        )
                    )

            if STREAMK_REFERENCE_CONFIG_PATH:
                reference_configs = _load_streamk_reference_configs(
                    STREAMK_REFERENCE_CONFIG_PATH
                )
                for key in _streamk_reference_shape_keys(
                    self.origami_selector.M,
                    self.origami_selector.N,
                    self.origami_selector.K,
                    self.origami_selector.a_dtype,
                ):
                    reference_config = reference_configs.get(key)
                    if reference_config is None:
                        continue

                    try:
                        ref_block_m = int(reference_config["BLOCK_M"])
                        ref_block_n = int(reference_config["BLOCK_N"])
                        ref_block_k = int(reference_config["BLOCK_K"])
                        ref_group_m = int(reference_config.get("GROUP_M", group_m))
                        ref_grid = int(reference_config["NUM_SMS"])
                        ref_streamk_tiles = int(reference_config.get("STREAMK_TILES", 0))
                    except (KeyError, TypeError, ValueError):
                        streamk_log_info(
                            f"Invalid StreamK reference config for {key}: {reference_config}"
                        )
                        break

                    label = "reference_log"
                    config = self._build_config(
                        block_m=ref_block_m,
                        block_n=ref_block_n,
                        block_k=ref_block_k,
                        group_m=ref_group_m,
                        grid=ref_grid,
                        streamk_tiles=ref_streamk_tiles,
                        num_xcds=num_xcds,
                    )
                    for meta_key in (
                        "num_warps",
                        "num_stages",
                        "waves_per_eu",
                        "matrix_instr_nonkdim",
                        "kpack",
                    ):
                        if meta_key in reference_config:
                            config[meta_key] = reference_config[meta_key]
                    self._config_candidates[label] = config
                    self._reference_autotune_labels.append(label)
                    streamk_log_debug(
                        f"matched StreamK reference config key={key}: "
                        f"{ref_block_m}x{ref_block_n}x{ref_block_k} "
                        f"GROUP_M={ref_group_m} NUM_SMS={ref_grid} "
                        f"STREAMK_TILES={ref_streamk_tiles}"
                    )
                    break
            return self._config_candidates

        def get_config(self):
            policy = _normalize_streamk_grid_policy(STREAMK_GRID_POLICY)
            if policy == "autotune":
                policy = "origami"
            config = dict(self._compute_config_candidates()[policy])
            self.grid = config["NUM_SMS"]
            streamk_log_debug(
                f"selected grid policy={policy} num_sms={config['NUM_SMS']} "
                f"streamk_tiles={config['STREAMK_TILES']}"
            )
            streamk_log_debug(f"selector config: {config}")
            return config

        def get_autotune_configs(self):
            candidates = self._compute_config_candidates()
            candidate_labels = ["origami", "manual"]
            if STREAMK_AUTOTUNE:
                candidate_labels.append("cu_no_split")
                candidate_labels.extend(self._extra_autotune_labels)
            candidate_labels.extend(self._reference_autotune_labels)

            seen = set()
            configs = []
            for label in candidate_labels:
                config = dict(candidates[label])
                config_key = tuple(sorted(config.items()))
                if config_key in seen:
                    continue
                seen.add(config_key)
                configs.append(config)
            streamk_log_debug(
                "grid policy autotune candidates: "
                + ", ".join(
                    f"{config['BLOCK_M']}x{config['BLOCK_N']}x{config['BLOCK_K']} "
                    f"GROUP_M={config['GROUP_M']} NUM_SMS={config['NUM_SMS']} "
                    f"STREAMK_TILES={config['STREAMK_TILES']}"
                    for config in configs
                )
            )
            return configs

        def get_grid(self):
            return self.grid if self.grid is not None else self.origami_selector.get_grid()

    wrapper_end = time.perf_counter()
    selector_end = time.perf_counter()

    return OrigamiSelectorWrapper(origami_selector)


def _clear_streamk_caches() -> None:
    _get_origami_hardware.cache_clear()
    _get_origami_hardware_info.cache_clear()
    _load_streamk_reference_configs.cache_clear()
    _make_streamk_selector.cache_clear()


atexit.register(_clear_streamk_caches)


def _get_streamk_autotune_configs(
    selected_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a few nearby StreamK configs to benchmark against the Origami pick."""
    if not STREAMK_AUTOTUNE:
        return []

    base_num_warps = int(selected_config.get("num_warps", 8))
    base_waves_per_eu = int(selected_config.get("waves_per_eu", 0))

    variant_overrides = [
        {"num_warps": 4 if base_num_warps != 4 else 8},
        {"waves_per_eu": 1 if base_waves_per_eu != 1 else 2},
    ]

    seen = {tuple(sorted(selected_config.items()))}
    configs: list[dict[str, Any]] = []
    for overrides in variant_overrides:
        config = dict(selected_config)
        config.update(overrides)
        config_key = tuple(sorted(config.items()))
        if config_key in seen:
            continue
        seen.add(config_key)
        configs.append(config)

    streamk_log_debug(
        f"generated {len(configs)} additional StreamK autotune configs"
    )
    return configs

@SymbolicGridFn
def streamk_mm_grid(m, n, meta, *, cdiv, min):
    """Launch NUM_SMS workgroups so every CU is occupied.
    Phase 1 naturally handles the case where pid >= total_tiles
    by looping zero times."""
    num_sms = meta.get("NUM_SMS", 108)
    return (num_sms, 1, 1)


_STREAMK_DEF_KERNEL_WITH_WORKSPACE = r"""
{% if QUANTIZED %}
{% if BIAS %}
{{def_kernel("A", "B", "A_SCALE_PTR", "B_SCALE_PTR", "BIAS_PTR")}}
{% else %}
{{def_kernel("A", "B", "A_SCALE_PTR", "B_SCALE_PTR")}}
{% endif %}
{% else %}
{% if BIAS %}
{{def_kernel("A", "B", "BIAS_PTR")}}
{% else %}
{{def_kernel("A", "B")}}
{% endif %}
{% endif %}
"""

_STREAMK_DEF_KERNEL_WITHOUT_WORKSPACE = r"""
{% if QUANTIZED %}
{% if BIAS %}
{{def_kernel("A", "B", "A_SCALE_PTR", "B_SCALE_PTR", "BIAS_PTR")}}
{% else %}
{{def_kernel("A", "B", "A_SCALE_PTR", "B_SCALE_PTR")}}
{% endif %}
{% else %}
{% if BIAS %}
{{def_kernel("A", "B", "BIAS_PTR")}}
{% else %}
{{def_kernel("A", "B")}}
{% endif %}
{% endif %}
"""

_STREAMK_SPLIT_PHASE_SOURCE = r"""
    # ========== Phase 2: Process StreamK Tiles ==========

    # Workspace and locks are zeroed by the wrapper before each launch.  Partial
    # contributors fully publish their workspace slot before setting the lock, so
    # the tile owner does not need the workspace payload to be preinitialized.
    WORKSPACE = tl.cast(ws_ptr, tl.pointer_type(tl.float32))
    LOCKS = tl.cast(
        ws_ptr + (NUM_SMS * BLOCK_M * BLOCK_N * 4),
        tl.pointer_type(tl.int32),
    )
    rm1 = tl.arange(0, BLOCK_M)
    rn1 = tl.arange(0, BLOCK_N)
    rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_M), BLOCK_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_N), BLOCK_N)
    P_ = WORKSPACE + pid * BLOCK_M * BLOCK_N + rm1[:, None] * BLOCK_N + rn1[None, :]

    tl.assume(pid >= 0)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)

    # StreamK main loop
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile

        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        {% if BIAS %}
        bias_ = BIAS_PTR + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)
        {% endif %}

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        {% if EVEN_K %}
        if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)):
            offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        else:
            offs_a_m = rm % M
        if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)):
            offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)

        for current_iter in range(start_iter, end_iter):
            k_offset = (current_iter % iters_per_tile) * BLOCK_K
            a_k_offs = offs_k[None, :] + k_offset
            b_k_offs = offs_k[:, None] + k_offset
            a = tl.load(A + offs_a_m[:, None] * stride_am + a_k_offs * stride_ak)
            b = tl.load(B + b_k_offs * stride_bk + offs_b_n[None, :] * stride_bn)
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=acc_dtype)

        {% else %}
        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))
        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N

        for current_iter in range(start_iter, end_iter):
            global_k_offset = (current_iter % iters_per_tile) * BLOCK_K
            k_mask = global_k_offset + rk < K
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m & k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m & k_mask[None, :], other=0.0, cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n & k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n & k_mask[:, None], other=0.0, cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk
        {% endif %}

        {% if QUANTIZED %}
        rm_A_scale = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_B_scale = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        A_scale = tl.load(A_SCALE_PTR + rm_A_scale * stride_a_scale, mask=rm_A_scale < M, other=0.0)
        B_scale = tl.load(B_SCALE_PTR + rn_B_scale * stride_b_scale, mask=rn_B_scale < N, other=0.0)
        acc *= A_scale[:, None] * B_scale[None, :]
        {% endif %}

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            # Partial contributor: publish its accumulation for the tile owner.
            tl.store(P_, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.store(LOCKS + pid, 1, cache_modifier=".wt")
        else:
            # Tile owner: linearly collect every following CTA that owns the
            # remaining K-iteration chunks for this output tile.  This supports
            # non-power-of-two split factors selected by Origami.
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter

            while end < tile_iter_end and next_pid < NUM_SMS:
                while tl.load(LOCKS + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_M)
                rn1 = tl.arange(0, BLOCK_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_M), BLOCK_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_N), BLOCK_N)
                P_partner = WORKSPACE + next_pid * BLOCK_M * BLOCK_N + rm1[:, None] * BLOCK_N + rn1[None, :]
                acc += tl.load(P_partner, cache_modifier=".cv")
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            {% if BIAS %}
            acc += bias[:, None]
            {% endif %}

            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            idx_m = rm[:, None]
            idx_n = rn[None, :]
            {{store_output(("idx_m", "idx_n"), "acc", "mask", indent_width=12, val_shape=("BLOCK_M", "BLOCK_N"))}}

        start_iter = end_iter
"""

_STREAMK_TEMPLATE_SOURCE = r"""
__DEF_KERNEL__
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        return

    stride_am: tl.constexpr = {{stride("A", 0)}}
    {% set k_size = size("A", 1)|int %}
    {% if k_size >= 16 %}
    stride_ak: tl.constexpr = {{stride("A", 1)}}
    stride_bk: tl.constexpr = {{stride("B", 0)}}
    {% else %}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    {% endif %}
    stride_bn: tl.constexpr = {{stride("B", 1)}}

    {% if BIAS %}
    stride_bias: tl.constexpr = {{stride("BIAS_PTR", 0)}}
    {% endif %}
    {% if QUANTIZED %}
    stride_a_scale: tl.constexpr = {{stride("A_SCALE_PTR", 0)}}
    stride_b_scale: tl.constexpr = {{stride("B_SCALE_PTR", 0)}}
    {% endif %}

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    pid = tl.program_id(0)
    if NUM_XCDS != 1 and CHUNK_SIZE >= 1:
        pid = triton_helpers.chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES
    acc_dtype = tl.float32

    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        {% if BIAS %}
        bias_ = BIAS_PTR + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)
        {% endif %}

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        {% if EVEN_K %}
        if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)):
            offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        else:
            offs_a_m = rm % M
        if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)):
            offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            a_k_offs = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_offs = offs_k[:, None] + (k_idx * BLOCK_K)
            a = tl.load(A + offs_a_m[:, None] * stride_am + a_k_offs * stride_ak)
            b = tl.load(B + b_k_offs * stride_bk + offs_b_n[None, :] * stride_bn)
            acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=acc_dtype)
        {% else %}
        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        mask_m = rm[:, None] < M
        mask_n = rn[None, :] < N

        loop_k = tl.cdiv(K, BLOCK_K) - 1
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), mask=mask_m, other=0.0)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), mask=mask_m, other=0.0)
            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), mask=mask_n, other=0.0)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), mask=mask_n, other=0.0)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        rk_tail = loop_k * BLOCK_K + tl.arange(0, BLOCK_K)
        A_TAIL = A + rm[:, None] * stride_am + rk_tail[None, :] * stride_ak
        B_TAIL = B + rk_tail[:, None] * stride_bk + rn[None, :] * stride_bn
        if stride_ak == 1:
            A_TAIL = tl.multiple_of(A_TAIL, (1, 16))
        else:
            A_TAIL = tl.multiple_of(A_TAIL, (16, 1))
        if stride_bk == 1:
            B_TAIL = tl.multiple_of(B_TAIL, (16, 1))
        else:
            B_TAIL = tl.multiple_of(B_TAIL, (1, 16))
        a = tl.load(A_TAIL, mask=mask_m & (rk_tail[None, :] < K), other=0.0)
        b = tl.load(B_TAIL, mask=mask_n & (rk_tail[:, None] < K), other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        {% endif %}

        {% if QUANTIZED %}
        rm_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn_q = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        A_scale = tl.load(A_SCALE_PTR + rm_q * stride_a_scale, mask=rm_q < M, other=0.0)
        B_scale = tl.load(B_SCALE_PTR + rn_q * stride_b_scale, mask=rn_q < N, other=0.0)
        acc *= A_scale[:, None] * B_scale[None, :]
        {% endif %}

        {% if BIAS %}
        c = acc + bias[:, None]
        {% else %}
        c = acc
        {% endif %}

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        {{store_output(("idx_m", "idx_n"), "c", "mask", indent_width=8, val_shape=("BLOCK_M", "BLOCK_N"))}}

    {% if STREAMK_TILES != 0 %}
__STREAMK_PHASE__
    {% endif %}
"""


def _build_streamk_template_source(include_workspace_locks: bool) -> str:
    return (
        _STREAMK_TEMPLATE_SOURCE.replace(
            "__DEF_KERNEL__",
            (
                _STREAMK_DEF_KERNEL_WITH_WORKSPACE
                if include_workspace_locks
                else _STREAMK_DEF_KERNEL_WITHOUT_WORKSPACE
            ),
        ).replace(
            "__STREAMK_PHASE__",
            _STREAMK_SPLIT_PHASE_SOURCE if include_workspace_locks else "",
        )
    )


class _BaseStreamKTemplate(TritonTemplate):
    requires_workspace_locks: bool

    def __init__(self, name: str, include_workspace_locks: bool):
        super().__init__(
            name=name,
            grid=streamk_mm_grid,
            source=_build_streamk_template_source(include_workspace_locks),
            cache_codegen_enabled_for_template=True,
            prologue_loads_all_inputs=True,
        )
        self.requires_workspace_locks = include_workspace_locks

    def maybe_append_choice(self, choices, **kwargs):
        if not ENABLE_STREAMK:
            return None

        try:
            input_nodes = kwargs.get("input_nodes", ())
            prefix_args = int(kwargs.get("prefix_args", 0) or 0)
            suffix_args = int(kwargs.get("suffix_args", 0) or 0)
            named_arg_count = len(input_nodes) - prefix_args - suffix_args
            if named_arg_count < 2:
                raise ValueError(
                    f"StreamK template requires at least 2 named input nodes (A, B), got {named_arg_count}"
                )

            A_node, B_node = input_nodes[prefix_args], input_nodes[prefix_args + 1]
            bias_node = (
                input_nodes[prefix_args + 2]
                if prefix_args == 0 and named_arg_count > 2
                else None
            )
            layout = kwargs.get("layout")

            template_kwargs = dict(kwargs)
            template_kwargs.pop("input_nodes", None)
            template_kwargs.pop("layout", None)

            is_quantized = self._detect_quantization(A_node, B_node, layout)
            template_kwargs.setdefault("QUANTIZED", is_quantized)

            has_bias = bias_node is not None
            template_kwargs.setdefault("BIAS", has_bias)

            num_sms = template_kwargs.get("NUM_SMS", 108)
            block_m = template_kwargs.get("BLOCK_M", 128)
            block_n = template_kwargs.get("BLOCK_N", 128)
            streamk_tiles = int(template_kwargs.get("STREAMK_TILES", 0))
            uses_workspace_locks = streamk_tiles > 0

            if uses_workspace_locks and not self.requires_workspace_locks:
                raise ValueError(
                    f"{self.name} received STREAMK_TILES={streamk_tiles}, "
                    f"requires_workspace_locks={self.requires_workspace_locks}"
                )

            M = A_node.get_size()[0]
            N = B_node.get_size()[1]
            streamk_input_nodes = list(input_nodes)
            mutated_inputs = []
            workspace_arg = None

            if is_quantized:
                a_scale = empty_strided([M], None, dtype=torch.float32, device=layout.device)
                b_scale = empty_strided([N], None, dtype=torch.float32, device=layout.device)
                insert_idx = prefix_args + 2
                streamk_input_nodes[insert_idx:insert_idx] = [a_scale, b_scale]

            if uses_workspace_locks:
                workspace_bytes = num_sms * block_m * block_n * 4
                locks_bytes = num_sms * 4
                workspace_arg = WorkspaceArg(
                    count=workspace_bytes + locks_bytes,
                    zero_mode=WorkspaceZeroMode.ZERO_ON_CALL,
                    device=layout.device,
                    outer_name=WorkspaceArg.unique_name(),
                )
                streamk_log_debug(
                    "workspace arg bytes: workspace=%s locks=%s total=%s"
                    % (workspace_bytes, locks_bytes, workspace_bytes + locks_bytes)
                )

            streamk_input_nodes = tuple(streamk_input_nodes)
            epilogue_fn = template_kwargs.pop("epilogue_fn", identity)
            epilogue_fn_hash = template_kwargs.pop("epilogue_fn_hash", None)

            return super().maybe_append_choice(
                choices,
                input_nodes=streamk_input_nodes,
                layout=layout,
                mutated_inputs=mutated_inputs,
                workspace_arg=workspace_arg,
                epilogue_fn=epilogue_fn,
                epilogue_fn_hash=epilogue_fn_hash,
                allow_epilogue_fusion=True,
                **template_kwargs,
            )
        except Exception as e:
            streamk_log_debug(f"Failed to add StreamK choice: {e}")
            if STREAMK_DEBUG:
                import traceback

                streamk_log_debug(traceback.format_exc())
            return e

    def _detect_quantization(self, A_node, B_node, layout):
        a_dtype = A_node.get_dtype()
        b_dtype = B_node.get_dtype()
        output_dtype = layout.dtype

        quantized_dtypes = {torch.int8, torch.uint8}
        if hasattr(torch, "float8_e4m3fn"):
            quantized_dtypes.add(torch.float8_e4m3fn)
        if hasattr(torch, "float8_e5m2"):
            quantized_dtypes.add(torch.float8_e5m2)

        is_quantized = (
            a_dtype in quantized_dtypes
            or b_dtype in quantized_dtypes
            or output_dtype in quantized_dtypes
        )
        if is_quantized:
            streamk_log_debug(
                f"Detected quantized operation: A={a_dtype}, B={b_dtype}, output={output_dtype}"
            )
        return is_quantized


class StreamKTemplate(_BaseStreamKTemplate):
    def __init__(self):
        super().__init__("mm_streamk", include_workspace_locks=True)


mm_streamk_template = StreamKTemplate()


def _get_streamk_template(streamk_tiles: int) -> _BaseStreamKTemplate:
    return mm_streamk_template


def can_use_streamk(m, n, k, dtype, device):
    """Return whether StreamK is supported for this mm instance."""

    def is_symbolic(val):
        try:
            int(val)
            return False
        except (TypeError, ValueError):
            val_str = str(val)
            return (
                hasattr(val, 'is_symbol')
                or val_str.startswith('s')
                or 'Symbol' in str(type(val))
                or 'Expr' in str(type(val))
                or any(c in val_str for c in ['s', 'Symbol', 'Expr', 'sympy'])
            )

    if is_symbolic(m) or is_symbolic(n) or is_symbolic(k):
        streamk_log_info(
            f"Symbolic variables detected ({m}x{n}x{k}). StreamK is disabled for this compilation."
        )
        streamk_log_debug(f"Variable types: m={type(m)}, n={type(n)}, k={type(k)}")
        return False

    if str(device).startswith('mtia'):
        streamk_log_debug(f"MTIA device ({device}) detected; skipping StreamK for {m}x{n}x{k}")
        return False

    if not torch.cuda.is_available():
        streamk_log_info(f"CUDA not available; skipping StreamK for {m}x{n}x{k}")
        return False

    streamk_log_debug(f"StreamK supported for {m}x{n}x{k} (dtype={dtype})")
    return True


def add_streamk_mm_choices(
    choices: list[ChoiceCaller],
    kernel_inputs: MMKernelInputs,
    layout: Layout,
    m,
    n,
    k,
    mat1: Buffer,
    mat2: Buffer,
    *,
    input_nodes: tuple[Any, ...] | None = None,
    extra_template_kwargs: dict[str, Any] | None = None,
) -> int:
    """Add Origami-selected StreamK choices to the current MM choice list."""
    selector = _make_streamk_selector(
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout.dtype,
        str(layout.device),
    )
    grid_policy = _normalize_streamk_grid_policy(STREAMK_GRID_POLICY)
    if grid_policy == "autotune" and hasattr(selector, "get_autotune_configs"):
        streamk_configs = [dict(config) for config in selector.get_autotune_configs()]
        optimal_config = streamk_configs[0]
    else:
        optimal_config = selector.get_config()
        streamk_configs = [dict(optimal_config)]

    streamk_log_info(
        "Selected StreamK config: "
        f"BLOCK_M={optimal_config['BLOCK_M']}, "
        f"BLOCK_N={optimal_config['BLOCK_N']}, "
        f"BLOCK_K={optimal_config['BLOCK_K']}, "
        f"STREAMK_TILES={optimal_config['STREAMK_TILES']} "
        f"grid_policy={grid_policy}"
    )

    for base_config in list(streamk_configs):
        streamk_configs.extend(_get_streamk_autotune_configs(base_config))
    seen_configs = set()
    unique_streamk_configs = []
    for config in streamk_configs:
        config_key = tuple(sorted(config.items()))
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        unique_streamk_configs.append(config)
    streamk_configs = unique_streamk_configs
    added_before = len(choices)

    for config_idx, streamk_config in enumerate(streamk_configs):
        streamk_choice_config = dict(streamk_config)
        if extra_template_kwargs:
            streamk_choice_config.update(extra_template_kwargs)
        num_warps = streamk_choice_config.pop("num_warps", 8)
        num_stages = streamk_choice_config.pop("num_stages", 2)
        choice_label = grid_policy if config_idx == 0 else f"autotune_{config_idx}"
        streamk_template = _get_streamk_template(
            int(streamk_choice_config.get("STREAMK_TILES", 0))
        )
        streamk_log_debug(
            f"Adding StreamK competitor choice ({choice_label}) with template "
            f"{streamk_template.name}: {streamk_choice_config}, "
            f"num_warps={num_warps}, num_stages={num_stages}"
        )
        error = streamk_template.maybe_append_choice(
            choices,
            input_nodes=input_nodes
            or (kernel_inputs.nodes()[0], kernel_inputs.nodes()[1]),
            layout=layout,
            num_warps=num_warps,
            num_stages=num_stages,
            **streamk_choice_config,
        )
        if error is not None:
            streamk_log_debug(
                f"StreamK choice generation failed for {choice_label}: {error}"
            )

    added = len(choices) - added_before
    if added:
        streamk_log_info(
            f"StreamK added as competitor for {m}x{n}x{k}: {added} choices"
        )
    else:
        streamk_log_info(
            f"Failed to add StreamK competitor for {m}x{n}x{k}; using existing choices"
        )
    return added


# We define each template kernel in a separate file which is the name of the input to load_kernel_template
# (e.g. triton_mm for templates/triton_mm.py.jinja).
# If you are adding a new template, please follow that pattern and add a new file with your implementation in the templates folder.
mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=load_kernel_template("triton_mm")
    if (torch.version.hip is None) or triton_version >= "3.3.0"
    # FIXME: To get around rocm failures like https://github.com/pytorch/pytorch/actions/runs/13123783322/job/36617154943
    # The only difference between the two templates is M >= BLOCK_M and N >= BLOCK_N checking.
    # See more details in https://github.com/pytorch/pytorch/pull/146293
    else load_kernel_template("triton_mm_rocm"),
    cache_codegen_enabled_for_template=True,
    prologue_loads_all_inputs=True,
)

persistent_tma_mm_template = TritonTemplate(
    name="mm_persistent_tma",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_persistent_tma_mm"),
)

# Non-TMA Triton template for persistent MM
# used on AMD
persistent_mm_template = TritonTemplate(
    name="mm_persistent",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_persistent_mm"),
)


scaled_mm_device_tma_epilogue_scaling_template = TritonTemplate(
    name="scaled_mm_device_tma_epilogue_scaling",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_epilogue_scaled_mm"),
)


scaled_mm_device_tma_main_loop_scaling_template = TritonTemplate(
    name="scaled_mm_device_tma_main_loop_scaling",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_main_loop_scaled_mm"),
)

blackwell_ws_persistent_device_tma_mm_template = TritonTemplate(
    name="blackwell_ws_persistent_device_tma",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_blackwell_ws_persistent_device_tma_mm"),
)


# prevent duplication registration of extern functions
@functools.cache
def lazy_register_extern_choice(fn):
    return ExternKernelChoice(fn)


aten_mm = ExternKernelChoice(torch.mm, "at::mm_out", op_overload=aten.mm.out)
aten_mm_dtype = ExternKernelChoice(
    torch.mm,
    "at::mm_dtype_out",
    name="mm_dtype",
    op_overload=aten.mm.dtype_out,
)

aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.out
)

aten__int_mm = ExternKernelChoice(
    torch._int_mm, "at::_int_mm_out", op_overload=aten._int_mm.out
)

aten__sparse_semi_structured_mm = ExternKernelChoice(
    torch._sparse_semi_structured_mm,
    "at::_sparse_semi_structured_mm",
    has_out_variant=False,
    op_overload=aten._sparse_semi_structured_mm.default,
)

aten__fp8_mm = ExternKernelChoice(
    torch._scaled_mm, "at::_scaled_mm_out", op_overload=aten._scaled_mm.out
)


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if (inp.stride(0) == 0 and inp.size(0) != 0) or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


def check_supported_striding(mat_a, mat_b) -> None:
    def is_row_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[1], 1)

    def is_col_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[0], 1)

    def has_zero_dim(size) -> bool:
        return bool(
            V.graph.sizevars.statically_known_equals(size[0], 0)
            or V.graph.sizevars.statically_known_equals(size[1], 0)
        )

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


def decomposeK(a, b, k_splits):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    k_parts = k // k_splits
    B = k_splits
    a_reshaped = torch.permute(a.reshape(m, B, k_parts), (1, 0, 2))
    b_reshaped = b.reshape(B, k_parts, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    reduced_buf = torch.sum(result, 0)
    return reduced_buf.to(a.dtype)


class DecomposeKSugraphTemplate(SubgraphTemplate):
    def __init__(self):
        super().__init__(
            name="decompose_k",
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        k_split: int,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        name = f"decompose_k_mm_{k_split}_split"
        description = f"{k_split=}"

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                functools.partial(decomposeK, k_splits=k_split),
                decompositions,
            )

            return super().generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=description,
            )


decompose_k_subgraph_template = DecomposeKSugraphTemplate()


class ContiguousTemplate(SubgraphTemplate):
    def __init__(self, name: str, description: str, fn: Any):
        self.name = name
        self.description = description
        self.fn = fn
        super().__init__(
            name=name,
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                self.fn,
                decompositions,
            )

            return super().generate(
                name=self.name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=self.description,
            )


def contiguous_mm(a, b):
    return torch.mm(a, b.contiguous())


def contiguous_addmm(inp, a, b):
    return torch.addmm(inp, a, b.contiguous())


mm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_mm", "contiguous mm", contiguous_mm
)
addmm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_addmm", "contiguous addmm", contiguous_addmm
)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, out_dtype=None, *, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    refresh_streamk_env()
    if out_dtype is not None:
        input_dtype = mat1.get_dtype()
        torch._check(
            mat2.get_dtype() == input_dtype,
            lambda: "input dtypes must be the same",
        )
        torch._check(
            mat1.get_device().type in ("cuda", "xpu"),
            lambda: "out_dtype is only supported for CUDA or XPU",
        )
        torch._check(
            out_dtype == input_dtype
            or (
                out_dtype == torch.float32
                and input_dtype in (torch.float16, torch.bfloat16)
            ),
            lambda: "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs",
        )

    # Lower matmul-related operations (e.g., torch.matmul / torch.bmm / torch.addmm)
    # into native matmul IR using `ops.dot`. When we see a matmul pattern
    # (C[y, x] = A[y, r] * B[r, x]), the core idea is to emulate a broadcasted
    # multiply followed by a sum.
    #
    # For example, given `C = torch.matmul(A, B)`, this can be rewritten as:
    #
    #     Prod = A.unsqueeze(-1) * B.unsqueeze(0)
    #     C = Prod.sum(dim=1)
    #
    # Instead of explicitly using `ops.mul` and `ops.reduction("sum")`, we lower
    # these into `ops.dot` (pointwise) and `ops.reduction("dot")`. These IR nodes
    # are semantically equivalent to the `ops.mul` + `ops.reduction("sum")`
    # combination, but are lowered to `tl.dot` during the code generation phase.
    if use_native_matmul(mat1, mat2):
        mat1 = lowerings[aten.unsqueeze](mat1, -1)
        mat2 = lowerings[aten.unsqueeze](mat2, 0)
        args, kwargs = transform_args(
            args=[mat1, mat2],
            kwargs={},
            broadcast=True,
            type_promotion_kind=None,
            convert_input_to_bool=False,
        )  # Handles broadcasting the arguments

        if inductor_config.triton.codegen_upcast_to_fp32 and mat1.dtype in [
            torch.float16,
            torch.bfloat16,
        ]:

            def _to_dtype(x):
                return ops.to_dtype(x, mat1.dtype, use_compute_types=False)

            args = [make_pointwise(_to_dtype)(x) for x in args]
        mul_pointwise = make_pointwise(ops.dot)(*args)
        dot_reduction = make_reduction("dot")(mul_pointwise, 1)

        return dot_reduction

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "mm"

    # Create MMKernelInputs for standard MM at the top
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    choices: list[ChoiceCaller] = []
    static_shape, is_nonzero = _is_static_problem(layout)

    aten_handler: ExternKernelChoice = aten_mm
    aten_extra_kwargs: dict[str, Any] = {}
    if out_dtype is not None:
        aten_handler = aten_mm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides: dict[str, dict[str, Any]] = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        if aten_extra_kwargs:
            kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout, check_max_autotune=True)
    ):
        if use_decompose_k_choice(m, n, k):
            templates_to_use.append(decompose_k_subgraph_template)
        # Triton Templates typically perform very poorly for large K.
        # Its highly unlikely that if we want to use decompose_k, then
        # Triton will ever win.
        #
        # To be conservative we increase this threshold for N/M by 2.
        is_exhaustive = inductor_config.max_autotune_gemm_search_space == "exhaustive"
        if is_exhaustive or not use_decompose_k_choice(m, n, k, threshold_multiple=2):
            templates_to_use.append(mm_template)

            if use_triton_blackwell_tma_template(
                mat1, mat2, output_layout=layout, add_guards=True
            ):
                templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            elif use_triton_tma_template(
                mat1, mat2, output_layout=layout, add_guards=True
            ):
                if torch.version.hip is None:
                    templates_to_use.append(persistent_tma_mm_template)
                else:
                    templates_to_use.append(persistent_mm_template)

        templates_to_use.append(mm_contiguous_subgraph_template)

    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            "mm",
            kwarg_overrides=kwarg_overrides,
        )
    )

    streamk_only_active = False
    if out_dtype is None and static_shape and is_nonzero and ENABLE_STREAMK:
        streamk_supported = can_use_streamk(m, n, k, mat1.get_dtype(), layout.device)
        if streamk_supported:
            standard_choices = list(choices)
            if STREAMK_ONLY:
                streamk_log_info(
                    f"TORCHINDUCTOR_STREAMK_ONLY=1: using only StreamK candidates for {m}x{n}x{k}"
                )
                choices.clear()
                streamk_only_active = True

            try:
                streamk_added = add_streamk_mm_choices(
                    choices, kernel_inputs, layout, m, n, k, mat1, mat2
                )
            except Exception as exc:
                streamk_added = 0
                streamk_log_info(f"StreamK competitor setup failed: {exc}")
                if STREAMK_DEBUG:
                    import traceback

                    streamk_log_debug(traceback.format_exc())

            if STREAMK_ONLY and streamk_added == 0:
                streamk_log_info(
                    f"No StreamK choices generated for {m}x{n}x{k}; restoring standard choices"
                )
                choices[:] = standard_choices
                streamk_only_active = False
        elif STREAMK_DEBUG:
            streamk_log_info(
                f"StreamK was not added as a candidate for {m}x{n}x{k}"
            )

    if (
        out_dtype is None
        and is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("mm")
        and not streamk_only_active
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes()
        )

    if (
        out_dtype is None
        and is_nonzero
        and use_ck_gemm_template(layout, m, n, k)
        and not streamk_only_active
    ):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())
    if (
        out_dtype is None
        and is_nonzero
        and use_ck_tile_gemm_template(layout, m, n, k)
        and not streamk_only_active
    ):
        CKTileGemmTemplate.add_choices(choices, layout, kernel_inputs.nodes())

    if (
        out_dtype is None
        and is_nonzero
        and use_nv_universal_gemm_template(layout, m, n, k, mat1, mat2)
        and not streamk_only_active
    ):
        from ..codegen.nv_universal_gemm import add_nv_universal_gemm_choices

        add_nv_universal_gemm_choices(choices, layout, kernel_inputs)

    if (
        out_dtype is None
        and use_cpp_gemm_template(layout, mat1, mat2)
        and not streamk_only_active
    ):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
        )

    input_nodes = [mat1, mat2]
    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout)
        and torch._inductor.config.run_autoheuristic(name)
        and is_triton(mat1)
        and not streamk_only_active
    ):
        always_included = []
        if use_aten_gemm_kernels():
            always_included.append("extern_mm")
        num_choices_before_extra_configs = len(choices)
        choices.extend(
            V.choices.get_template_configs(
                # TODO(coconutruben): remove once we deprecate ah
                # mm-extra is a hack to keep the ah functionality alive
                # while we transition to the unified kwargs retrieval
                kernel_inputs,
                [mm_template],
                "mm-ah",
            )
        )

        # using AutoHeuristic for ranking
        ah_choices = mm_autoheuristic(
            mat1,
            mat2,
            m,
            n,
            k,
            choices,
            name,
            input_nodes,
            mm_operations(),
            None,
            top_k=10,
            always_included=always_included,
        )
        if not torch._inductor.config.collect_autoheuristic(name):
            # if we are collecting data, we do not want to modify choices
            if ah_choices is not None and len(ah_choices) > 0:
                # the order in which autoheuristic returns choices is not the same as
                # as the order of choices, which affects things like epilogue fusion.
                # once epilogue fusion benchmarks choices in sorted order, I think we can
                # just use the order returned by autoheuristic
                choices = [choice for choice in choices if choice in ah_choices]
            else:
                choices = choices[:num_choices_before_extra_configs]

    if out_dtype is None and not streamk_only_active:
        for k in inductor_config.external_matmul:
            choices.append(
                lazy_register_extern_choice(k).bind(kernel_inputs.nodes(), layout)
            )

    best_config_future = None
    if out_dtype is None and torch._inductor.config.remote_gemm_autotune_cache:
        # Purposely not awaiting the future here - this kicks off the best config lookup at lowering time
        # The future will be awaited at scheduling time in select_algorithm.py
        best_config_future = gen_best_config(mat1, mat2)

    if box := distributed_autotune.maybe_autotune_remote(
        name, choices, kernel_inputs.nodes(), layout
    ):
        return box

    node, _ = autotune_select_algorithm(
        name,
        choices,
        kernel_inputs.nodes(),
        layout,
        best_config_future=best_config_future,
    )
    return node


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    name = "int_mm"
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._int_mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._int_mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    static_shape, is_nonzero = _is_static_problem(layout)
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)
    choices: list[ChoiceCaller] = []

    # Create MMKernelInputs for Int MM
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=torch.int32)

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten__int_mm)

    if is_nonzero and use_triton_template(
        layout, enable_int32=True, check_max_autotune=False
    ):
        templates_to_use.append(mm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    if use_cutlass and _use_cutlass_for_op(name):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes(), fuseable=True, non_fuseable=True
        )

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.addmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if use_native_matmul(mat1, mat2):
        if beta == 0:
            arg1 = 0
        else:
            arg1 = lowerings[aten.mul](beta, inp)

        if alpha == 0:
            arg2 = 0
        else:
            arg2 = lowerings[aten.mul](alpha, lowerings[aten.mm](mat1, mat2))

        return lowerings[aten.add](arg1, arg2)

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    inp = realize_inputs(inp)
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "addmm"

    # Create MMKernelInputs for AddMM at the top
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    kernel_inputs_aten = MMKernelInputs(
        [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )

    choices: list[ChoiceCaller] = []

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.addmm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.addmm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )
    if (not is_nonzero) or (
        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)
    ):
        choices.extend(
            V.choices.get_template_configs(
                kernel_inputs_aten,
                [aten_addmm],
                name,
            )
        )
        node, _ = autotune_select_algorithm(
            name, choices, kernel_inputs.nodes(), layout
        )
        return node

    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []

    if use_aten_gemm_kernels():
        aten_templates: list[ExternKernelChoice | KernelTemplate] = [aten_addmm]
        if (
            inp.get_stride()[0] == 0
            and len(inp.get_size()) == 2
            and inductor_config.triton.autotune_cublasLt
            and not V.graph.cpp_wrapper  # bias_addmm only has a Python implementation
        ):
            aten_templates.append(aten_bias_addmm)

        # On ROCm, ATen choices use original bias input; non-ROCm keeps unified inputs.
        choices.extend(
            V.choices.get_template_configs(kernel_inputs_aten, aten_templates, name)
        )

    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(mm_template)

        if use_triton_blackwell_tma_template(
            mat1, mat2, output_layout=layout, add_guards=True
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
        elif use_triton_tma_template(mat1, mat2, output_layout=layout, add_guards=True):
            if torch.version.hip is None:
                templates_to_use.append(persistent_tma_mm_template)
            else:
                templates_to_use.append(persistent_mm_template)

        # Manually call get_template_configs as use 1-D bias if possible
        choices.extend(
            V.choices.get_template_configs(
                kernel_inputs_aten, [addmm_contiguous_subgraph_template], name
            )
        )
    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    streamk_only_active = False
    if static_shape and is_nonzero and ENABLE_STREAMK:
        streamk_supported = can_use_streamk(m, n, k, mat1.get_dtype(), layout.device)
        if streamk_supported:
            standard_choices = list(choices)
            if STREAMK_ONLY:
                streamk_log_info(
                    f"TORCHINDUCTOR_STREAMK_ONLY=1: using only StreamK candidates for addmm {m}x{n}x{k}"
                )
                choices.clear()
                streamk_only_active = True

            try:
                streamk_added = add_streamk_mm_choices(
                    choices,
                    kernel_inputs,
                    layout,
                    m,
                    n,
                    k,
                    mat1,
                    mat2,
                    input_nodes=tuple(kernel_inputs.nodes()),
                    extra_template_kwargs={
                        "prefix_args": 1,
                        "epilogue_fn": addmm_epilogue(
                            kernel_inputs.out_dtype(), alpha, beta
                        ),
                        "epilogue_fn_hash": str(
                            ["addmm_epilogue", kernel_inputs.out_dtype(), alpha, beta]
                        ),
                        "BIAS": False,
                    },
                )
            except Exception as exc:
                streamk_added = 0
                streamk_log_info(f"StreamK addmm competitor setup failed: {exc}")
                if STREAMK_DEBUG:
                    import traceback

                    streamk_log_debug(traceback.format_exc())

            if STREAMK_ONLY and streamk_added == 0:
                streamk_log_info(
                    f"No StreamK choices generated for addmm {m}x{n}x{k}; restoring standard choices"
                )
                choices[:] = standard_choices
                streamk_only_active = False
        elif STREAMK_DEBUG:
            streamk_log_info(
                f"StreamK was not added as an addmm candidate for {m}x{n}x{k}"
            )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
        and not streamk_only_active
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            # reorder here because CUTLASS expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k) and not streamk_only_active:
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            # reorder here because CK expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if use_cpp_gemm_template(layout, mat1, mat2) and not streamk_only_active:
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@register_lowering(aten._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    from torch._inductor.select_algorithm import realize_inputs

    # TODO(coconturuben): support V.choices.get_mm_configs for sparse_semi_structured_mm
    mat1, mat1_meta, mat2 = realize_inputs(mat1, mat1_meta, mat2)
    m1, k1 = mat1.get_size()
    m2, _ = mat1_meta.get_size()
    k2, n = mat2.get_size()
    m = V.graph.sizevars.check_equals_and_simplify(m1, m2)
    k = V.graph.sizevars.check_equals_and_simplify(2 * k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        layout = FixedLayout(
            mat2.get_device(),
            out_dtype if out_dtype else mat2.get_dtype(),
            [m, n],
            [n, 1],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    choices = (
        [
            aten__sparse_semi_structured_mm.bind(
                (mat1, mat1_meta, mat2), layout, out_dtype=out_dtype
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        m * n != 0
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("sparse_semi_structured_mm")
    ):
        CUTLASS2xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2, mat1_meta], fuseable=True, non_fuseable=True
        )

    node, _ = autotune_select_algorithm(
        "sparse_semi_structured_mm", choices, (mat1, mat1_meta, mat2), layout
    )
    return node


scaling_pairs = [
    (ScalingType.TensorWise, ScalingType.TensorWise),
    (ScalingType.RowWise, ScalingType.RowWise),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise128x128),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise1x128),
    (ScalingType.BlockWise128x128, ScalingType.BlockWise1x128),
]


epilogue_scaling_types = [ScalingType.TensorWise, ScalingType.RowWise]
main_loop_scaling_types = [ScalingType.BlockWise1x128, ScalingType.BlockWise128x128]


def _is_tensorwise_scaling(sz: Any) -> bool:
    return (len(sz) == 0) or all(
        V.graph.sizevars.statically_known_equals(d, 1) for d in sz
    )


def _is_rowwise_scaling(sz: Any, transpose: bool) -> bool:
    idx = 0 if transpose else -1
    return V.graph.sizevars.statically_known_equals(sz[idx], 1)


def _is_blockwise1xTILESIZE_scaling(
    sz: Any, tensor_sz: Any, tile_size: int, transpose: bool
) -> bool:
    lhs = 1 if transpose else 0
    rhs = 0 if transpose else 1
    return V.graph.sizevars.statically_known_equals(
        sz[lhs], tensor_sz[lhs]
    ) and V.graph.sizevars.statically_known_equals(
        sz[rhs], ceildiv(tensor_sz[rhs], tile_size)
    )


def _is_blockwise128x128_scaling(sz: Any, tensor_sz: Any) -> bool:
    return V.graph.sizevars.statically_known_equals(
        sz[0], ceildiv(tensor_sz[0], 128)
    ) and V.graph.sizevars.statically_known_equals(sz[1], ceildiv(tensor_sz[1], 128))


def is_desired_scaling(
    t: Any,
    scale_size: torch.Tensor,
    scaling_type: ScalingType,
    transpose: bool = False,
) -> bool:
    match scaling_type:
        case ScalingType.TensorWise:
            return _is_tensorwise_scaling(scale_size)
        case ScalingType.RowWise:
            return _is_rowwise_scaling(scale_size, transpose)
        case ScalingType.BlockWise1x128:
            return _is_blockwise1xTILESIZE_scaling(
                scale_size, t.get_size(), 128, transpose
            )
        case ScalingType.BlockWise128x128:
            return _is_blockwise128x128_scaling(scale_size, t.get_size())
        case _:
            raise AssertionError(f"Unsupported scaling type {scaling_type}")


def get_tile_size(scale_option) -> int:
    match scale_option:
        case ScalingType.BlockWise128x128:
            return 128
        case ScalingType.BlockWise1x128:
            return 128
        case _:
            raise AssertionError(
                f"Unsupported scaling type {scale_option} in get_tile_size"
            )


def get_scaling_options(
    mat_a: Any,
    mat_b: Any,
    scale_a_size: torch.Tensor,
    scale_b_size: torch.Tensor,
) -> tuple[ScalingType, ScalingType]:
    for scale_option_a, scale_option_b in scaling_pairs:
        if is_desired_scaling(
            mat_a, scale_a_size, scale_option_a
        ) and is_desired_scaling(mat_b, scale_b_size, scale_option_b, transpose=True):
            return scale_option_a, scale_option_b

    raise AssertionError(
        f"Inductor Triton does not support scale_a.shape = {scale_a_size}, scale_b.shape = {scale_b_size}"
    )  # verify that shapes are supported by at least one existing pairing


@register_lowering(aten._scaled_mm_v2.default, type_promotion_kind=None)
def tuned_scaled_mm_v2(
    mat_a,
    mat_b,
    scale_a: list[Any],
    recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[Any],
    recipe_b: list[int],
    swizzle_b: list[int],
    bias=None,
    out_dtype=None,
    contraction_dim=None,
    use_fast_accum=False,
    layout=None,
):
    """
    Performs an optimized matrix multiplication where scaling factors are
    applied to the inputs per the supplied recipes, and optionally swizzled.

    This is the _scaled_mm_v2 API, which takes scale recipes (ScalingType) and
    swizzle patterns alongside the scale tensors, and supports multi-level
    scaling via lists.
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._scaled_mm_v2.default_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_mm_v2.default: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)

    assert len(scale_a) >= 1 and len(scale_b) >= 1, (
        "scale_a and scale_b must each have at least one entry"
    )

    is_single_level_scale = len(scale_a) == 1 and len(scale_b) == 1

    # Swizzling is not yet wired into any template here; reject anything other
    # than NO_SWIZZLE (=0) so we don't silently produce wrong results once a
    # caller starts passing real swizzle patterns.
    assert all(s == 0 for s in swizzle_a) and all(s == 0 for s in swizzle_b), (
        "Inductor _scaled_mm_v2 lowering does not yet support non-trivial "
        f"swizzles (got swizzle_a={list(swizzle_a)}, swizzle_b={list(swizzle_b)})"
    )

    def check_supported_recipe(recipe: list[int]) -> bool:
        disallowed = OrderedSet([ScalingType.BlockWise1x16, ScalingType.BlockWise1x32])
        return all(ScalingType(r) not in disallowed for r in recipe)

    supported_recipe = check_supported_recipe(recipe_a) and check_supported_recipe(
        recipe_b
    )

    # Only handle single-level scales (no MX/NV)
    scale_a_real, scale_b_real = realize_inputs(scale_a[0], scale_b[0])

    input_nodes: list[Any]

    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]

    # Create MMKernelInputs for Scaled MM (matrices are at indices 0, 1)
    kernel_inputs = MMKernelInputs(
        input_nodes, mat1_idx=0, mat2_idx=1, out_dtype=out_dtype
    )

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten__fp8_mm)
        kwarg_overrides[aten__fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )

    _, is_nonzero = _is_static_problem(layout)

    if (
        # We dont have triton lowerings for the MX variants yet
        is_single_level_scale
        and supported_recipe
        and scale_a[0].dtype == torch.float32
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)

        # Note: No NVFP4 support at this point - can ignore swizzling, and take only the
        #       first scale types passed.
        scale_option_a, scale_option_b = (
            ScalingType(recipe_a[0]),
            ScalingType(recipe_b[0]),
        )

        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exist
        if (
            use_triton_tma_template(mat_a, mat_b, output_layout=layout, add_guards=True)
            and not bias
        ):
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value

            if use_triton_scaling_template(
                scale_option_a, scale_option_b, epilogue_scaling_types
            ):
                templates_to_use.append(scaled_mm_device_tma_epilogue_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_epilogue_scaling_template.uid] = (
                    overriders
                )
            elif use_triton_scaling_template(
                scale_option_a, scale_option_b, main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)

                templates_to_use.append(scaled_mm_device_tma_main_loop_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_main_loop_scaling_template.uid] = (
                    overriders
                )
            else:
                raise AssertionError(
                    "Inductor Triton does not support scaling options that are present "
                    + "in both epilogue scaling and main loop scaling"
                )

        if (
            use_triton_blackwell_tma_template(
                mat_a, mat_b, output_layout=layout, add_guards=True
            )
            and not bias
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[blackwell_ws_persistent_device_tma_mm_template.uid] = (
                overriders
            )

        if use_triton_scaling_template(
            scale_option_a, scale_option_b, epilogue_scaling_types
        ):
            templates_to_use.append(mm_template)
            kwarg_overrides[mm_template.uid] = overriders

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )

    # NVGEMM get_kernels() will return empty if the scaling mode/dtype is unsupported
    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat_a, mat_b):
        from ..codegen.nv_universal_gemm import add_nv_universal_scaled_gemm_choices

        add_nv_universal_scaled_gemm_choices(
            choices,
            layout,
            input_nodes,
            kernel_inputs=kernel_inputs,
        )

    # Early return for MX variants
    if (
        scale_a[0].dtype != torch.float32
        or (not supported_recipe)
        or (not is_single_level_scale)
    ):
        node, _ = autotune_select_algorithm(name, choices, input_nodes, layout)
        return node

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            kernel_inputs.nodes(),  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)  # type: ignore[misc]
def tuned_scaled_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    layout=None,
):
    """
    Performs an optimized matrix multiplication where scaling factors are applied
    to the inputs and/or output.

    Args:
        mat1 (Tensor): First input matrix
        mat2 (Tensor): Second input matrix
        scale1 (Tensor): Scale factor applied to mat1 (supports broadcasting)
        scale2 (Tensor): Scale factor applied to mat2 (supports broadcasting)
        bias (Tensor, optional): Optional bias tensor to add to the result
        layout: Layout hint for optimization

    Returns:
        Tensor: The result of the scaled matrix multiplication
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._scaled_mm.default_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_mm.default: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)

    scale_a_real, scale_b_real = realize_inputs(scale_a, scale_b)

    input_nodes: list[Any]

    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]

    # Create MMKernelInputs for Scaled MM (matrices are at indices 0, 1)
    kernel_inputs = MMKernelInputs(
        input_nodes, mat1_idx=0, mat2_idx=1, out_dtype=out_dtype
    )

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten__fp8_mm)
        kwarg_overrides[aten__fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )

    _, is_nonzero = _is_static_problem(layout)

    if (
        # We don't have triton lowerings for the MX variants yet
        scale_a.dtype == torch.float32
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)

        scale_a_size, scale_b_size = scale_a_real.shape, scale_b_real.shape

        scale_option_a, scale_option_b = get_scaling_options(
            mat_a, mat_b, scale_a_size, scale_b_size
        )

        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exist
        if (
            use_triton_tma_template(mat_a, mat_b, output_layout=layout, add_guards=True)
            and not bias
        ):
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value

            if use_triton_scaling_template(
                scale_option_a, scale_option_b, epilogue_scaling_types
            ):
                templates_to_use.append(scaled_mm_device_tma_epilogue_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_epilogue_scaling_template.uid] = (
                    overriders
                )
            elif use_triton_scaling_template(
                scale_option_a, scale_option_b, main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)

                templates_to_use.append(scaled_mm_device_tma_main_loop_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_main_loop_scaling_template.uid] = (
                    overriders
                )
            else:
                raise AssertionError(
                    "Inductor Triton does not support scaling options that are present "
                    + "in both epilogue scaling and main loop scaling"
                )

        if (
            use_triton_blackwell_tma_template(
                mat_a, mat_b, output_layout=layout, add_guards=True
            )
            and not bias
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[blackwell_ws_persistent_device_tma_mm_template.uid] = (
                overriders
            )

        if use_triton_scaling_template(
            scale_option_a, scale_option_b, epilogue_scaling_types
        ):
            templates_to_use.append(mm_template)
            kwarg_overrides[mm_template.uid] = overriders

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )

    # NVGEMM get_kernels() will return empty if the scaling mode/dtype is unsupported
    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat_a, mat_b):
        from ..codegen.nv_universal_gemm import add_nv_universal_scaled_gemm_choices

        add_nv_universal_scaled_gemm_choices(
            choices,
            layout,
            input_nodes,
            kernel_inputs=kernel_inputs,
        )

    # Early return for MX variants
    if scale_a.dtype != torch.float32:
        node, _ = autotune_select_algorithm(name, choices, input_nodes, layout)
        return node

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            kernel_inputs.nodes(),  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@functools.cache
def _is_sm7x_or_older_gpu(index: int | None) -> bool:
    props = torch.cuda.get_device_properties(index or 0)
    return props.major <= 7


def dims_are_int(dims):
    return all(isinstance(dim, int) for dim in dims)


def mm_autoheuristic(
    mat1,
    mat2,
    m,
    n,
    k,
    choices,
    name,
    input_nodes,
    ops,
    precondition,
    top_k: int | None = None,
    always_included=None,
):
    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None
    mat1_stride, mat2_stride = get_size_hints_strides(mat1, mat2)

    def get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride):
        context = AHContext()
        context.add_feature("m", m)
        context.add_feature("k", k)
        context.add_feature("n", n)
        context.add_feature("mat1_dtype", mat1.layout.dtype, is_categorical=True)
        context.add_feature("mat2_dtype", mat2.layout.dtype, is_categorical=True)
        context_add_strides(context, "mat1", mat1_stride)
        context_add_strides(context, "mat2", mat2_stride)
        context.add_feature(
            "mat1_iscontig", mat1.layout.is_contiguous(), is_categorical=True
        )
        context.add_feature(
            "mat2_iscontig", mat2.layout.is_contiguous(), is_categorical=True
        )
        if name == "mm":
            context_add_using_tf32(context, mat1.layout.dtype)
        return context

    def fallback():
        return None

    context = get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride)
    autoheuristic = AutoHeuristicSelectAlgorithm(
        fallback=fallback,
        choices=choices,
        input_nodes=input_nodes,
        context=context,
        name=name,
        augment_context=ops,
        precondition=precondition,
    )

    if top_k is not None:
        # TODO: is there a cleaner way to ensure aten.mm is always included?
        return autoheuristic.get_top_k_choices_caller(
            top_k, always_included=always_included
        )

    return autoheuristic.get_choice_caller()


def get_size_hints(mat1, mat2, m, n, k):
    if not isinstance(m, int) or not isinstance(k, int):
        (m, k) = V.graph.sizevars.optimization_hints(mat1.get_size())

    if not isinstance(n, int) or not isinstance(k, int):
        (k, n) = V.graph.sizevars.optimization_hints(mat2.get_size())
    return m, n, k


def get_size_hints_strides(mat1, mat2):
    mat1_stride = mat1.layout.stride
    mat2_stride = mat2.layout.stride
    strides = [mat1_stride, mat2_stride]
    strides_hints = []
    for stride in strides:
        if not isinstance(stride, int):
            stride = V.graph.sizevars.optimization_hints(stride)
        strides_hints.append(stride)
    return strides_hints[0], strides_hints[1]
