"""AOT instantiations for the initial floating pointwise TMA operations.

The structured hook runs after ATen meta, so its output layout is authoritative.
Unsupported calls return false and continue through ATen's original CUDA kernel.
This module must remain torch-free at import time for stage-1 code generation.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

from torchgen.native_aot_decl import load_by_path


# Stage 1 loads this declaration before torch can be imported.
_limits = load_by_path(
    "pointwise_limits", os.path.join(os.path.dirname(__file__), "_limits.py")
)

_DTYPES = {
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
}
_OPS: tuple[tuple[str, str, int], ...] = (
    ("add", "_add", 2),
    ("sub", "_sub", 2),
    ("mul", "_mul", 2),
    ("div.Tensor", "_div", 2),
    ("neg", "_neg", 1),
    ("maximum", "_maximum", 2),
    ("minimum", "_minimum", 2),
    ("atan2", "_atan2", 2),
    ("floor", "_floor", 1),
    ("ceil", "_ceil", 1),
    ("trunc", "_trunc", 1),
    ("sign", "_sign", 1),
    ("exp", "_exp", 1),
    ("exp2", "_exp2", 1),
    ("expm1", "_expm1", 1),
    ("log", "_log", 1),
    ("log2", "_log2", 1),
    ("log10", "_log10", 1),
    ("log1p", "_log1p", 1),
    ("sqrt", "_sqrt", 1),
    ("rsqrt", "_rsqrt", 1),
    ("reciprocal", "_reciprocal", 1),
    ("sin", "_sin", 1),
    ("cos", "_cos", 1),
    ("tan", "_tan", 1),
    ("asin", "_asin", 1),
    ("acos", "_acos", 1),
    ("atan", "_atan", 1),
    ("tanh", "_tanh", 1),
    ("erf", "_erf", 1),
    ("sigmoid", "_sigmoid", 1),
    ("addcmul", "_addcmul", 3),
    ("gt.Tensor", "_gt", 2),
    ("lt.Tensor", "_lt", 2),
    ("ge.Tensor", "_ge", 2),
    ("le.Tensor", "_le", 2),
    ("eq.Tensor", "_eq", 2),
    ("ne.Tensor", "_ne", 2),
)


def _axes(inputs: tuple[Any, ...], out: Any, boolean: bool) -> dict[str, Any]:
    import torch
    from torch._native.ops.pointwise._layout import _layout_config, _OperandLayout

    # Tensor/device/dtype checks stay in this adapter. Pure layout selection is
    # shared with JIT, without importing the CuTeDSL compiler on an AOT-only install.
    dtypes = (torch.bfloat16, torch.float32)
    if not all(isinstance(t, torch.Tensor) for t in inputs):
        return {"eligible": False}
    ref = next((t for t in inputs if t.numel() != 1), None)
    if ref is None or not ref.is_cuda:
        return {"eligible": False}
    contiguous = all(t.is_contiguous() for t in inputs)
    vector = contiguous and any(t.numel() == 1 for t in inputs)
    if (ref.ndim != 2 and not vector) or (contiguous and not vector):
        return {"eligible": False}
    cc = torch.cuda.get_device_capability(ref.device)
    if cc[0] != 10:
        return {"eligible": False}
    if any(
        t.layout != torch.strided
        or t.device != ref.device
        or t.dtype not in dtypes
        or (t.numel() != 1 and t.dtype != ref.dtype)
        or t.is_neg()
        or t.is_conj()
        for t in inputs
    ):
        return {"eligible": False}
    result_dtype = ref.dtype
    for t in inputs:
        if t.ndim != 0 and t.dtype != result_dtype:
            result_dtype = torch.promote_types(result_dtype, t.dtype)
    # Boolean outputs still need the promoted input dtype for scalar conversion.
    common_dtype = result_dtype
    if boolean:
        result_dtype = torch.bool
    shape = (1,) * (max(t.ndim for t in inputs) - ref.ndim) + tuple(ref.shape)
    if out is not None and (
        out.layout != torch.strided
        or out.device != ref.device
        or tuple(out.shape) != shape
        or out.dtype != result_dtype
        or out.is_neg()
        or out.is_conj()
        or (vector and not out.is_contiguous())
        or any(torch._C._overlaps(out, t) for t in inputs)
    ):
        return {"eligible": False}
    layouts = tuple(
        _OperandLayout(
            tuple(t.shape) if t.numel() != 1 else (),
            t.stride() if t.numel() != 1 else (),
            t.element_size(),
            t.const_data_ptr() % _limits._TMA_ALIGNMENT,
            t.is_contiguous(),
        )
        for t in inputs
    )
    output_address = (
        0 if out is None else out.const_data_ptr() % _limits._MAX_VECTOR_BYTES
    )
    output_strides = None if out is None else tuple(out.stride()[-ref.ndim :])
    if out is None and not vector:
        # Before meta, use the reference orientation, like the C++ coverage probe.
        output_strides = (1, ref.size(0)) if ref.stride(0) == 1 else (ref.size(1), 1)
    config = _layout_config(
        layouts,
        tuple(ref.shape),
        output_address,
        torch.cuda.get_device_properties(ref.device).shared_memory_per_block_optin,
        output_strides,
        cc,
        result_dtype.itemsize,
    )
    if config is None:
        return {"eligible": False}
    return {
        "eligible": True,
        "dtype": ref.dtype,
        "out_dtype": "bool" if boolean else result_dtype,
        "common_dtype": common_dtype,
        "scalar_dtype": next((t.dtype for t in inputs if t.numel() == 1), None),
        "transposed": config.transposed,
        "box": 0 if vector else config.box,
        "block": config.block,
        "stages": 0 if vector else config.stages,
        "vector_ept": config.ept if vector else 0,
        "vector_bytes": (config.box * ref.element_size() if config.box != 1 else 1)
        if vector
        else 0,
    }


class _Declaration:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernels"
    ARCHS = ("sm_100f",)

    def __init__(self, aten: str, fn: str, nin: int) -> None:
        self.ATEN_OP = aten
        self.fn = fn
        self.nin = nin
        self.boolean = fn in ("_gt", "_lt", "_ge", "_le", "_eq", "_ne")
        self.cuda_scalars = aten in ("add", "sub", "mul", "div.Tensor", "gt.Tensor")
        self.inputs = (
            ("self", "tensor1", "tensor2") if nin == 3 else ("self", "other")[:nin]
        )
        self.scalars = {
            "add": ("alpha",),
            "sub": ("alpha",),
            "addcmul": ("value",),
        }.get(aten, ())

    def kernel_precompile_grid(self) -> list[dict[str, Any]]:
        sizes = {"bfloat16": 2, "float32": 4, "bool": 1}
        points = []
        # Keep CUDA-scalar variants limited to the five existing binary exports.
        layouts = (False, True, None) if self.cuda_scalars else (False, True)
        for mask in itertools.product(layouts, repeat=self.nin):
            if all(value is None for value in mask):
                continue
            for dtype in _DTYPES:
                for scalar_dtype in _DTYPES if None in mask else (None,):
                    common_dtypes = (
                        (dtype, "float32")
                        if scalar_dtype is not None
                        and scalar_dtype != dtype
                        and dtype != "float32"
                        else (dtype,)
                    )
                    for common_dtype in common_dtypes:
                        out_dtype = "bool" if self.boolean else common_dtype
                        geometries = dict.fromkeys(
                            geometry
                            for cc in ((10, 0), (10, 7))
                            for geometry in _limits._tma_configs(mask, cc, sizes[dtype])
                        )
                        configs = [(0, 0, *geometry) for geometry in geometries]
                        if None in mask and True not in mask:
                            # Emit every alignment fallback used by ATen, plus
                            # the larger-work-per-thread variant for SM107.
                            max_vec = min(
                                _limits._MAX_VECTOR_ELEMENTS,
                                _limits._VECTOR_BYTES // sizes[out_dtype],
                            )
                            for vec in (1, 2, 4, 8):
                                if vec <= max_vec:
                                    configs.append(
                                        (
                                            1 if vec == 1 else vec * sizes[dtype],
                                            _limits._UNROLLED_EPT
                                            if vec == 1
                                            else _limits._DEFAULT_VECTOR_EPT,
                                            0,
                                            _limits._THREADS_PER_BLOCK,
                                            0,
                                        )
                                    )
                            for vec in (2, 4, 8):
                                if vec <= max_vec:
                                    configs.append(
                                        (
                                            vec * sizes[dtype],
                                            max(
                                                _limits._DEFAULT_VECTOR_EPT,
                                                _limits._SM100_VECTOR_BYTES_PER_THREAD
                                                // max(sizes[dtype], sizes[out_dtype]),
                                            ),
                                            0,
                                            _limits._SM100_VECTOR_THREADS,
                                            0,
                                        )
                                    )
                            configs.append(
                                (
                                    _limits._MAX_VECTOR_ELEMENTS * sizes[dtype],
                                    _limits._VECTOR_INPUT_BYTES_PER_THREAD
                                    // sizes[dtype],
                                    0,
                                    _limits._THREADS_PER_BLOCK,
                                    0,
                                )
                            )
                        for vector_bytes, vector_ept, box, block, stages in configs:
                            points.append(
                                {
                                    "dtype": dtype,
                                    "scalar_dtype": scalar_dtype,
                                    "out_dtype": out_dtype,
                                    "common_dtype": common_dtype,
                                    "transposed": mask,
                                    "fn": self.fn,
                                    "boolean": self.boolean,
                                    "eligible": True,
                                    "vector_bytes": vector_bytes,
                                    "vector_ept": vector_ept,
                                    "box": box,
                                    "block": block,
                                    "stages": stages,
                                }
                            )
        return points

    def covered_axes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        for i, name in enumerate(self.scalars, self.nin):
            value = kwargs.get(name, args[i] if i < len(args) else 1)
            if isinstance(value, complex):
                return {"eligible": False}
        inputs = args[: self.nin]
        if (
            self.nin > 1
            and not self.cuda_scalars
            and any(t is None or t.numel() == 1 for t in inputs)
        ):
            return {"eligible": False}
        return _axes(inputs, kwargs.get("out"), self.boolean)

    def _inputs(self) -> str:
        return "{" + ", ".join(self.inputs) + "}"

    def _argument_gate(self) -> str:
        gates = [f"if ({name}.isComplex()) return false;" for name in self.scalars]
        if self.nin > 1 and not self.cuda_scalars:
            scalars = " || ".join(f"{name}.numel() == 1" for name in self.inputs)
            gates.append(f"if ({scalars}) return false;")
        return "\n".join(gates)

    def cpp_helpers(self) -> str:
        return _CPP_HELPERS

    def cpp_dispatch_prelude(self) -> str:
        return f"""
if (_naot_props->major != 10) return false;
{self._argument_gate()}
auto cfg = pointwise_tma_config({self._inputs()}, &out, {str(self.boolean).lower()},
                                _naot_props->sharedMemPerBlockOptin, _naot_props->minor);
if (cfg.mask < 0) return false;
"""

    def cpp_dispatch(self, spec: dict[str, Any]) -> str:
        # Two bits distinguish normal, transposed and scalar operands.
        mask = sum(
            (2 if value is None else 1 if value else 0) << (2 * i)
            for i, value in enumerate(spec["transposed"])
        )
        matrix = "other" if spec["transposed"][0] is None else "self"
        result = (
            f"{matrix}.scalar_type() == {_DTYPES[spec['dtype']]} && cfg.mask == {mask}"
        )
        if spec["scalar_dtype"] is not None:
            scalar = "self" if matrix == "other" else "other"
            result += f" && {scalar}.scalar_type() == {_DTYPES[spec['scalar_dtype']]}"
        out_type = "at::kBool" if self.boolean else _DTYPES[spec["out_dtype"]]
        return (
            result
            + f" && out.scalar_type() == {out_type} && cfg.common_dtype == {_DTYPES[spec['common_dtype']]} && cfg.vector_bytes == {spec['vector_bytes']} && cfg.vector_ept == {spec['vector_ept']} && cfg.box == {spec['box']} && cfg.block == {spec['block']} && cfg.stages == {spec['stages']}"
        )

    def cpp_launch(self, spec: dict[str, Any], launch_fn: str) -> str:
        views = []
        for i, transposed in enumerate(spec["transposed"]):
            name = self.inputs[i]
            if transposed is None:
                views.append(f"auto x{i} = {name}.as_strided({{1}}, {{1}});")
                continue
            if spec["vector_bytes"] != 0:
                views.append(f"auto x{i} = {name}.as_strided({{cfg.cols}}, {{1}});")
                continue
            shape = "{cfg.rows, cfg.cols}"
            stride = (
                f"{{1, {name}.stride(cfg.fast)}}"
                if transposed
                else f"{{{name}.stride(1 - cfg.fast), 1}}"
            )
            views.append(f"auto x{i} = {name}.as_strided({shape}, {stride});")
        output = (
            "auto result = out.as_strided({1, cfg.rows, cfg.cols}, {0, cfg.cols, 1});"
        )
        if spec["vector_bytes"] != 0:
            output = "auto result = out.as_strided({cfg.cols}, {1});"
        scalar = f"{self.scalars[0]}.toFloat()" if self.scalars else "1.0f"
        operands = ", ".join(f"x{i}" for i in range(self.nin))
        return "\n".join(
            [
                *views,
                output,
                f"{launch_fn}({operands}, result, {scalar}, at::cuda::getCurrentCUDAStream());",
            ]
        )

    def cpp_covers(self) -> str:
        return f"""
if (_naot_props->major != 10) return false;
{self._argument_gate()}
return pointwise_tma_config({self._inputs()}, out ? &*out : nullptr,
                            {str(self.boolean).lower()}, _naot_props->sharedMemPerBlockOptin, _naot_props->minor).mask >= 0;
"""


def declarations() -> list[_Declaration]:
    return [_Declaration(*op) for op in _OPS]


_CPP_HELPERS = (
    f"""
constexpr int kPointwiseMinAlignment = {_limits._MIN_ALIGNMENT};
constexpr int kPointwiseTmaAlignment = {_limits._TMA_ALIGNMENT};
constexpr int kPointwiseVectorBytes = {_limits._VECTOR_BYTES};
constexpr int kPointwiseMaxVectorBytes = {_limits._MAX_VECTOR_BYTES};
constexpr int kPointwiseMaxVectorElements = {_limits._MAX_VECTOR_ELEMENTS};
constexpr int kPointwiseVectorEpt = {_limits._DEFAULT_VECTOR_EPT};
constexpr int kPointwiseUnrolledEpt = {_limits._UNROLLED_EPT};
constexpr int kPointwiseInputBytesPerThread = {_limits._VECTOR_INPUT_BYTES_PER_THREAD};
constexpr int64_t kPointwiseSm107MinIOBytes = {_limits._SM107_MIN_IO_BYTES};
constexpr int kPointwiseBox = {_limits._TMA_BOX_SIDE};
constexpr int kPointwiseThreads = {_limits._THREADS_PER_BLOCK};
constexpr int kPointwiseBf16MinStages = {_limits._TMA_BF16_MIN_STAGES};
constexpr int64_t kPointwiseLargeInputBytes = {_limits._TMA_LARGE_INPUT_BYTES};
constexpr int kPointwiseSm107LargeBox = {_limits._TMA_SM107_LARGE_BOX};
constexpr int kPointwiseLargeBytesPerThread = {_limits._TMA_LARGE_BYTES_PER_THREAD};
constexpr int kPointwiseSm100VectorThreads = {_limits._SM100_VECTOR_THREADS};
constexpr int kPointwiseSm100VectorBytesPerThread = {_limits._SM100_VECTOR_BYTES_PER_THREAD};
constexpr int kPointwiseSm100Transpose[] = {{{", ".join(map(str, _limits._TMA_SM100_TRANSPOSE))}}};
constexpr int kPointwiseSm107Transpose[] = {{{", ".join(map(str, _limits._TMA_SM107_TRANSPOSE))}}};
constexpr int kPointwiseSm100Pointwise[] = {{{", ".join(map(str, _limits._TMA_SM100_POINTWISE))}}};
constexpr int kPointwiseSm100LargePointwise[] = {{{", ".join(map(str, _limits._TMA_SM100_LARGE_POINTWISE))}}};
constexpr int kPointwiseSm107Pointwise[] = {{{", ".join(map(str, _limits._TMA_SM107_POINTWISE))}}};
constexpr int kPointwiseSmemAlignment = {_limits._SMEM_ALIGNMENT};
constexpr int kPointwiseBarrierBytes = {_limits._BARRIER_BYTES};
constexpr int kPointwiseMaxGridYZ = {_limits._MAX_GRID_YZ};
"""
    + r"""
// Mirrors _axes and the shared _layout_config/_tma2d_config eligibility rules.
// Keep the steps aligned: reference/capability, output, tile limits, input views.
// This native hook runs after ATen meta and must work without Python/CuTeDSL.
// AOT currently handles 2-D TMA views and dense output in either orientation.
struct PointwiseTmaConfig {
  int mask = -1;
  at::ScalarType common_dtype = at::kFloat;
  int vector_bytes = 0;
  int vector_ept = 0;
  int box = 0, block = kPointwiseThreads, stages = 0;
  int fast = 1;
  int64_t rows = 0, cols = 0;
};

PointwiseTmaConfig pointwise_tma_config(
    std::initializer_list<at::Tensor> inputs, const at::Tensor* out,
    bool boolean, size_t smem_limit, int minor) {
  PointwiseTmaConfig cfg;
  auto matrix = std::find_if(inputs.begin(), inputs.end(), [](const auto& t) { return t.numel() != 1; });
  if (matrix == inputs.end()) return cfg;
  const auto& ref = *matrix;
  bool contiguous = std::all_of(inputs.begin(), inputs.end(), [](const auto& t) { return t.is_contiguous(); });
  bool vector = contiguous && std::any_of(inputs.begin(), inputs.end(), [](const auto& t) { return t.numel() == 1; });
  if (!ref.is_cuda() || ref.layout() != at::kStrided || (ref.dim() != 2 && !vector) ||
      ref.numel() <= 0 || ref.numel() > INT32_MAX) return cfg;
  auto dt = ref.scalar_type();
  if (dt != at::kBFloat16 && dt != at::kFloat) return cfg;
  if (contiguous && !vector) return cfg;
  cfg.common_dtype = dt;
  int64_t rank = ref.dim();
  size_t ntma = 0;
  for (const auto& t : inputs) {
    rank = std::max(rank, t.dim());
    if (t.numel() != 1) ++ntma;
    else if (t.dim() != 0 && t.scalar_type() != dt) cfg.common_dtype = at::kFloat;
  }
  auto result_dt = boolean ? at::kBool : cfg.common_dtype;
  cfg.fast = ref.stride(0) == 1 ? 0 : 1;
  if (out) {
    auto prefix = rank - ref.dim();
    if (out->layout() != at::kStrided || out->device() != ref.device() || out->dim() != rank ||
        out->sizes().slice(prefix) != ref.sizes() || out->scalar_type() != result_dt ||
        out->is_neg() || out->is_conj() ||
        (!vector && reinterpret_cast<uintptr_t>(out->const_data_ptr()) % kPointwiseMinAlignment != 0) ||
        (vector && !out->is_contiguous())) return cfg;
    for (int64_t i = 0; i < prefix; ++i) if (out->size(i) != 1) return cfg;
    cfg.fast = out->stride(prefix) == 1 ? 0 : 1;
    if (!vector && (out->stride(prefix + cfg.fast) != 1 ||
        out->stride(prefix + 1 - cfg.fast) != out->size(prefix + cfg.fast))) return cfg;
    for (const auto& t : inputs) if (out->is_alias_of(t)) return cfg;
  }
  cfg.rows = vector ? 1 : ref.size(1 - cfg.fast);
  cfg.cols = vector ? ref.numel() : ref.size(cfg.fast);
  int mask = 0, i = 0;
  bool transpose_only = true;
  for (const auto& t : inputs) {
    if (!t.is_cuda() || t.layout() != at::kStrided || t.device() != ref.device() ||
        t.is_neg() || t.is_conj()) return cfg;
    if (t.numel() == 1) {
      auto st = t.scalar_type();
      if (st != at::kBFloat16 && st != at::kFloat) return cfg;
      mask |= 2 << (2 * i++);
      continue;
    }
    if (t.sizes() != ref.sizes() || t.scalar_type() != dt) return cfg;
    if (vector) { ++i; continue; }
    bool transposed = t.stride(cfg.fast) != 1;
    transpose_only = transpose_only && transposed;
    int inner = transposed ? 1 - cfg.fast : cfg.fast;
    int64_t pitch = t.stride(1 - inner);
    if (t.stride(inner) != 1 || pitch < t.size(inner) || pitch > INT32_MAX ||
        pitch * t.element_size() % kPointwiseMinAlignment != 0 ||
        reinterpret_cast<uintptr_t>(t.const_data_ptr()) % (transposed ? kPointwiseTmaAlignment : kPointwiseMinAlignment) != 0 ||
        (t.size(0) - 1) * t.stride(0) + (t.size(1) - 1) * t.stride(1) > INT32_MAX) return cfg;
    mask |= (transposed ? 1 : 0) << (2 * i++);
  }
  if (!vector) {
    // Same ordered candidates as _tma_configs, including its 32-element fallback.
    const int* tuning = minor == 7
        ? (transpose_only ? kPointwiseSm107Transpose : kPointwiseSm107Pointwise)
        : (transpose_only ? kPointwiseSm100Transpose : kPointwiseSm100Pointwise);
    if (minor != 7 && !transpose_only && cfg.rows * cfg.cols * ref.element_size() >= kPointwiseLargeInputBytes)
      tuning = kPointwiseSm100LargePointwise;
    cfg.box = tuning[0]; cfg.block = tuning[1]; cfg.stages = tuning[2];
    if (ref.element_size() == 2) cfg.stages = std::max(cfg.stages, kPointwiseBf16MinStages);
    if (minor == 7 && transpose_only && cfg.rows * cfg.cols * ref.element_size() >= kPointwiseLargeInputBytes &&
        cfg.rows % kPointwiseSm107LargeBox == 0 && cfg.cols % kPointwiseSm107LargeBox == 0) {
      cfg.box = kPointwiseSm107LargeBox;
      cfg.block = cfg.box * cfg.box * ref.element_size() / kPointwiseLargeBytesPerThread;
      cfg.stages = 1;
    } else if (cfg.rows % cfg.box != 0 || cfg.cols % cfg.box != 0) {
      cfg.box = kPointwiseBox; cfg.block = kPointwiseThreads;
      if (cfg.rows % cfg.box != 0 || cfg.cols % cfg.box != 0) return cfg;
    }
    if (cfg.rows / cfg.box > kPointwiseMaxGridYZ ||
        ntma * (cfg.stages * cfg.box * cfg.box * ref.element_size() + kPointwiseSmemAlignment) +
        cfg.stages * kPointwiseBarrierBytes > smem_limit) return cfg;
  }
  cfg.mask = mask;
  if (vector) {
    const auto input_bytes = ntma * ref.element_size();
    const auto output_bytes = c10::elementSize(result_dt);
    const bool large = minor == 7 && ref.numel() * (input_bytes + output_bytes) >= kPointwiseSm107MinIOBytes;
    int vec = std::min<int>(kPointwiseMaxVectorElements, kPointwiseVectorBytes / output_bytes);
    if (large) vec = std::min<int>(kPointwiseMaxVectorElements,
        kPointwiseMaxVectorBytes / std::max<size_t>(output_bytes, ref.element_size()));
    auto aligned = [&](int width) {
      return (!out || reinterpret_cast<uintptr_t>(out->const_data_ptr()) % (width * output_bytes) == 0) &&
          std::all_of(inputs.begin(), inputs.end(), [=](const auto& t) {
            return t.numel() == 1 || reinterpret_cast<uintptr_t>(t.const_data_ptr()) % (width * t.element_size()) == 0;
          });
    };
    while (vec > 1 && !aligned(vec)) vec /= 2;
    cfg.vector_bytes = vec == 1 ? 1 : vec * ref.element_size();
    cfg.vector_ept = vec == 1 ? kPointwiseUnrolledEpt : kPointwiseVectorEpt;
    if (vec > 1 && minor != 7) {
      cfg.block = kPointwiseSm100VectorThreads;
      cfg.vector_ept = std::max<int>(kPointwiseVectorEpt,
          kPointwiseSm100VectorBytesPerThread / std::max<size_t>(ref.element_size(), output_bytes));
    }
    if (large && vec == kPointwiseMaxVectorElements) {
      size_t rounded = 1;
      while (rounded < input_bytes) rounded *= 2;
      auto ept = kPointwiseInputBytesPerThread / rounded;
      if (ept >= kPointwiseVectorEpt) cfg.vector_ept = ept;
    }
  }
  return cfg;
}
"""
)
