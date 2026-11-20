from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, cast, Literal, TYPE_CHECKING

from torchgen.native_aot_decl import load_by_path, NativeCoverage


if TYPE_CHECKING:
    from torchgen.model import FunctionSchema, NativeFunction


_CFG = load_by_path(
    "pointwise_aot_config", str(Path(__file__).parents[2] / "cutedsl/config.py")
)
_META = load_by_path(
    "pointwise_export_metadata", str(Path(__file__).with_name("export_metadata.py"))
)
_DTYPES = {
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
    "float64": "at::kDouble",
}
_UNSTRUCTURED = {"abs", "deg2rad", "rad2deg", "angle", "rsub.Tensor"}


@cache
def _schema(op: str) -> FunctionSchema:
    from tools.native_aot.gen_aot_lib import _native_function

    return cast("NativeFunction", _native_function(op, structured=False)).func


class Arithmetic(NativeCoverage):
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernels.py"
    ARCHS = _CFG.pointwise_aot_archs()

    def __init__(self, op: str, nin: int) -> None:
        self.ATEN_OP: str = op
        self.nin: int = nin
        self.STRUCTURED: bool = op not in _UNSTRUCTURED
        self.EXACT: bool = op == "round"

    def _presets(self, dtype: str) -> tuple[tuple[int, int, int], ...]:
        bits = 64 if dtype == "float64" else 32 if dtype == "float32" else 16
        return _CFG.pointwise_presets(
            self.ATEN_OP, max(bits, 32), bits, self.nin, dtype=dtype
        )

    def kernel_precompile_grid(self) -> list[dict[str, Any]]:
        common: dict[str, Any] = {
            "op": self.ATEN_OP,
            "dtype": list(_DTYPES),
            "eligible": True,
        }
        grid: list[dict[str, Any]] = [
            dict(
                common,
                dtype=dtype,
                layout="dense",
                work_per_thread=work_per_thread,
                vec_bits=bits,
                transpose=0,
            )
            for dtype in _DTYPES
            for work_per_thread, bits in dict.fromkeys(
                (e, b) for _, e, b in self._presets(dtype)
            )
        ]
        if self.ATEN_OP != "rsub.Tensor":
            for shape in range(len(_CFG.POINTWISE_TMA_SHAPES) + 1):
                first = int(
                    shape > 0
                    and _CFG.POINTWISE_TMA_SHAPES[shape - 1][0]
                    * _CFG.POINTWISE_TMA_SHAPES[shape - 1][1]
                    < _CFG.POINTWISE_PADDED_TMA_MIN
                )
                grid.append(
                    dict(
                        common,
                        dtype=["float16", "bfloat16", "float32"],
                        layout="tma",
                        work_per_thread=1,
                        vec_bits=128,
                        transpose=list(range(first, 1 << self.nin)),
                        shape=shape,
                    )
                )
        return grid

    def cpp_includes(self) -> str:
        return "#include <cmath>\n#include <initializer_list>\n#include <c10/core/impl/COW.h>\n#include <ATen/native/CanUse32BitIndexMath.h>"

    def cpp_helpers(self) -> str:
        shapes = "\n".join(
            f"  if (rows == {r} && cols == {c}) m.shape = {i + 1};"
            for i, (r, c) in enumerate(_CFG.POINTWISE_TMA_SHAPES)
        )
        return f"""
struct PointwiseAotMatch {{ int layout=0, work_per_thread=1, vec_bits=128, transpose=0, shape=0; }};
struct PointwiseAotPreset {{ at::ScalarType dtype; int64_t min_n; int work_per_thread, vec_bits; }};
inline PointwiseAotMatch pointwise_aot_match(
    std::initializer_list<const at::Tensor*> ins, const at::Tensor* out,
    const cudaDeviceProp* props, std::initializer_list<PointwiseAotPreset> presets) {{
  const auto& ref = **ins.begin();
  PointwiseAotMatch m;
  if (!ref.is_cuda() || ref.layout() != at::kStrided) return m;
  const auto n = ref.numel();
  if (n == 0 || n >= std::numeric_limits<int32_t>::max()) return m;
  if (out && (out->sizes() != ref.sizes() || out->scalar_type() != ref.scalar_type() ||
      out->device() != ref.device() || out->is_neg() || out->is_conj() || out->_is_zerotensor() ||
      c10::impl::cow::is_cow_data_ptr(out->storage().data_ptr()))) return m;
  bool dense = ref.is_non_overlapping_and_dense();
  if (out) dense = dense && out->is_non_overlapping_and_dense() && out->strides() == ref.strides();
  for (const auto* t : ins) {{
    if (t->layout() != at::kStrided || t->device() != ref.device() ||
        t->sizes() != ref.sizes() || t->scalar_type() != ref.scalar_type() ||
        t->is_neg() || t->is_conj() || t->_is_zerotensor() ||
        reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16) return m;
    dense = dense && t->is_non_overlapping_and_dense() && t->strides() == ref.strides();
    if (out && out->is_alias_of(*t) && at::get_overlap_status(*out, *t) != at::MemOverlapStatus::No &&
        !(out->const_data_ptr() == t->const_data_ptr() && out->strides() == t->strides())) return m;
  }}
  if (out && reinterpret_cast<uintptr_t>(out->const_data_ptr()) % 16) return m;
  if (dense) {{
    const auto bits = ref.element_size() * 8;
    const double fill = props->multiProcessorCount * props->maxThreadsPerMultiProcessor / double({_CFG.POINTWISE_REFERENCE_LANES});
    for (const auto& p : presets) if (p.dtype == ref.scalar_type() && n >= p.min_n * fill) {{
      m.work_per_thread = p.work_per_thread; m.vec_bits = p.vec_bits;
    }}
    if (n % (m.vec_bits / bits) == 0) m.layout = 1;
    return m;
  }}
  if (ref.scalar_type() == at::kDouble || ref.dim() != 2 || (out ? !out->is_contiguous() : ref.stride(1) != 1)) return m;
  if (out) for (const auto* t : ins) if (out->is_alias_of(*t)) return m;
  const auto rows = ref.size(0), cols = ref.size(1), width = ref.element_size();
{shapes}
  if (rows % {_CFG.TMA_TILE} || cols % {_CFG.TMA_TILE}) return m;
  int index = 0, transposed = 0;
  for (const auto* t : ins) {{
    if (!at::native::canUse32BitIndexMath(*t)) return m;
    const bool trans = t->stride(0) == 1 && t->stride(1) >= rows;
    if (!trans && !(t->stride(1) == 1 && t->stride(0) >= cols)) return m;
    const auto pitch = trans ? t->stride(1) : t->stride(0);
    if ((pitch * width) % 16 || reinterpret_cast<uintptr_t>(t->const_data_ptr()) % (trans ? 128 : 16)) return m;
    if (trans) {{ m.transpose |= 1 << index; ++transposed; }}
    ++index;
  }}
  if (transposed == 0 && n < {_CFG.POINTWISE_PADDED_TMA_MIN}) return m;
  const bool rows_first = transposed > int(ins.size()) - transposed + 1;
  const auto groups = (cols + {_CFG.TMA_TILE * _CFG.TMA_STAGES - 1}) / {_CFG.TMA_TILE * _CFG.TMA_STAGES};
  if ((rows_first ? groups : rows / {_CFG.TMA_TILE}) > 65535) return m;
  const auto smem = ins.size() * ({_CFG.TMA_STAGES * _CFG.TMA_TILE**2} * width + {_CFG.TMA_SMEM_ALIGNMENT}) + {_CFG.TMA_STAGES * 8};
  if (smem <= props->sharedMemPerBlockOptin) m.layout = 2;
  return m;
}}
"""

    def _argument_names(self, kind: Literal["Tensor", "Scalar"]) -> list[str]:
        from torchgen.model import BaseTy, BaseType

        expected = BaseType(BaseTy[kind])
        return [
            arg.name
            for arg in _schema(self.ATEN_OP).arguments.flat_all
            if arg.type == expected
        ]

    def _match_call(self, out: str) -> str:
        inputs = ", ".join("&" + n for n in self._argument_names("Tensor"))
        presets = ", ".join(
            f"{{{ctype}, {n}, {e}, {b}}}"
            for dtype, ctype in _DTYPES.items()
            for n, e, b in self._presets(dtype)
        )
        return f"pointwise_aot_match({{{inputs}}}, {out}, _naot_props, {{{presets}}})"

    def _checks(self) -> str:
        check = ""
        for scalar in self._argument_names("Scalar"):
            check += f"if ({scalar}.isComplex() || !std::isfinite({scalar}.to<double>()) || ({self._argument_names('Tensor')[0]}.scalar_type() != at::kDouble && std::abs({scalar}.to<double>()) > std::numeric_limits<float>::max())) return false;\n"
            if self.ATEN_OP in ("add.Tensor", "sub.Tensor", "rsub.Tensor"):
                check += f"if ({scalar}.isBoolean()) return false;\n"
        return check

    def cpp_covers(self) -> str:
        names = self._argument_names("Tensor")
        types = " || ".join(
            f"{names[0]}.scalar_type() == {v}" for v in _DTYPES.values()
        )
        out = "out ? &*out : nullptr" if self.STRUCTURED else "nullptr"
        layouts = "== 1" if self.ATEN_OP == "rsub.Tensor" else "!= 0"
        return (
            self._checks()
            + f"if (!({types})) return false;\nreturn {self._match_call(out)}.layout {layouts};"
        )

    def cpp_dispatch_prelude(self) -> str:
        out = "&out" if self.STRUCTURED else "nullptr"
        return (
            self._checks()
            + f"const auto match = {self._match_call(out)};\nif (!match.layout) return false;"
        )

    def cpp_dispatch(self, spec: dict[str, Any]) -> str:
        dtype = f"{self._argument_names('Tensor')[0]}.scalar_type() == {_DTYPES[spec['dtype']]}"
        layout = (
            f"match.layout == 1 && match.work_per_thread == {spec['work_per_thread']} && match.vec_bits == {spec['vec_bits']}"
            if spec["layout"] == "dense"
            else f"match.layout == 2 && match.transpose == {spec['transpose']} && match.shape == {spec['shape']}"
        )
        return dtype + " && " + layout

    def cpp_launch(self, spec: dict[str, Any], launch_fn: str) -> str:
        names = self._argument_names("Tensor")
        args = names + [names[0]] * (3 - len(names))
        if spec["layout"] == "dense":
            vector = spec["vec_bits"] // (
                64
                if spec["dtype"] == "float64"
                else 32
                if spec["dtype"] == "float32"
                else 16
            )
            input_views = [f"input{k}" for k in range(len(names))]
            views = "".join(
                f"auto input{k} = {name}.as_strided({{{names[0]}.numel() / {vector}, {vector}}}, {{{vector}, 1}});\n"
                for k, name in enumerate(names)
            )
            tensors = (
                input_views
                + [input_views[0]] * (3 - len(names))
                + [
                    f"out.as_strided({{{names[0]}.numel() / {vector}, {vector}}}, {{{vector}, 1}})"
                ]
            )
        else:
            views = ""
            tensors = args + ["out.view({1, out.size(0), out.size(1)})"]
        scalar_type = "double" if spec["dtype"] == "float64" else "float"
        scalars = [f"{s}.to<{scalar_type}>()" for s in self._argument_names("Scalar")]
        scalars += ["1.0f"] * (1 - len(scalars))
        prelude = ""
        result = ""
        if not self.STRUCTURED:
            native_order = (
                list(reversed(names)) if self.ATEN_OP == "rsub.Tensor" else names
            )
            inputs = "".join(f".add_const_input({n})" for n in native_order)
            prelude = f"at::Tensor out;\nauto iter = at::TensorIteratorConfig().add_output(out){inputs}.build();\nout = iter.output();\n"
            result = "\naot_result = out;"
        return (
            prelude
            + views
            + f"{launch_fn}({', '.join([*tensors, *scalars])}, at::cuda::getCurrentCUDAStream());"
            + result
        )


def declarations() -> list[Arithmetic]:
    copy = load_by_path(
        "pointwise_aot_copy", str(Path(__file__).with_name("aot_copy.py"))
    )
    return [
        Arithmetic(spec.name, spec.nin) for spec in _META.ARITHMETIC_SPECS if spec.aot
    ] + [copy.Copy()]
