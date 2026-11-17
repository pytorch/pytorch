from __future__ import annotations

from pathlib import Path
from typing import Any

from torchgen.native_aot_decl import load_by_path, NativeCoverage


_CFG = load_by_path(
    "copy_aot_config", str(Path(__file__).parents[2] / "cutedsl/config.py")
)


class Copy(NativeCoverage):
    ATEN_OP = "copy_"
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernels.py"
    STRUCTURED = False
    ARCHS = _CFG.pointwise_aot_archs()
    LAYOUTS = ("matrix", "vector", "tma", "payload", "packed")

    def kernel_precompile_grid(self) -> list[dict[str, Any]]:
        common = dict(
            op=self.ATEN_OP,
            word=1,
            lanes=1,
            normalize=False,
            pack=1,
            full=False,
            eligible=True,
        )
        grid = []
        for word in (1, 2, 4, 8):
            for lanes in (1, 2) if word == 8 else (1,):
                for normalize in (False, True) if word == 1 else (False,):
                    modes = [("matrix", [False, True]), ("payload", False)]
                    if word <= 4 and not normalize:
                        modes += [("vector", [False, True]), ("tma", True)]
                    grid.extend(
                        dict(
                            common,
                            word=word,
                            lanes=lanes,
                            normalize=normalize,
                            layout=layout,
                            full=full,
                        )
                        for layout, full in modes
                    )
        return grid + [dict(common, pack=pack, layout="packed") for pack in (2, 4)]

    def cpp_includes(self) -> str:
        return "#include <c10/core/impl/COW.h>"

    def cpp_helpers(self) -> str:
        vector = _CFG.TRANSPOSE_VECTOR_MIN_BYTES
        tma = _CFG.TRANSPOSE_TMA_MIN_ELEMS
        return f"""
struct TransposeAotMatch {{
  int layout=0, word=1, lanes=1, pack=1;
  bool normalize=false, full=false;
  int64_t outer=1, b=0, middle=1, a=0, inner=1;
}};
inline TransposeAotMatch transpose_aot_match(const at::Tensor& out, const at::Tensor& src, const cudaDeviceProp* props) {{
  TransposeAotMatch p;
  if (!out.is_cuda() || src.device() != out.device() || out.layout() != at::kStrided ||
      src.layout() != at::kStrided || !out.is_contiguous() || src.is_contiguous() ||
      !src.is_non_overlapping_and_dense() ||
      out.sizes() != src.sizes() || out.scalar_type() != src.scalar_type() || src.dim() < 2 ||
      out.is_alias_of(src) || out.is_neg() || out.is_conj() || src.is_neg() || src.is_conj() ||
      out._is_zerotensor() || src._is_zerotensor() ||
      c10::impl::cow::is_cow_data_ptr(out.storage().data_ptr()) || src.numel() == 0) return p;
  const auto item = src.element_size();
  if (item != 1 && item != 2 && item != 4 && item != 8 && item != 16) return p;
  p.word = std::min<int64_t>(item, 8); p.lanes = item / p.word;
  if (src.numel() > INT32_MAX / p.lanes) return p;
  p.normalize = src.scalar_type() == at::kBool;
  if (src.is_quantized()) {{
    if (src.qscheme() != at::kPerTensorAffine && src.qscheme() != at::kPerTensorSymmetric) return p;
    if (out.qscheme() != src.qscheme() || out.q_scale() != src.q_scale() || out.q_zero_point() != src.q_zero_point()) return p;
    if (src.scalar_type() == at::kQUInt4x2) p.pack = 2;
    if (src.scalar_type() == at::kQUInt2x4) p.pack = 4;
    if (out.storage_offset() % p.pack || src.storage_offset() % p.pack) return p;
  }}
  if (src.dim() == 2 && src.stride(0) == 1 && src.stride(1) == src.size(0)) {{
    p.b = src.size(0); p.a = src.size(1);
  }} else {{
    auto iter = at::TensorIteratorConfig().set_check_mem_overlap(false).resize_outputs(false)
        .check_all_same_dtype(false).check_all_same_device(false).add_output(out).add_const_input(src).build();
    const auto shape = iter.shape();
    std::vector<int64_t> order(iter.ndim());
    for (int d = 0; d < iter.ndim(); ++d) order[d] = d;
    std::sort(order.begin(), order.end(), [&](int x, int y) {{ return iter.strides(1)[x] < iter.strides(1)[y]; }});
    int first = -1, second = -1;
    for (int d = 0; d < iter.ndim(); ++d) {{
      if (order[d] != d) {{ if (first < 0) first = d; else if (second < 0) second = d; else return p; }}
    }}
    if (first < 0 || second < 0 || order[first] != second || order[second] != first) return p;
    p.a = shape[first]; p.b = shape[second];
    for (int d = 0; d < first; ++d) p.inner *= shape[d];
    for (int d = first + 1; d < second; ++d) p.middle *= shape[d];
    for (int d = second + 1; d < iter.ndim(); ++d) p.outer *= shape[d];
  }}
  if (p.pack > 1) {{ p.layout = 5; return p; }}
  if (p.inner != 1) {{ p.layout = 4; return p; }}
  const auto tile = p.word >= 8 ? 32 : 64;
  if (p.outer * p.middle > 65535 || (p.a + tile - 1) / tile > 65535 || (p.b + tile - 1) / tile > 65535) return p;
  const bool aligned = reinterpret_cast<uintptr_t>(out.const_data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(src.const_data_ptr()) % 16 == 0;
  const bool tiled = p.word <= 4 && !p.normalize && aligned && p.a % (16 / p.word) == 0 && p.b % (16 / p.word) == 0;
  const auto bytes = p.a * p.b * item;
  const bool vec = tiled && bytes >= (p.word == 1 ? {vector[1]} : p.word == 2 ? {vector[2]} : {vector[4]});
  const bool use_tma = tiled && p.outer * p.middle > 1 && props->major == 10 &&
      p.outer * p.middle * p.a * p.b >= (p.word == 1 ? {tma[1]} : p.word == 2 ? {tma[2]} : {tma[4]});
  const int pack = vec ? 4 / p.word : p.lanes;
  const int actual_tile = vec && p.word < 4 ? 32 : tile;
  p.full = aligned && p.a % (actual_tile * pack) == 0 && p.b % (actual_tile * pack) == 0;
  p.layout = use_tma ? 3 : vec ? 2 : 1;
  if (use_tma) p.full = true;
  return p;
}}
inline at::Tensor transpose_aot_words(const at::Tensor& t, int word, int pack = 1) {{
  const auto dtype = word == 1 ? at::kByte : word == 2 ? at::kUInt16 : word == 4 ? at::kUInt32 : at::kUInt64;
  auto view = at::empty({{0}}, t.options().dtype(dtype));
  return view.set_(t.storage(), t.storage_offset() * t.element_size() / (word * pack),
      {{(t.numel() * t.element_size() + word * pack - 1) / (word * pack)}}, {{1}});
}}
"""

    def cpp_covers(self) -> str:
        return "const auto p = transpose_aot_match(self, src, _naot_props);\nreturn p.layout > 0 && p.layout < 4;"

    def cpp_dispatch_prelude(self) -> str:
        return "const auto p = transpose_aot_match(self, src, _naot_props);\nif (!p.layout) return false;"

    def cpp_dispatch(self, spec: dict[str, Any]) -> str:
        mode = self.LAYOUTS.index(spec["layout"]) + 1
        return f"p.layout == {mode} && p.word == {spec['word']} && p.lanes == {spec['lanes']} && p.normalize == {str(spec['normalize']).lower()} && p.full == {str(spec['full']).lower()} && p.pack == {spec['pack']}"

    def cpp_launch(self, spec: dict[str, Any], launch_fn: str) -> str:
        word, pack = spec["word"], spec["pack"]
        if spec["layout"] in ("payload", "packed"):
            inner = "p.inner" if pack > 1 else "p.inner * p.lanes"
            return f"{launch_fn}(transpose_aot_words(src, {word}, {pack}), transpose_aot_words(self, {word}, {pack}), p.b, p.middle, p.a, {inner}, src.numel(), at::cuda::getCurrentCUDAStream());\naot_result = self;"
        vector = spec["layout"] == "vector"
        setup = "auto a = p.a, b = p.b;\n"
        if vector:
            lanes = 4 // word
            setup += f"a /= {lanes}; b /= {lanes};\n"
            source = f"{{a * {lanes} * p.middle * b, 1, b, {lanes} * p.middle * b, p.middle * b}}"
            target = f"{{b * {lanes} * p.middle * a, {lanes} * p.middle * a, a, 1, p.middle * a}}"
            shape = f"{{p.outer, b, p.middle, a, {lanes}}}"
            word = 4
        else:
            source = "{a * p.middle * b * p.lanes, p.lanes, b * p.lanes, p.middle * b * p.lanes, 1}"
            target = "{b * p.middle * a * p.lanes, p.middle * a * p.lanes, a * p.lanes, p.lanes, 1}"
            shape = "{p.outer, b, p.middle, a, p.lanes}"
        return (
            setup
            + f"auto input = transpose_aot_words(src, {word}).as_strided({shape}, {source});\nauto output = transpose_aot_words(self, {word}).as_strided({shape}, {target});\n{launch_fn}(input, output, at::cuda::getCurrentCUDAStream());\naot_result = self;"
        )
