"""Native-AOT declarations for the pointwise family @ CUDA (vec path).

A FAMILY module: declarations() returns one declaration object per
AOT-able row of POINTWISE_DEF_TABLE -- structured aten op, single
output, no dtype restriction conflicts -- covering the vec fast path
only: all operands same dtype (fp32/bf16), same shape, contiguous,
numel % V == 0, 16B-aligned, scalar args at their defaults. Broadcast,
strided, mixed-dtype, bool-output (comparisons), int-promotion inputs
and non-default scalars stay JIT (see PAIN_POINTS P1/P8 for why the
line sits here).

Structured-overload note: comparison rows (gt.Tensor etc.) would need
overload-qualified declarations AND have bool outputs (not vec-able),
so they are excluded by the bool check, not the overload one.

Module scope must import with stdlib alone AND without package context
(torchgen file-path-loads this pre-build), so the row data comes from
the torch-free ``_table_data.py`` sibling, loaded by file path -- the
same single source of truth ``table.py`` rebuilds the typed
PointwiseDef rows from (PAIN_POINTS P13).
"""

import collections
import importlib.util
import os


def _load_rows():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_table_data.py")
    spec = importlib.util.spec_from_file_location("pointwise_table_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PW_ROWS


_Row = collections.namedtuple(
    "_Row", ["aten", "nin", "fn", "promotion", "scalars", "nout", "out_tag", "dt_tag"]
)
_ROWS = [_Row(*r) for r in _load_rows()]

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}

# Rows NOT AOT-able. Values are the reason (documentation; the factory
# also derives most of these mechanically from the row).
_EXCLUDED = {
    "relu": "not structured (composite to clamp_min)",
    "frexp.Tensor": "not structured; multi-output, mixed out dtypes",
    "gt.Tensor": "bool output: not vec-able (out dtype != compute)",
    "lt.Tensor": "bool output",
    "ge.Tensor": "bool output",
    "le.Tensor": "bool output",
    "eq.Tensor": "bool output",
    "ne.Tensor": "bool output",
}


import itertools


# Promotion table baked at DECLARATION time (stdlib-only mirror of
# elementwise_dtypes on this grid, verified against it by the export
# builder, which calls the real thing): compute is always fp32; out is
# bf16 iff every input is bf16, else fp32. V = 128 bits / widest input
# element width (mixed tuples contain fp32 -> V=4; all-bf16 -> V=8).
def _out_name(in_names):
    return "bfloat16" if all(n == "bfloat16" for n in in_names) else "float32"


def _tuple_vec(in_names):
    return 8 if all(n == "bfloat16" for n in in_names) else 4


class _PointwiseDecl:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernel.py"

    def __init__(self, row):
        self.ATEN_OP = row.aten
        self._row = row
        # All input-dtype tuples over the grid (mixed included): 2
        # points for unary, 4 for binary, 8 for ternary (P8 corrected).
        self._tuples = list(itertools.product(list(_DTYPES), repeat=row.nin))

    def kernel_precompile_grid(self):
        # in_dtypes is a TUPLE, not a list: expand_specs cross-multiplies
        # list-valued fields, and the dtype tuple must stay one axis
        # value per point.
        return [{"aten": self._row.aten, "in_dtypes": t} for t in self._tuples]

    def covered_axes(self, *args, **kwargs):
        import torch

        row = self._row
        ins = args[: row.nin]
        if len(ins) < row.nin:
            return {"aten": None}
        names = tuple(
            {torch.float32: "float32", torch.bfloat16: "bfloat16"}.get(t.dtype)
            if isinstance(t, torch.Tensor)
            else None
            for t in ins
        )
        t0 = ins[0]
        if any(n is None for n in names):
            return {"aten": None}
        v = _tuple_vec(names)
        # Alignment via storage_offset, NOT data_ptr() (COW-safe; see
        # PAIN_POINTS P15): each operand needs one V-wide row of ITS
        # dtype aligned. Allocator bases are >=256B aligned; the C++
        # prelude re-checks the real pointer.
        ok = (
            all(
                t.shape == t0.shape
                and t.is_contiguous()
                and (t.storage_offset() * t.element_size()) % (v * t.element_size())
                == 0
                for t in ins
            )
            and t0.numel() > 0
            and t0.numel() % v == 0
        )
        # Non-default scalar args (add's alpha=2 etc.) stay JIT: the AOT
        # kernels bake the default (1).
        pos_scalars = args[row.nin :]
        for i, sname in enumerate(row.scalars):
            val = kwargs.get(sname, pos_scalars[i] if i < len(pos_scalars) else 1)
            if val != 1:
                ok = False
        # Tuple to match the grid points (list != tuple in coverage
        # matching's equality check).
        return {"aten": row.aten if ok else None, "in_dtypes": names}

    def cpp_covers(self):
        # C++ port of covered_axes + grid matching (registered as
        # torch.ops._native_aot.covers_<decl_id>; ~1.3us vs ~3.8us for
        # the Python path). Standalone body rather than prelude reuse:
        # the covers signature carries `out` as an optional (functional
        # schema + trailing out), and V is derived at runtime from the
        # dtype tuple instead of baked per grid point. Every tuple over
        # {fp32, bf16} is on the grid, so on-grid dtypes ARE grid
        # membership. Out, when supplied, must pass the stub's own
        # checks (contiguity, promotion dtype, alignment) or coverage
        # would be wider than the stub's acceptance and gated calls
        # would lose their JIT route to stock aten.
        row = self._row
        names = self._tensor_params()
        lines = [
            "const int64_t numel = self.numel();",
            "if (numel == 0) return false;",
        ]
        for n in names:
            lines.append(
                f"if ({n}.scalar_type() != at::kFloat && {n}.scalar_type() != at::kBFloat16) return false;"
            )
            lines.append(f"if (!{n}.is_contiguous()) return false;")
            if n != "self":
                lines.append(f"if ({n}.sizes() != self.sizes()) return false;")
        all_bf16 = " && ".join(f"{n}.scalar_type() == at::kBFloat16" for n in names)
        lines.append(f"const bool all_bf16 = {all_bf16};")
        lines.append("const int64_t v = all_bf16 ? 8 : 4;")
        lines.append("if (numel % v != 0) return false;")
        for n in names:
            lines.append(
                f"if (reinterpret_cast<uintptr_t>({n}.const_data_ptr()) % (v * {n}.element_size()) != 0) return false;"
            )
        for s in row.scalars:
            lines.append(f"if (!{s}.equal(1)) return false;")
        lines += [
            "if (out.has_value()) {",
            "  if (out->scalar_type() != (all_bf16 ? at::kBFloat16 : at::kFloat)) return false;",
            "  if (!out->is_contiguous() || out->sizes() != self.sizes()) return false;",
            "  if (reinterpret_cast<uintptr_t>(out->const_data_ptr()) % (v * out->element_size()) != 0) return false;",
            "}",
            "return true;",
        ]
        return "\n      ".join(["", *lines]) + "\n"

    # ---- C++ side (over the structured impl signature: tensor inputs,
    # then declared Scalars, then out) ----

    def _tensor_params(self):
        # Structured impl arg names for the row's tensor inputs. aten's
        # binary/ternary structured ops name them (self, other) and
        # (self, tensor1, tensor2); unary just (self).
        n = self._row.nin
        if n == 1:
            return ["self"]
        if n == 2:
            return ["self", "other"]
        return ["self", "tensor1", "tensor2"]

    def cpp_dispatch_prelude(self):
        row = self._row
        names = self._tensor_params()
        # Per-operand dtype must be ON the grid; the exact tuple match
        # happens in cpp_dispatch. Alignment/vec checks are per-point
        # (V varies by tuple), so they live in cpp_dispatch too via the
        # baked constants; here only tuple-independent gates.
        on_grid = " || ".join(f"{{n}}.scalar_type() == {t}" for t in _DTYPES.values())
        checks = [
            "const int64_t numel = self.numel();",
            "if (numel == 0) return false;",
        ]
        for n in names:
            checks.append(
                f"if (!(({on_grid.replace('{n}', n)}))) return false;".replace("{n}", n)
            )
            checks.append(f"if (!{n}.is_contiguous()) return false;")
            checks.append(
                f"if ({n}.sizes() != self.sizes()) return false;" if n != "self" else ""
            )
        checks.append("if (!out.is_contiguous()) return false;")
        scalar_default = " && ".join(f"{s}.equal(1)" for s in row.scalars)
        if scalar_default:
            checks.append(f"if (!({scalar_default})) return false;")
        return "\n      ".join(["", *[c for c in checks if c]]) + "\n"

    def cpp_dispatch(self, spec):
        names = self._tensor_params()
        in_names = tuple(spec["in_dtypes"])
        v = _tuple_vec(in_names)
        conds = [f"numel % {v} == 0"]
        for n, dt in zip(names, in_names):
            conds.append(f"{n}.scalar_type() == {_DTYPES[dt]}")
            elem = 4 if dt == "float32" else 2
            conds.append(
                f"reinterpret_cast<uintptr_t>({n}.const_data_ptr()) % {v * elem} == 0"
            )
        out_elem = 4 if _out_name(in_names) == "float32" else 2
        conds.append(
            f"reinterpret_cast<uintptr_t>(out.data_ptr()) % {v * out_elem} == 0"
        )
        return " && ".join(conds)

    def cpp_launch(self, spec, launch_fn):
        in_names = tuple(spec["in_dtypes"])
        v = _tuple_vec(in_names)
        names = self._tensor_params()
        views = "\n      ".join(
            f"auto {n}_v = {n}.view({{numel / {v}, {v}}});" for n in names
        )
        args = ", ".join(f"{n}_v" for n in names)
        return f"""
      {views}
      auto out_v = out.view({{numel / {v}, {v}}});
      {launch_fn}({args}, out_v, at::cuda::getCurrentCUDAStream());
    """


def declarations():
    return [_PointwiseDecl(row) for row in _ROWS if row.aten not in _EXCLUDED]
