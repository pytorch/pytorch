"""Native-AOT declaration for aten::sort @ CUDA (the out-variant with narrow indices).

Embeds, for float32 and bfloat16 keys (as topk does), the single-launch kernels
the JIT route picks (cutedsl_impl._launcher): the register kernel for rows of
up to REGISTER_MAX_NSORT at full vector width or width 1, and the in-place leaf
for unstable rows of up to BITONIC_MAX_NSORT. Everything else keeps the JIT
route: other key dtypes, bool keys (partition), longer rows (their kernels run
long enough that host overhead does not matter), and register calls at an
intermediate vector width (rows whose length is not a multiple of the full
width, or partly aligned operands).

As in topk, eligibility is stated three times on purpose: covered_axes()
subtracts from JIT coverage, cpp_covers() is its C++ fast path, and
cpp_dispatch_prelude() / cpp_dispatch() are the stub's chain. They must accept
the same calls: a covered call that the stub declines would land on stock aten.
The stub sits in sort.stable's structured impl, which also serves functional
sort.stable; the prelude declines int64 indices first so that path is unchanged.

Module scope stays torch-free (torchgen loads this before torch is built); the
thresholds come from policy.py, loaded by path.
"""

import os

from torchgen.native_aot_decl import load_by_path


_policy = load_by_path(
    "_native_aot_sort_policy", os.path.join(os.path.dirname(__file__), "policy.py")
)

ATEN_OP = "sort"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "cutedsl_kernels.py"
ARCHS = ("sm_90", "sm_100f", "sm_120")

_KEYS = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
_RAW = {"float32": "at::kUInt32", "bfloat16": "at::kUInt16"}
_INDICES = {"uint8": "at::kByte", "uint16": "at::kUInt16", "int32": "at::kInt"}
# One warp per row: the shortest rung, and the full vector width is rung / 32.
_MIN_RUNG = _policy.register_rung(1)
_REGISTER_RUNGS = tuple(
    1 << b
    for b in range(_MIN_RUNG.bit_length() - 1, _policy.REGISTER_MAX_NSORT.bit_length())
)


def kernel_precompile_grid():
    common = {"key": list(_KEYS), "descending": [False, True], "eligible": True}
    register = [
        {
            **common,
            "kernel": "register",
            "index": index,
            "rung": rung,
            "store": store,
            "rows": _policy.REGISTER_ROWS_PER_CTA,
        }
        for rung in _REGISTER_RUNGS
        for index in _INDICES
        if rung - 1 <= _policy.INDEX_MAX[index]
        for store in sorted({rung // _MIN_RUNG, 1})
    ]
    # The in-place radix leaf (rows past REGISTER_MAX_NSORT) runs 20 us or more,
    # so embedding it saves nothing over JIT; only the bitonic leaf is short.
    bitonic = {
        **common,
        "kernel": "inplace",
        "index": list(_INDICES),
        "sort_size": _policy.BITONIC_MAX_NSORT,
    }
    return [*register, bitonic]


def _final_address(out, like):
    """Where ``out``'s data will be once meta has resized it to ``like``'s shape.

    resize_ keeps the storage offset and reuses a storage that is big enough;
    otherwise the caching allocator's new block is at least 512-byte aligned, so
    only the offset matters modulo the 16-byte vectors."""
    if out.shape == like.shape:
        return out.const_data_ptr()
    offset = out.storage_offset() * out.element_size()
    storage = out.untyped_storage()
    if storage.nbytes() >= offset + like.numel() * out.element_size():
        return storage.data_ptr() + offset
    return offset


def covered_axes(
    self, stable=None, dim=-1, descending=False, values=None, indices=None
):
    import torch

    is_alias_of = torch._C._is_alias_of  # pyrefly: ignore[missing-attribute]
    # Functional sort and int64 out= are never covered.
    if values is None or indices is None or indices.dtype == torch.int64:
        return {"kernel": None, "eligible": False}
    ndim = self.dim()
    nsort = self.size(-1) if ndim >= 1 else 0
    nrows = self.numel() // nsort if nsort else 0
    eligible = (
        self.is_cuda
        and ndim >= 1
        and -ndim <= dim < ndim
        and dim % ndim == ndim - 1
        and self.is_contiguous()
        # Outs meta resizes come out dense like self, so contiguous.
        and (values.shape != self.shape or values.is_contiguous())
        and (indices.shape != self.shape or indices.is_contiguous())
        and not is_alias_of(values, self)
        and not is_alias_of(indices, self)
        and not is_alias_of(values, indices)
        and 0 < nsort <= _policy.REGISTER_MAX_NSORT
        and 0 < nrows <= _policy.MAX_ROWS
    )
    axes = {
        "key": str(self.dtype).removeprefix("torch."),
        "index": str(indices.dtype).removeprefix("torch."),
        "descending": bool(descending),
        "eligible": eligible,
    }
    kernel = (
        _policy.route(nsort, nrows, bool(stable), self.dtype == torch.bool)
        if eligible
        else None
    )
    if kernel == "register":
        # An intermediate vector width matches no grid point: it stays JIT.
        operands = [
            (_final_address(t, self), t.element_size()) for t in (self, values, indices)
        ]
        return axes | {
            "kernel": kernel,
            "rung": _policy.register_rung(nsort),
            "store": _policy.register_store(nsort, operands),
            "rows": _policy.REGISTER_ROWS_PER_CTA,
        }
    sort_size = _policy.inplace_sort_size(nsort, bool(stable)) if kernel else None
    return axes | {"kernel": kernel, "sort_size": sort_size}


# The JIT route's kernel choice and policy.register_store's vector width. The
# row count is bounded by hand: codegen's int32 gate only sees the named
# tensors' own dims, and the JIT cond leaves empty sorts to stock.
_ROUTE_CPP = f"""
      const int64_t N = self.size(-1);
      if (N == 0 || N > {_policy.REGISTER_MAX_NSORT}) return false;
      const int64_t M = self.numel() / N;
      if (M == 0 || M > {_policy.MAX_ROWS}) return false;
      const bool use_register = stable.value_or(false) || N > {_policy.BITONIC_MAX_NSORT};
      const int64_t rung = N <= {_MIN_RUNG} ? {_MIN_RUNG} : int64_t{{1}} << (64 - __builtin_clzll(static_cast<uint64_t>(N - 1)));
      int64_t store = rung / {_MIN_RUNG};
      // aot._final_address: an out meta still has to resize gets self's sizes,
      // keeping its storage offset; a too-small storage is reallocated
      // 512-byte aligned.
      auto _naot_final_address = [&](const at::Tensor& t) -> uintptr_t {{
        if (t.sizes().equals(self.sizes())) return reinterpret_cast<uintptr_t>(t.const_data_ptr());
        const auto offset = static_cast<uintptr_t>(t.storage_offset() * t.element_size());
        if (t.storage().nbytes() >= offset + static_cast<uintptr_t>(self.numel() * t.element_size())) {{
          return reinterpret_cast<uintptr_t>(t.storage().data()) + offset;
        }}
        return offset;
      }};
      auto _naot_misaligned = [&](const at::Tensor& t) {{
        const auto bytes = static_cast<uintptr_t>(std::min<int64_t>(16, store * t.element_size()));
        return _naot_final_address(t) % bytes != 0;
      }};
"""


def _store_cpp(values, indices):
    return f"""
      while (store > 1 && (N % store != 0 || _naot_misaligned(self) || _naot_misaligned({values}) || _naot_misaligned({indices}))) {{
        store /= 2;
      }}
    """


def cpp_covers():
    keys = " || ".join(f"st == {t}" for t in _KEYS.values())
    idx = " || ".join(f"it == {t}" for t in _INDICES.values())
    return f"""
      if (!indices.has_value() || !values.has_value()) return false;
      const auto it = indices->scalar_type();
      if (!({idx})) return false;
      const auto st = self.scalar_type();
      if (!({keys})) return false;
      if (self.dim() < 1) return false;
      if (c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      if (!self.is_contiguous()) return false;
      // Outs meta resizes come out dense like self, so contiguous.
      if (values->sizes().equals(self.sizes()) && !values->is_contiguous()) return false;
      if (indices->sizes().equals(self.sizes()) && !indices->is_contiguous()) return false;
      if (indices->is_alias_of(self) || values->is_alias_of(self) || values->is_alias_of(*indices)) return false;
{_ROUTE_CPP}
      if (use_register) {{
{_store_cpp("*values", "*indices")}
        if (it == at::kByte && rung > {_policy.INDEX_MAX["uint8"] + 1}) return false;
        return store == 1 || store == rung / {_MIN_RUNG};
      }}
      return N <= {_policy.BITONIC_MAX_NSORT};
    """


def cpp_dispatch_prelude():
    return f"""
      // Functional sort.stable and int64 out= keep the stock leaves.
      if (indices.scalar_type() == at::kLong) return false;
      if (self.dim() < 1) return false;
      if (c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      if (!self.is_contiguous() || !values.is_contiguous() || !indices.is_contiguous()) return false;
      // Aliased outs go to stock, which copies first or raises.
      if (values.is_alias_of(self) || indices.is_alias_of(self) || values.is_alias_of(indices)) return false;
{_ROUTE_CPP}
      if (use_register) {{
{_store_cpp("values", "indices")}
      }}
      const int64_t sort_size = N <= {_policy.BITONIC_MAX_NSORT} ? {_policy.BITONIC_MAX_NSORT} : 0;
    """


def cpp_dispatch(spec):
    common = (
        f"self.scalar_type() == {_KEYS[spec['key']]} && "
        f"indices.scalar_type() == {_INDICES[spec['index']]} && "
        f"descending == {'true' if spec['descending'] else 'false'}"
    )
    if spec["kernel"] == "register":
        return f"{common} && use_register && rung == {spec['rung']} && store == {spec['store']}"
    return f"{common} && !use_register && sort_size == {spec['sort_size']}"


def cpp_launch(spec, launch_fn):
    raw = _RAW[spec["key"]]
    return f"""
      auto x_2d = self.view({{M, N}}).view({raw});
      auto values_2d = values.view({{M, N}}).view({raw});
      auto indices_2d = indices.view({{M, N}});
      {launch_fn}(x_2d, values_2d, indices_2d, at::cuda::getCurrentCUDAStream());
    """
