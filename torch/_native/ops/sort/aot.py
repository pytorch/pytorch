"""Native-AOT declaration for aten::sort @ CUDA (the out-variant with narrow indices).

Covers the in-place leaves (rows of at most 4096) for the hottest key and index
dtypes; everything else (other dtypes, unstable rows of at most 32, longer rows,
whose leaves launch several kernels per call) keeps the JIT route. As in topk,
eligibility is stated three times on purpose: covered_axes() subtracts from JIT
coverage, cpp_covers() is its C++ fast path, and cpp_dispatch_prelude() /
cpp_dispatch() are the stub's chain. They must accept the same calls: a covered
call that the stub declines would land on stock aten.

The stub sits in sort.stable's structured impl, which also serves functional
sort.stable; the prelude declines int64 indices first so that path is unchanged.

Module scope stays torch-free (torchgen loads this before torch is built).
"""

ATEN_OP = "sort"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "cutedsl_kernels.py"
ARCHS = ("sm_90", "sm_100f", "sm_120")

_KEYS = {
    "bool": "at::kBool",
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
}
_RAW = {
    "bool": "at::kByte",
    "float16": "at::kUInt16",
    "bfloat16": "at::kUInt16",
    "float32": "at::kUInt32",
}
_INDICES = {"uint8": "at::kByte", "uint16": "at::kUInt16", "int32": "at::kInt"}
# Stock sortKeyValueInplace leaf sizes for stable rows (Sort.cu); uint8 indices
# hold rows of at most 256, which the 1024 leaf already covers.
_SORT_SIZES = (128, 1024, 2048, 4096)
_UINT8_MAX_SORT_SIZE = 1024
_INPLACE_MAX_NSORT = 4096
_INT32_MAX = 2**31 - 1


def _sort_size(nsort, stable):
    # cutedsl_inplace.sort_size; duplicated because this module must stay torch-free.
    if not stable and nsort <= 32:
        return 32
    if nsort <= 128:
        return 128
    return max(1024, 1 << (nsort - 1).bit_length())


def kernel_precompile_grid():
    common = {
        "kernel": "inplace",
        "key": list(_KEYS),
        "descending": [False, True],
        "eligible": True,
    }
    return [
        {**common, "index": ["uint16", "int32"], "sort_size": list(_SORT_SIZES)},
        {
            **common,
            "index": "uint8",
            "sort_size": [s for s in _SORT_SIZES if s <= _UINT8_MAX_SORT_SIZE],
        },
    ]


def covered_axes(
    self, stable=None, dim=-1, descending=False, values=None, indices=None
):
    import torch

    is_alias_of = torch._C._is_alias_of  # pyrefly: ignore[missing-attribute]
    eligible = (
        indices is not None
        and indices.dtype != torch.int64
        and not is_alias_of(indices, self)
        and (
            values is None
            or not (is_alias_of(values, self) or is_alias_of(values, indices))
        )
        and self.is_cuda
        and self.dim() >= 1
        and -self.dim() <= dim < self.dim()
        and dim % self.dim() == self.dim() - 1
        and self.is_contiguous()
        and indices.is_contiguous()
        and (values is None or values.is_contiguous())
    )
    nsort = self.size(-1) if self.dim() >= 1 else 0
    eligible = (
        eligible
        and 0 < nsort <= _INPLACE_MAX_NSORT
        and 0 < self.numel() // nsort <= _INT32_MAX
    )
    return {
        "kernel": "inplace",
        "key": str(self.dtype).removeprefix("torch."),
        "index": str(indices.dtype).removeprefix("torch.")
        if indices is not None
        else None,
        "descending": bool(descending),
        "sort_size": _sort_size(nsort, bool(stable)) if eligible else None,
        "eligible": eligible,
    }


# Rows collapse every leading dim, so bound M here: codegen's int32 gate only sees
# the named tensors' own dims.
_SORT_SIZE_CPP = f"""
      const int64_t N = self.size(-1);
      if (N == 0 || N > {_INPLACE_MAX_NSORT}) return false;
      const int64_t M = self.numel() / N;
      // The JIT cond leaves empty sorts to stock; a launch with no rows fails.
      if (M == 0 || _naot_dim_too_big(M)) return false;
      const bool _naot_stable = stable.value_or(false);
      const int64_t sort_size = (!_naot_stable && N <= 32) ? 32
          : N <= 128 ? 128
          : std::max<int64_t>(1024, int64_t{{1}} << (64 - __builtin_clzll(static_cast<uint64_t>(N - 1))));
"""


def cpp_covers():
    keys = " || ".join(f"st == {t}" for t in _KEYS.values())
    idx = " || ".join(f"it == {t}" for t in _INDICES.values())
    sizes = " || ".join(f"sort_size == {s}" for s in _SORT_SIZES)
    return f"""
      if (!indices.has_value()) return false;
      const auto it = indices->scalar_type();
      if (!({idx})) return false;
      const auto st = self.scalar_type();
      if (!({keys})) return false;
      if (!self.is_cuda() || self.dim() < 1) return false;
      if (c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      if (!self.is_contiguous() || !indices->is_contiguous()) return false;
      if (values.has_value() && !values->is_contiguous()) return false;
      if (indices->is_alias_of(self)) return false;
      if (values.has_value() && (values->is_alias_of(self) || values->is_alias_of(*indices))) return false;
{_SORT_SIZE_CPP}
      if (it == at::kByte && sort_size > {_UINT8_MAX_SORT_SIZE}) return false;
      return {sizes};
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
{_SORT_SIZE_CPP}    """


def cpp_dispatch(spec):
    return (
        f"self.scalar_type() == {_KEYS[spec['key']]} && "
        f"indices.scalar_type() == {_INDICES[spec['index']]} && "
        f"descending == {'true' if spec['descending'] else 'false'} && "
        f"sort_size == {spec['sort_size']}"
    )


def cpp_launch(spec, launch_fn):
    raw = _RAW[spec["key"]]
    return f"""
      auto x_2d = self.view({{M, N}}).view({raw});
      auto values_2d = values.view({{M, N}}).view({raw});
      auto indices_2d = indices.view({{M, N}});
      {launch_fn}(x_2d, values_2d, indices_2d, at::cuda::getCurrentCUDAStream());
    """
