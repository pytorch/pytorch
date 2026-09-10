"""Native-AOT declaration for aten::topk @ CUDA.

Eligibility is stated three times, deliberately: covered_axes() subtracts from JIT
coverage, cpp_covers() is its C++ fast path and must decide the same set, and
cpp_dispatch_prelude()/cpp_dispatch() are the AOT library's dispatch chain. Keep them
in sync by hand; drift is benign but wasteful, since a call all sides decline lands on
stock aten.

Module scope must stay torch-free, because torchgen loads this before torch is built;
torch is imported lazily inside covered_axes.
"""

ATEN_OP = "topk"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "cutedsl_kernels.py"
# Stated rather than defaulted: the radix kernel is measured against aten on both
# (48/48 of the grid faster on an H100, median 2.16x; Blackwell is the tuning target).
# Both spellings of each capability, since either can appear in TORCH_CUDA_ARCH_LIST
# and they are distinct nvcc targets.
ARCHS = ("sm_90", "sm_90a", "sm_100", "sm_100a")

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
_NS = [2048, 4096, 8192, 16384]
_KS = [64, 128, 256]
_RADIX_KS = (64, 128, 256, 512, 1024)
_REGISTER_DYNAMIC_NS = (64, 128, 256, 512, 1024)
_REGISTER_RUNGS = {
    16: (_REGISTER_DYNAMIC_NS, (2048,)),
    32: ((256,),),
}
_RADIX_MIN_N = {
    "float32": {
        9: {64: 2048, 128: 2048, 256: 2048, 512: 4096, 1024: 32768},
        10: {64: 128, 128: 256, 256: 512, 512: 4096, 1024: 32768},
    },
    "bfloat16": {
        9: {64: 2048, 128: 2048, 256: 2048, 512: 4096, 1024: 32768},
        10: {64: 2048, 128: 2048, 256: 2048, 512: 4096, 1024: 32768},
    },
}
_MAX_TAIL_ITERS = 4


def _radix_min_n(dtype, k, major):
    if dtype not in _RADIX_MIN_N or k not in _RADIX_KS or major < 9:
        return None
    capability = 10 if major >= 10 else 9
    return _RADIX_MIN_N[dtype][capability][k]


def _register_rung(n, k):
    for rung in _REGISTER_RUNGS.get(k, ()):
        if n in rung:
            return "_".join(str(value) for value in rung)
    return None


def _specialization(n, k, deterministic):
    if k not in _RADIX_KS or n % 4:
        return None, None
    num_threads = max(k, 256)
    tile = num_threads * 4
    vec_iters = n // tile
    fixed_vec_iters = None
    if k <= 256 and vec_iters <= 1:
        fixed_vec_iters = 1
    elif deterministic and k == 512 and vec_iters == 2:
        fixed_vec_iters = vec_iters
    scalar_tail_iters = (
        _MAX_TAIL_ITERS if deterministic or fixed_vec_iters is not None else None
    )
    return scalar_tail_iters, fixed_vec_iters


def kernel_precompile_grid():
    # fp32 and bf16 radix kernels in both determinism modes, the deterministic one
    # bit-exact vs aten. fp16 and off-grid shapes stay JIT-eligible.
    return [
        {"dtype": list(_DTYPES), "N": _NS, "K": _KS, "deterministic": [False, True]},
    ]


def covered_axes(self, k, dim=-1, largest=True, sorted=True):
    import torch

    n = self.shape[-1] if self.dim() >= 1 else 0
    # Mirror the stub's gates, so coverage is never wider than its acceptance: a call
    # the stub declines must keep its JIT route rather than fall to stock aten.
    if dim != -1 and dim != self.dim() - 1:
        n = 0
    if not largest or not sorted:
        n = 0
    # Only gate CUDA tensors, since the device query would throw on CPU.
    if n > 0 and self.is_cuda:
        sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        if self.numel() // n < sm:
            n = 0
    return {
        "dtype": self.dtype,
        "N": n,
        "K": k,
        # Coverage-neutral, since both modes are on the grid, but it is a grid axis:
        # cpp_dispatch keys each point on it to pick the deterministic kernel.
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def cpp_covers():
    # C++ port of covered_axes plus grid matching, registered as
    # torch.ops._native_aot.covers_topk, so a call does not walk the 48-point grid in
    # Python. Covered means on-grid (dtype, N, K) at full-wave M in either determinism
    # mode, with the flags the stub requires; layout is not part of coverage.
    dtype_accept = " || ".join(f"st == {t}" for t in _DTYPES.values())
    n_accept = " || ".join(f"N == {n}" for n in _NS)
    k_accept = " || ".join(f"k == {kk}" for kk in _KS)
    return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (!self.is_cuda()) return false;
      if (!largest || !sorted) return false;
      if (self.dim() < 1 || c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      const int64_t N = self.dim() >= 1 ? self.size(-1) : 0;
      if (N == 0) return false;
      if (self.numel() / N < at::cuda::getDeviceProperties(self.device().index())->multiProcessorCount) return false;
      return ({n_accept}) && ({k_accept});
    """


def cpp_dispatch_prelude():
    dtype_reject = " && ".join(f"self.scalar_type() != {t}" for t in _DTYPES.values())
    return f"""
      if ({dtype_reject}) return false;
      if (!largest || !sorted) return false;
      if (self.dim() < 1) return false;
      if (c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      if (!self.is_contiguous() || !values.is_contiguous() || !indices.is_contiguous()) return false;
      // The exported kernels were compiled with assumed_align = 4 elements (see
      // _make_fake_tensor in cutedsl_kernels.py), which every allocator block
      // satisfies. A tensor viewing a mid-block byte offset would not, and a
      // misaligned vector load faults or reads wrong rather than declining.
      auto _naot_aligned = [](const at::Tensor& t) {{
        return reinterpret_cast<uintptr_t>(t.const_data_ptr()) %
            static_cast<uintptr_t>(4 * t.element_size()) == 0;
      }};
      if (!_naot_aligned(self) || !_naot_aligned(values) || !_naot_aligned(indices)) return false;
      const bool det = at::globalContext().deterministicAlgorithms();
      const int64_t N = self.size(-1);
      if (N == 0) return false;
      const int64_t M = self.numel() / N;
      // Perf gate: one CTA per row; below a full wave aten wins.
      if (M < at::cuda::getCurrentDeviceProperties()->multiProcessorCount) return false;
    """


def cpp_dispatch(spec):
    det = "det" if spec["deterministic"] else "!det"
    return f"self.scalar_type() == {_DTYPES[spec['dtype']]} && N == {spec['N']} && k == {spec['K']} && {det}"


def cpp_launch(spec, launch_fn):
    return f"""
      auto self_2d = self.view({{M, N}});
      auto values_2d = values.view({{M, k}});
      auto indices_2d = indices.view({{M, k}});
      {launch_fn}(self_2d, values_2d, indices_2d, at::cuda::getCurrentCUDAStream());
    """
