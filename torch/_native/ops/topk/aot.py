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

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
_NS = [2048, 4096, 8192, 16384]
_KS = [64, 128, 256]


def kernel_precompile_grid():
    # fp32 and bf16 radix kernels in both determinism modes, the deterministic one
    # bit-exact vs aten. fp16 and off-grid shapes stay JIT-eligible.
    return [
        {"dtype": list(_DTYPES), "N": _NS, "K": _KS, "deterministic": [False, True]},
    ]


def covered_axes(self, k, dim=-1, largest=True, sorted=True):
    import torch

    n = self.shape[-1] if self.dim() >= 1 else 0
    # Mirror the prelude's full-wave perf gate (M >= SM count), so coverage is no
    # wider than the stub's acceptance. Only gate CUDA tensors, since the device query
    # would throw on CPU. dim/largest/sorted are not grid fields and so not axes; a
    # mismatch declines in the stub.
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
    # Python. Covered means on-grid (dtype, N, K) at full-wave M, in either
    # determinism mode; largest/sorted/dim/layout are not part of coverage.
    dtype_accept = " || ".join(f"st == {t}" for t in _DTYPES.values())
    n_accept = " || ".join(f"N == {n}" for n in _NS)
    k_accept = " || ".join(f"k == {kk}" for kk in _KS)
    return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (!self.is_cuda()) return false;
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
