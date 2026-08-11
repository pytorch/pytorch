"""Native-AOT declaration for aten::topk @ CUDA.

Eligibility is stated three times, deliberately: covered_axes()
(Python, JIT-coverage subtraction), cpp_covers() (its C++ fast path;
must decide the same set), and cpp_dispatch_prelude()/cpp_dispatch()
(the AOT library's dispatch chain). Keep them in sync by hand; drift
is benign but wasteful (a call all sides decline lands on stock aten).

Module scope must stay torch-free (torchgen loads this pre-build;
torchgen itself is fine to import); torch is imported lazily inside
covered_axes.
"""

ATEN_OP = "topk"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "cutedsl_kernels.py"

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
_NS = [2048, 4096, 8192, 16384]
_KS = [64, 128, 256]


def kernel_precompile_grid():
    # fp32 + bf16 radix kernels, both determinism modes (the det
    # specialization is bit-exact vs aten: stable prefix-sum gather +
    # lex (ord, -idx) sort). fp16 and off-grid shapes stay
    # JIT-eligible.
    return [
        {"dtype": list(_DTYPES), "N": _NS, "K": _KS, "deterministic": [False, True]},
    ]


def covered_axes(self, k, dim=-1, largest=True, sorted=True):
    import torch

    n = self.shape[-1] if self.dim() >= 1 else 0
    # Mirror the prelude's full-wave perf gate (M >= SM count). The JIT
    # cond gates small M identically, so sub-wave calls land on stock
    # aten either way -- but coverage must still match the stub's
    # acceptance: an op whose JIT cond DIDN'T gate would silently lose
    # its JIT route for covered-but-stub-rejected calls, and keeping
    # the three eligibility statements (coverage, cpp_covers, prelude)
    # aligned is the declaration contract. Only gate CUDA tensors: the
    # declaration is CUDA-keyed and the device query would throw on CPU.
    # (dim/largest/sorted are deliberately NOT axes: they are not grid
    # fields, and a mismatch declines in the stub to aten by design.)
    if n > 0 and self.is_cuda:
        sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        if self.numel() // n < sm:
            n = 0
    return {
        "dtype": self.dtype,
        "N": n,
        "K": k,
        # Coverage-neutral since both modes are on the grid (any value
        # matches some point); retained because it IS a grid axis --
        # cpp_dispatch keys each point on it to pick the det kernel.
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def cpp_covers():
    # C++ port of covered_axes + grid matching (registered by the AOT
    # library as torch.ops._native_aot.covers_topk; avoids walking the
    # 48-point grid in Python on every call). Covered = on-grid
    # (dtype, N, K) at full-wave M (the prelude's perf gate -- coverage
    # must be no wider than the stub's acceptance or gated calls lose
    # their JIT route to stock aten); both determinism modes are on the
    # grid (the det kernel is bit-exact vs aten);
    # largest/sorted/dim/layout are NOT part of coverage (matching
    # covered_axes) -- off-prelude calls decline in the stub by design.
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
