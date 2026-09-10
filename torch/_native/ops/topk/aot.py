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
# Stated rather than defaulted: the declared ranges and work rungs are measured
# against aten on Hopper and Blackwell.
# Both spellings of each capability, since either can appear in TORCH_CUDA_ARCH_LIST
# and they are distinct nvcc targets.
ARCHS = ("sm_90", "sm_90a", "sm_100", "sm_100a")

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
_RADIX_KS = (64, 128, 256, 512, 1024)
_REGISTER_DYNAMIC_NS = (64, 128, 256, 512, 1024)
_REGISTER_NS = {
    16: (*_REGISTER_DYNAMIC_NS, 2048),
    32: (256,),
}
_REGISTER_RUNGS = {
    16: (_REGISTER_DYNAMIC_NS, (2048,)),
    32: ((256,),),
}
_AOT_REGISTER_RUNGS = {16: (_REGISTER_DYNAMIC_NS,)}
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
_ARCH_TAIL_ITERS = -1


def _radix_min_n(dtype, k, major):
    if dtype not in _RADIX_MIN_N or k not in _RADIX_KS or major < 9:
        return None
    capability = 10 if major >= 10 else 9
    return _RADIX_MIN_N[dtype][capability][k]


def _kernel_for(dtype, n, k, major):
    if major < 9:
        return None
    if dtype == "float32" and k in _REGISTER_NS and n in _REGISTER_NS[k]:
        return "register"
    min_n = _radix_min_n(dtype, k, major)
    if min_n is not None and n >= min_n and n % 4 == 0:
        return "radix"
    return None


def _register_rung(n, k, major):
    if major >= 10 and k == 16 and n == 1024:
        return "1024"
    for rung in _REGISTER_RUNGS.get(k, ()):
        if n in rung:
            return "_".join(str(value) for value in rung)
    return None


def _aot_register_rung(n, k, major):
    if major >= 10 and k == 16 and n == 1024:
        return None
    for rung in _AOT_REGISTER_RUNGS.get(k, ()):
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
    # Blackwell handles the runtime tail loop well, while Hopper needs the
    # fixed upper bound. The sentinel lets one manifest point compile to the
    # best loop for each target architecture.
    if fixed_vec_iters is not None:
        scalar_tail_iters = _MAX_TAIL_ITERS
    elif deterministic:
        scalar_tail_iters = _ARCH_TAIL_ITERS
    else:
        scalar_tail_iters = None
    return scalar_tail_iters, fixed_vec_iters


def kernel_precompile_grid():
    # Only the multi-N register ladder is AOTed; exact-N occupancy outliers stay JIT.
    # Radix kernels share dynamic N, with fixed upper work bounds where measured.
    return [
        *(
            {
                "kernel": "register",
                "dtype": "float32",
                "N": None,
                "N_rung": "_".join(str(value) for value in rung),
                "K": k,
                "eligible": True,
            }
            for k, rungs in _AOT_REGISTER_RUNGS.items()
            for rung in rungs
        ),
        {
            "kernel": "radix",
            "dtype": list(_DTYPES),
            "N": None,
            "K": list(_RADIX_KS),
            "deterministic": False,
            "scalar_tail_iters": None,
            "fixed_vec_iters": None,
            "eligible": True,
        },
        {
            "kernel": "radix",
            "dtype": list(_DTYPES),
            "N": None,
            "K": [64, 128, 256, 1024],
            "deterministic": True,
            "scalar_tail_iters": _ARCH_TAIL_ITERS,
            "fixed_vec_iters": None,
            "eligible": True,
        },
        {
            "kernel": "radix",
            "dtype": list(_DTYPES),
            "N": None,
            "K": 512,
            "deterministic": True,
            "scalar_tail_iters": _MAX_TAIL_ITERS,
            "fixed_vec_iters": 2,
            "eligible": True,
        },
        {
            "kernel": "radix",
            "dtype": list(_DTYPES),
            "N": None,
            "K": 512,
            "deterministic": True,
            "scalar_tail_iters": _ARCH_TAIL_ITERS,
            "fixed_vec_iters": None,
            "eligible": True,
        },
        {
            "kernel": "radix",
            "dtype": "float32",
            "N": None,
            "K": [64, 128, 256],
            "deterministic": [False, True],
            "scalar_tail_iters": _MAX_TAIL_ITERS,
            "fixed_vec_iters": 1,
            "eligible": True,
        },
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
    eligible = n > 0 and self.is_cuda and self.is_contiguous()
    dtype = None
    major = 0
    if eligible:
        if self.dtype == torch.float32:
            dtype = "float32"
        elif self.dtype == torch.bfloat16:
            dtype = "bfloat16"
        props = torch.cuda.get_device_properties(self.device)
        major = props.major
        if self.numel() // n < props.multi_processor_count:
            eligible = False
        if self.const_data_ptr() % (4 * self.element_size()):
            eligible = False
    kernel = _kernel_for(dtype, n, k, major) if eligible else None
    eligible = eligible and kernel is not None
    deterministic = torch.are_deterministic_algorithms_enabled()
    axes = {
        "kernel": kernel,
        "dtype": self.dtype,
        "N": None,
        "K": k,
        "eligible": eligible,
    }
    if kernel == "register":
        axes["N_rung"] = _aot_register_rung(n, k, major)
        axes["eligible"] = eligible and axes["N_rung"] is not None
    elif kernel == "radix":
        scalar_tail_iters, fixed_vec_iters = _specialization(n, k, deterministic)
        axes.update(
            {
                "deterministic": deterministic,
                "scalar_tail_iters": scalar_tail_iters,
                "fixed_vec_iters": fixed_vec_iters,
            }
        )
    return axes


def cpp_covers():
    # C++ port of covered_axes plus grid matching, registered as
    # torch.ops._native_aot.covers_topk, so a call does not walk the full grid in
    # Python.
    dtype_accept = " || ".join(f"st == {t}" for t in _DTYPES.values())
    register_accept = " || ".join(
        f"(k == {k} && ({' || '.join(f'N == {n}' for n in rung)}))"
        for k, rungs in _AOT_REGISTER_RUNGS.items()
        for rung in rungs
    )
    radix_accept = {}
    for dtype, dtype_cpp in _DTYPES.items():
        sm90 = " || ".join(
            f"(k == {k} && N >= {n})" for k, n in _RADIX_MIN_N[dtype][9].items()
        )
        sm100 = " || ".join(
            f"(k == {k} && N >= {n})" for k, n in _RADIX_MIN_N[dtype][10].items()
        )
        radix_accept[dtype] = (
            f"(st == {dtype_cpp} && "
            f"((props->major >= 10 && ({sm100})) || "
            f"(props->major == 9 && ({sm90}))))"
        )
    radix_expr = " || ".join(radix_accept.values())
    return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (!self.is_cuda()) return false;
      if (!largest || !sorted) return false;
      if (self.dim() < 1 || c10::maybe_wrap_dim(dim, self.dim()) != self.dim() - 1) return false;
      const int64_t N = self.dim() >= 1 ? self.size(-1) : 0;
      if (N == 0) return false;
      if (!self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) %
          static_cast<uintptr_t>(4 * self.element_size()) != 0) return false;
      const auto* props = at::cuda::getDeviceProperties(self.device().index());
      if (self.numel() / N < props->multiProcessorCount) return false;
      if (st == at::kFloat && ({register_accept}) &&
          !(props->major >= 10 && k == 16 && N == 1024)) return true;
      return N % 4 == 0 && ({radix_expr});
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
      const int64_t cc_major = _naot_props->major;
      // Perf gate: one CTA per row; below a full wave aten wins.
      if (M < _naot_props->multiProcessorCount) return false;
    """


def cpp_dispatch(spec):
    kernel = spec["kernel"]
    k = spec["K"]
    if kernel == "register":
        ns = tuple(int(n) for n in spec["N_rung"].split("_"))
        n_accept = " || ".join(f"N == {n}" for n in ns)
        condition = (
            f"self.scalar_type() == {_DTYPES[spec['dtype']]} && "
            f"k == {k} && ({n_accept})"
        )
        if k == 16 and 1024 in ns:
            condition += " && (cc_major < 10 || N != 1024)"
        return condition

    deterministic = spec["deterministic"]
    num_threads = max(k, 256)
    tile = num_threads * 4
    sm90_min = _RADIX_MIN_N[spec["dtype"]][9][k]
    sm100_min = _RADIX_MIN_N[spec["dtype"]][10][k]
    conditions = [
        f"self.scalar_type() == {_DTYPES[spec['dtype']]}",
        f"k == {k}",
        "det" if deterministic else "!det",
        "N % 4 == 0",
        f"N >= (cc_major >= 10 ? {sm100_min} : {sm90_min})",
    ]
    fixed_vec_iters = spec["fixed_vec_iters"]
    if k <= 256:
        if fixed_vec_iters is None:
            conditions.append(f"N / {tile} >= 2")
        else:
            conditions.append(f"N / {tile} <= {fixed_vec_iters}")
    elif deterministic and k == 512:
        op = "==" if fixed_vec_iters is not None else "!="
        conditions.append(f"N / {tile} {op} 2")
    return " && ".join(conditions)


def cpp_launch(spec, launch_fn):
    return f"""
      auto self_2d = self.view({{M, N}});
      auto values_2d = values.view({{M, k}});
      auto indices_2d = indices.view({{M, k}});
      {launch_fn}(self_2d, values_2d, indices_2d, at::cuda::getCurrentCUDAStream());
    """
