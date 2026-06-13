//  Copyright (c) 2026 Apple Inc.

#include <ATen/native/mps/operations/GemmHeuristics.h>

namespace at::native::mps {

GemvConfig GemvPolicy::clamp_t(GemvConfig cfg, int64_t align) {
  while (cfg.vec > 1 && (align & (cfg.vec - 1))) {
    cfg.vec >>= 1;
  }
  return cfg;
}

GemvConfig GemvPolicy::clamp_nt(GemvConfig cfg, int64_t align) {
  while (cfg.vec > 1 && (align & (cfg.vec - 1))) {
    cfg.vec >>= 1;
  }
  // ROWS=2 kernels are instantiated for vec 4 and 8 in both low-precision and
  // float; narrower vec configs fall back to one row per simdgroup.
  if (cfg.rows == 2 && cfg.vec < 4) {
    cfg.rows = 1;
  }
  return cfg;
}

namespace {
using AppleGPUFamily = at::mps::AppleGPUFamily;

// Default config for families without a dedicated table: M1 (Apple7) and M5+ (Apple10).
GemvConfig pick_t_default(c10::ScalarType dt, int64_t outlen, int64_t K) {
  int nsimd = 32;
  int vec = 2;
  if (outlen <= 512) {
    vec = 1;
  } else if (dt == at::kFloat) {
    if (K >= 16384) {
      // Very long reductions: scalar columns keep more k-rows in flight.
      vec = 1;
    }
  } else if (outlen <= 4096 && K >= 2048 && K <= 3584) {
    nsimd = 8;
    vec = 4;
  } else if (outlen > 24576) {
    // Very wide outputs: bigger column blocks, fewer threadgroups.
    nsimd = 16;
    vec = 4;
  }
  return {nsimd, vec};
}

GemvConfig t2d(int nsimd, int kq) {
  GemvConfig cfg{nsimd, 1};
  cfg.kq = kq;
  cfg.kernel = GemvKernel::T2D;
  return cfg;
}

GemvConfig pick_t_m3(c10::ScalarType dt, int64_t outlen, int64_t K) {
  if (dt == at::kFloat) {
    if (outlen <= 1024) {
      return t2d(16, 8);
    }
    if (outlen > 24576) {
      return {4, 1};
    }
    if (K == 4096 && outlen >= 8192) {
      return {4, 1};
    }
    if (K == 3072 && outlen >= 12288) {
      return {4, 1};
    }
    if (K == 3072 && outlen >= 8192) {
      return {8, 4};
    }
    return {32, 2};
  }
  if (outlen <= 1024) {
    // Few column-blocks: a moderate simdgroup count with 16-byte t2d loads
    // keeps the small matrix streaming without over-splitting K.
    return t2d(16, 8);
  }
  if (outlen <= 3072) {
    return {4, 2};
  }
  if (outlen <= 6144) {
    // Enough column-blocks already fill the GPU, so splitting K across many
    // simdgroups only adds cross-simdgroup reduction overhead; a 2-way split
    // across K wins here.
    return {2, 2};
  }
  if (outlen > 65536) {
    // Very wide outputs: maximally wide t2d loads amortize best.
    return t2d(16, 2);
  }
  return {32, 2};
}

GemvConfig pick_t_for_family(c10::ScalarType dt, int64_t outlen, int64_t K, AppleGPUFamily family) {
  if (family == AppleGPUFamily::APPLE_9_PLUS) {
    return pick_t_m3(dt, outlen, K);
  }
  if (family != AppleGPUFamily::APPLE_8_PLUS) {
    return pick_t_default(dt, outlen, K);
  }

  if (dt == at::kFloat) {
    if (outlen <= 512) {
      return t2d(16, 2);
    }
    if (outlen == 1024 && K == 1024) {
      return t2d(16, 4);
    }
    if (outlen <= 1024) {
      return {8, 2};
    }
    if (outlen == 3072) {
      return {2, 1};
    }
    if (outlen == 3584) {
      return K >= 8192 ? GemvConfig{1, 1} : GemvConfig{4, 4};
    }
    if (outlen == 4096 && K == 14336) {
      return {2, 8};
    }
    if (outlen == 4096) {
      return {2, 1};
    }
    if (outlen == 9216 && K <= 3072) {
      return {2, 8};
    }
    if (outlen == 11008 && K == 4096) {
      return {2, 8};
    }
    if (outlen == 18944 && K == 3584) {
      return {4, 4};
    }
    return pick_t_default(dt, outlen, K);
  }

  int nsimd = 32;
  int vec = 2;
  GemvKernel kernel = GemvKernel::Standard;
  if (outlen == 512 && K == 3584) {
    return t2d(16, 8);
  } else if (dt == at::kBFloat16 && outlen == 1024 && K == 1024) {
    // The standard gemv_t (24,2) leaves the SLC-resident 2MB matrix
    // bandwidth on the table here; the 16-byte-load t2d path is ~30% faster.
    return t2d(16, 4);
  } else if (outlen == 16384 && K == 3072) {
    nsimd = 1;
    vec = 2;
  } else if (outlen <= 512) {
    nsimd = 16;
    vec = 1;
  } else if (outlen == 4096 && K >= 14336) {
    nsimd = 4;
    vec = 8;
  } else if (outlen == 3584 && K >= 8192) {
    nsimd = 4;
    vec = 2;
  } else if (outlen == 4096 && K >= 8192) {
    nsimd = 4;
    vec = 2;
  } else if (outlen == 9216 && K <= 3072) {
    nsimd = 1;
    vec = 1;
  } else if (outlen == 11008 && K == 4096) {
    nsimd = 1;
    vec = 2;
  } else if (outlen == 14336 && K == 4096) {
    nsimd = 1;
    vec = 2;
  } else if (outlen == 32000 && K == 4096) {
    nsimd = 8;
    vec = 8;
  } else if (outlen > 65536 && K >= 4096) {
    nsimd = 8;
    vec = 4;
  } else if (outlen > 24576) {
    nsimd = 16;
    vec = 4;
  }
  GemvConfig cfg{nsimd, vec};
  cfg.kernel = kernel;
  return cfg;
}

GemvConfig pick_nt_default(c10::ScalarType dt, int64_t outlen, int64_t K) {
  int nsimd, vec;
  int rows = 1;
  if (dt == at::kFloat) {
    if (outlen <= 1024) {
      // Few rows: wide loads + big threadgroups fill the cores best.
      nsimd = 16;
      vec = 4;
    } else {
      nsimd = K <= 3072 ? 8 : 4;
      vec = 2;
    }
  } else if (K < 512) {
    nsimd = 4;
    vec = 1;
  } else if (K <= 2048 || (outlen <= 4096 && K <= 3584)) {
    nsimd = 4;
    vec = 8;
  } else if (outlen <= 4096 && K < 8192) {
    // One x load feeds two rows.
    nsimd = 4;
    vec = 8;
    rows = 2;
  } else if (outlen > 24576) {
    nsimd = 8;
    vec = 2;
  } else {
    nsimd = 8;
    vec = 4;
  }
  return {nsimd, vec, rows};
}

GemvConfig pick_nt_m3(c10::ScalarType dt, int64_t outlen, int64_t K) {
  if (K < 512) {
    return {4, 1};
  }
  if (dt == at::kFloat) {
    if (outlen <= 1024) {
      return {4, 4};
    }
    return {4, 8};
  }
  if (outlen > 24576) {
    return {8, 8};
  }
  return {4, 8};
}

GemvConfig pick_nt_for_family(c10::ScalarType dt, int64_t outlen, int64_t K, AppleGPUFamily family) {
  if (family == AppleGPUFamily::APPLE_9_PLUS) {
    return pick_nt_m3(dt, outlen, K);
  }
  if (family != AppleGPUFamily::APPLE_8_PLUS) {
    return pick_nt_default(dt, outlen, K);
  }
  if (dt == at::kFloat) {
    if (outlen <= 512 && K >= 2048) {
      return K == 3584 ? GemvConfig{8, 4} : GemvConfig{4, 4};
    }
    if (outlen == 1024 && K == 1024) {
      return {8, 4};
    }
    if (outlen == 4096 && K == 14336) {
      return {2, 8};
    }
    if (outlen > 24576 && K == 4096) {
      return {8, 4, 2};
    }
    if (K == 4096) {
      return {4, 4};
    }
    if (K <= 3072 && outlen > 1024) {
      return {4, 4};
    }
    if (K <= 3584 && outlen > 1024) {
      return {8, 4};
    }
    if (outlen <= 4096 && K >= 8192) {
      return {8, 4};
    }
    if (outlen > 1024) {
      return {16, 4};
    }
    return pick_nt_default(dt, outlen, K);
  }
  if (K < 512) {
    return {4, 1};
  }
  if (dt == at::kBFloat16 && outlen == 512 && K == 3584) {
    return {4, 8, 2};
  }
  if (outlen <= 512) {
    return K >= 2048 ? GemvConfig{4, 4} : GemvConfig{4, 1};
  }
  if (outlen <= 1024) {
    return {4, 8};
  }
  if (outlen <= 4096) {
    if (K >= 11008 && K != 14336) {
      return {4, 4};
    }
    return {4, 8};
  }
  if (K <= 3072 || K == 4096) {
    return {4, 8, 2};
  }
  if (K <= 3584) {
    return {8, 4};
  }
  if (outlen > 24576) {
    if (K == 4096) {
      return {16, 8};
    }
    return {4, 8, 2};
  }
  return {4, 8};
}

} // namespace

GemvPolicy::GemvPolicy(at::mps::AppleGPUFamily family) : family_(family) {}

GemvPolicy GemvPolicy::current() {
  return GemvPolicy(at::mps::get_apple_gpu_family());
}

GemvConfig GemvPolicy::pick_t(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const {
  return clamp_t(pick_t_for_family(dt, outlen, K, family_), align);
}

GemvConfig GemvPolicy::pick_nt(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const {
  return clamp_nt(pick_nt_for_family(dt, outlen, K, family_), align);
}

} // namespace at::native::mps
