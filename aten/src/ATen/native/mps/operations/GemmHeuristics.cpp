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

GemvConfig t2d(int nsimd, int kq) {
  GemvConfig cfg{nsimd, 1};
  cfg.kq = kq;
  cfg.kernel = GemvKernel::T2D;
  return cfg;
}

// gemv_t is bandwidth bound, so the pick just has to keep enough simdgroups
// issuing loads to hide memory latency on every device size: the output gives
// ceil(outlen / (32 * vec)) blocks of them, splitting K gives nsimd per block.
// These few numbers are all that change per dtype and GPU generation.
struct GemvTuning {
  int vec; // load width in elements
  int nsimd_min; // smallest built nsimd for this dtype and vec
  int nsimd_max; // largest built nsimd
  int min_k_per_simd; // fewest K elements worth one simdgroup
  int waves; // target simdgroups per core, sets the occupancy knees
  int small_outlen; // at or below this, use t2d
  int t2d_kq; // t2d k-sublane count
  int scalar_cols_k; // K at or above this uses scalar columns, 0 to disable
};

// nsimd_min and nsimd_max must match the built MB_GEMV_* kernels, since the
// snap assumes contiguous powers of two.
GemvTuning gemv_tuning_t(c10::ScalarType dt, AppleGPUFamily family) {
  // Base profile, measured on M2 Max. fp32 and bf16 differ only in two knobs.
  GemvTuning t{
      .vec = 4,
      .nsimd_min = 4,
      .nsimd_max = 16,
      .min_k_per_simd = 256,
      .waves = 14,
      .small_outlen = 1024,
      .t2d_kq = 8,
      .scalar_cols_k = 0};
  if (dt == at::kFloat) {
    t.t2d_kq = 2; // fp32 t2d uses fewer k sublanes
    t.scalar_cols_k = 16384; // long fp32 reductions use scalar columns
  }

  // Per family overrides go here. Unlisted knobs keep the base.
  switch (family) {
    case AppleGPUFamily::APPLE_10_PLUS:
      // Measured on M5 Pro: this generation rewards narrower loads and a much
      // deeper K-split, and fp32 saturates with far fewer waves than bf16.
      t.vec = 2;
      t.nsimd_max = 32;
      t.min_k_per_simd = 32;
      if (dt == at::kFloat) {
        t.waves = 56;
        t.t2d_kq = 4;
      } else {
        t.nsimd_min = 16;
        t.waves = 896;
      }
      break;
    case AppleGPUFamily::APPLE_8_PLUS: // base is the M2 measurement
    default:
      break; // other families keep the base until measured
  }
  return t;
}

GemvConfig pick_t_for_family(c10::ScalarType dt, int64_t outlen, int64_t K, AppleGPUFamily family, uint32_t cores) {
  const GemvTuning t = gemv_tuning_t(dt, family);

  // Small matrices sit in cache, so let t2d stream them.
  if (outlen <= t.small_outlen) {
    return t2d(16, t.t2d_kq);
  }
  // Very long fp32 reductions prefer scalar columns.
  if (t.scalar_cols_k && K >= t.scalar_cols_k) {
    return {32, 1};
  }

  // Aim for about waves simdgroups per core. The output gives blocks of them,
  // the rest comes from splitting K.
  const int64_t block_n = int64_t{32} * t.vec;
  const int64_t target = int64_t(cores > 0 ? cores : 10) * t.waves;
  const int64_t narrow = target * block_n / t.nsimd_max;
  const int64_t wide = target * block_n / t.nsimd_min;

  // Narrow output splits K the most, wide output the least.
  int nsimd = outlen <= narrow ? t.nsimd_max : outlen <= wide ? t.nsimd_max / 2 : t.nsimd_min;

  // Keep enough K on each simdgroup to be worth the split.
  int k_cap = static_cast<int>(K / t.min_k_per_simd);
  k_cap = k_cap < t.nsimd_min ? t.nsimd_min : (k_cap > t.nsimd_max ? t.nsimd_max : k_cap);
  if (nsimd > k_cap) {
    nsimd = k_cap;
  }

  // Round down to a built nsimd (power of two in range).
  int chosen = t.nsimd_min;
  while (chosen * 2 <= nsimd && chosen < t.nsimd_max) {
    chosen *= 2;
  }
  return {chosen, t.vec};
}

// gemv_nt reduces ROWS whole rows per simdgroup, so occupancy is outlen / rows
// simdgroups no matter what nsimd is; nsimd only sets threadgroup granularity
// and vec the K-loop load width.
struct GemvTuningNt {
  int vec; // load width in elements
  int nsimd_narrow; // threadgroup size below wide_outlen
  int nsimd_wide; // threadgroup size at or above wide_outlen
  int wide_outlen;
  int small_k; // below this K, scalar loads win, 0 to disable
  int r2_min_k; // two rows share one x load at or above this K, 0 to disable
  int waves; // occupancy floor (simdgroups per core) required for rows=2
};

GemvTuningNt gemv_tuning_nt(c10::ScalarType dt, AppleGPUFamily family) {
  // Base profile, measured on M2. fp32 caps the load width at float4 and its
  // only built rows=2 kernel is {8, 4}, so it keeps rows=1 until measured.
  GemvTuningNt t{
      .vec = 8,
      .nsimd_narrow = 4,
      .nsimd_wide = 4,
      .wide_outlen = 16384,
      .small_k = 512,
      .r2_min_k = 2048,
      .waves = 14};
  if (dt == at::kFloat) {
    t.vec = 4;
    t.nsimd_wide = 8;
    t.small_k = 0;
    t.r2_min_k = 0;
  }

  // Per family overrides go here. Unlisted knobs keep the base.
  switch (family) {
    case AppleGPUFamily::APPLE_10_PLUS:
      // Measured on M5 Pro: wide loads win even at tiny K and the x-load
      // sharing of rows=2 never pays, so both cutoffs are off.
      if (dt == at::kFloat) {
        t.nsimd_narrow = 16;
        t.nsimd_wide = 4;
        t.wide_outlen = 2048;
      } else {
        t.nsimd_wide = 8;
        t.wide_outlen = 8192;
        t.small_k = 0;
        t.r2_min_k = 0;
      }
      break;
    case AppleGPUFamily::APPLE_8_PLUS: // base is the M2 measurement
    default:
      break; // other families keep the base until measured
  }
  return t;
}

GemvConfig pick_nt_for_family(c10::ScalarType dt, int64_t outlen, int64_t K, AppleGPUFamily family, uint32_t cores) {
  const GemvTuningNt t = gemv_tuning_nt(dt, family);

  if (t.small_k && K < t.small_k) {
    return {t.nsimd_narrow, 1};
  }
  // Two rows per simdgroup halve the simdgroups in flight; take the x-load
  // reuse only when the halved count still meets the occupancy target.
  const int64_t target = int64_t(cores > 0 ? cores : 10) * t.waves;
  if (t.r2_min_k && K >= t.r2_min_k && outlen / 2 >= target) {
    // rows=2 kernels are built at nsimd=4 (lp) and {8, 4} (fp32) only.
    return dt == at::kFloat ? GemvConfig{8, 4, 2} : GemvConfig{4, t.vec, 2};
  }
  return {outlen >= t.wide_outlen ? t.nsimd_wide : t.nsimd_narrow, t.vec};
}

} // namespace

GemvPolicy::GemvPolicy(at::mps::AppleGPUFamily family, uint32_t cores) : family_(family), cores_(cores) {}

GemvPolicy GemvPolicy::current() {
  static const GemvPolicy policy(at::mps::get_apple_gpu_family(), at::mps::MPSDevice::getInstance()->getCoreCount());
  return policy;
}

GemvConfig GemvPolicy::pick_t(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const {
  return clamp_t(pick_t_for_family(dt, outlen, K, family_, cores_), align);
}

GemvConfig GemvPolicy::pick_nt(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const {
  return clamp_nt(pick_nt_for_family(dt, outlen, K, family_, cores_), align);
}

} // namespace at::native::mps
