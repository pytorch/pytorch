#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/operations/GemmHeuristics.h>

namespace at::native::mps {

GemvConfig GemvPolicy::clamp_vec(GemvConfig cfg, int64_t align) {
  while (cfg.vec > 1 && (align & (cfg.vec - 1))) {
    cfg.vec >>= 1;
  }
  return cfg;
}

namespace {

GemvConfig t2d(int nsimd, int kq) {
  GemvConfig cfg{nsimd, 1};
  cfg.kq = kq;
  cfg.kernel = GemvKernel::T2D;
  return cfg;
}

// gemv_t is bandwidth bound, so the pick just has to keep enough simdgroups
// issuing loads to hide memory latency on every device size: the output gives
// ceil(outlen / (32 * vec)) blocks of them, splitting K gives nsimd per block.
// These few numbers are all that change per dtype.
struct GemvTuning {
  int vec; // load width in elements
  int nsimd_min; // smallest built nsimd for this dtype
  int nsimd_max; // largest built nsimd
  int min_k_per_simd; // fewest K elements worth one simdgroup
  int waves; // target simdgroups per core, sets the occupancy knees
  int small_outlen; // at or below this, use t2d
  int t2d_kq; // t2d k-sublane count
  int scalar_cols_k; // K at or above this uses scalar columns, 0 to disable
};

// One profile for all GPU generations: sweep-fitted on M5 Pro and biased
// toward oversubscription, since on unmeasured hardware extra simdgroups only
// add reduction overhead while missing ones leave memory latency exposed.
// nsimd_min and nsimd_max must match the built MB_GEMV_* kernels, since the
// snap assumes contiguous powers of two.
GemvTuning gemv_tuning_t(c10::ScalarType dt) {
  GemvTuning t{
      .vec = 2,
      .nsimd_min = 16,
      .nsimd_max = 32,
      .min_k_per_simd = 32,
      .waves = 896,
      .small_outlen = 1024,
      .t2d_kq = 8,
      .scalar_cols_k = 0};
  if (dt == at::kFloat) {
    // fp32 moves twice the bytes per element, so it saturates the bus with
    // far fewer waves and rewards backing off the K-split on wide outputs.
    t.nsimd_min = 4;
    t.waves = 56;
    t.t2d_kq = 4;
    t.scalar_cols_k = 16384; // long fp32 reductions prefer scalar columns
  }
  return t;
}

} // namespace

GemvPolicy::GemvPolicy(uint32_t cores) : cores_(cores) {}

GemvPolicy GemvPolicy::current() {
  static const GemvPolicy policy(at::mps::MPSDevice::getInstance()->getCoreCount());
  return policy;
}

GemvConfig GemvPolicy::pick_t(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const {
  const GemvTuning t = gemv_tuning_t(dt);

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
  const int64_t target = int64_t(cores_ > 0 ? cores_ : 10) * t.waves;
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
  return clamp_vec({chosen, t.vec}, align);
}

// gemv_nt reduces one whole row per simdgroup, so occupancy is outlen
// simdgroups no matter what nsimd is; nsimd only sets threadgroup granularity
// and vec the K-loop load width.
GemvConfig GemvPolicy::pick_nt(c10::ScalarType dt, int64_t outlen, int64_t /*K*/, int64_t align) const {
  if (dt == at::kFloat) {
    return clamp_vec({outlen >= 2048 ? 4 : 16, 4}, align);
  }
  return clamp_vec({outlen >= 8192 ? 8 : 4, 8}, align);
}

} // namespace at::native::mps
