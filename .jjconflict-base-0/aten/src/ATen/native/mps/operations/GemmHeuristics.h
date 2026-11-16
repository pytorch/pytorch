#pragma once

#include <c10/core/ScalarType.h>

#include <cstdint>

namespace at::native::mps {

enum class GemvKernel {
  Standard,
  T2D, // 2D lane layout, 16-byte loads; vec is implied, kq = k-sublanes
};

struct GemvConfig {
  int nsimd, vec;
  int kq = 4; // k-sublanes per simdgroup (gemv_t2d only)
  GemvKernel kernel = GemvKernel::Standard;
};

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
  int nt_nsimd_lo;
  int nt_nsimd_hi;
  int nt_vec;
};

GemvTuning gemv_tuning(c10::ScalarType dt);

// One shape-based launch heuristic shared by every GPU generation; only the
// device core count scales the occupancy targets. Per-device peak is deferred
// to an opt-in autotuner (follow-up).
class GemvPolicy {
 public:
  explicit GemvPolicy(uint32_t cores);

  static GemvPolicy current();

  GemvConfig pick_t(
      c10::ScalarType dt,
      int64_t outlen,
      int64_t K,
      int64_t align) const;
  GemvConfig pick_nt(
      c10::ScalarType dt,
      int64_t outlen,
      int64_t K,
      int64_t align) const;

  // Halves vec until the matrix leading dim and storage offset are aligned.
  static GemvConfig clamp_vec(GemvConfig cfg, int64_t align);

 private:
  uint32_t cores_; // scales the occupancy targets with device size
};

} // namespace at::native::mps
