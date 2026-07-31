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
