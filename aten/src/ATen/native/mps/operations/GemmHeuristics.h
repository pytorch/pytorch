//  Copyright (c) 2026 Apple Inc.

#pragma once

#include <ATen/mps/MPSDevice.h>
#include <c10/core/ScalarType.h>

namespace at::native::mps {

enum class GemvKernel {
  Standard,
  TUnroll8,
  T2D, // 2D lane layout, 16-byte loads; vec is implied, kq = k-sublanes
};

struct GemvConfig {
  int nsimd, vec;
  int rows = 1; // rows per simdgroup (gemv_nt only)
  int kq = 4; // k-sublanes per simdgroup (gemv_t2d only)
  GemvKernel kernel = GemvKernel::Standard;
};

class GemvPolicy {
 public:
  explicit GemvPolicy(at::mps::AppleGPUFamily family);

  static GemvPolicy current();

  GemvConfig pick_t(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const;
  GemvConfig pick_nt(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align) const;

  static GemvConfig clamp_t(GemvConfig cfg, int64_t align);
  static GemvConfig clamp_nt(GemvConfig cfg, int64_t align);

 private:
  at::mps::AppleGPUFamily family_;
};

} // namespace at::native::mps
