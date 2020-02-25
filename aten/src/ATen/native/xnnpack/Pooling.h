#pragma once

#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace pool {

struct Output final {
  struct Spatial final {
    int64_t kernel;
    int64_t padding;
    int64_t stride;
    int64_t dilation;
    bool ceil_mode;
  };

  int64_t channels;
  Spatial height, width;
};

} // namespace pool
} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
