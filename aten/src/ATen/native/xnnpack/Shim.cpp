#ifndef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace {

constexpr const char * const kError =
    "Not Implemented! Reason: PyTorch not built with XNNPACK support.";

} // namespace
} // namespace internal

bool available() {
    return false;
}

bool use_convolution2d(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t,
    const bool) {
  return false;
}

Tensor convolution2d(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t,
    const bool) {
  TORCH_CHECK(false, internal::kError);
}

bool use_linear(
    const Tensor&,
    const Tensor&,
    const Tensor&) {
  return false;
}

Tensor linear(
    const Tensor&,
    const Tensor&,
    const Tensor&) {
  TORCH_CHECK(false, internal::kError);
}

} // namespace xnnpack

} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
