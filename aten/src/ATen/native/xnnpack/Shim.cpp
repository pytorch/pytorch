#ifndef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/core/Tensor.h>

//
// This file is here so as to provide an implementation even in cases where
// PyTorch is compiled without XNNPACK support.  Under those scenarios, either
// all XNNPACK usage must be gated with #ifdefs at call-sites which would make
// for cluttered logic, or alternatively, all use can be routed to a central
// place, namely here, where available() calls return false preventing the
// XNNPACK related codepaths to be taken, and use of the actual operators
// trigger an error.
//

namespace at::native::xnnpack {
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
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const at::OptionalIntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const int64_t /*unused*/,
    bool /*unused*/) {
  return false;
}

Tensor convolution2d(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const int64_t /*unused*/) {
  TORCH_CHECK(false, internal::kError);
}

bool use_linear(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/) {
  return false;
}

Tensor linear(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/) {
  TORCH_CHECK(false, internal::kError);
}

bool use_max_pool2d(
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const bool /*unused*/,
    const float /*unused*/,
    const float /*unused*/) {
  return false;
}

Tensor max_pool2d(
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const bool /*unused*/,
    const float /*unused*/,
    const float /*unused*/) {
  TORCH_CHECK(false, internal::kError);
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
