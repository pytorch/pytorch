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
    const Tensor&,
    const Tensor&,
    const at::OptionalIntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t,
    bool) {
  return false;
}

Tensor convolution2d(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t) {
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

bool use_max_pool2d(
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    IntArrayRef,
    const IntArrayRef,
    const bool,
    const float,
    const float) {
  return false;
}

Tensor max_pool2d(
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    IntArrayRef,
    const IntArrayRef,
    const bool,
    const float,
    const float) {
  TORCH_CHECK(false, internal::kError);
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
