#include <ATen/native/mobile/cpu/Engine.h>

#ifndef USE_XNNPACK

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {
namespace {

constexpr const char * const kError =
    "Not Implemented! Reason: Not a mobile build.";

} // namespace
} // namespace internal

bool available() {
    return false;
}

bool initialize() {
  throw std::runtime_error(internal::kError);
}

bool deinitialize() {
  throw std::runtime_error(internal::kError);
}

bool use_add(
    const Tensor&,
    const Tensor&) {
  return false;
}

Tensor& add(
    Tensor&t,
    const Tensor&,
    const Tensor&) {
  throw std::runtime_error(internal::kError);
}

Tensor add(
    const Tensor&,
    const Tensor&) {
  throw std::runtime_error(internal::kError);
}

bool use_clamp(
    const Tensor&,
    const Scalar,
    const Scalar) {
  return false;
}

Tensor& clamp(
    Tensor&,
    const Tensor&,
    const Scalar,
    const Scalar) {
  throw std::runtime_error(internal::kError);
}

Tensor clamp(
    const Tensor&,
    const Scalar,
    const Scalar) {
  throw std::runtime_error(internal::kError);
}

bool use_convolution(
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

Tensor convolution(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const int64_t,
    const bool) {
  throw std::runtime_error(internal::kError);
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
  throw std::runtime_error(internal::kError);
}

bool use_max_pool(
    const Tensor& input,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const bool) {
  return false;
}

Tensor max_pool(
    const Tensor& input,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const IntArrayRef,
    const bool) {
  throw std::runtime_error(internal::kError);
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
