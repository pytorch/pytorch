#include <ATen/native/mobile/cpu/Engine.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {
namespace {

constexpr float kMin = 0.0f;
constexpr float kMax = std::numeric_limits<float>::infinity();

} // namespace
} // namespace internal

bool use_relu(const Tensor& input) {
  return mobile::cpu::use_clamp(input, internal::kMin, internal::kMax);
}

Tensor& relu(Tensor& result, const Tensor& input) {
  return mobile::cpu::clamp(result, input, internal::kMin, internal::kMax);
}

Tensor relu(const Tensor& input) {
  return mobile::cpu::clamp(input, internal::kMin, internal::kMax);
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at
