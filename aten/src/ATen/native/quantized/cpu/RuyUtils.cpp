#ifdef USE_RUY_QMATMUL

#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/RuyUtils.h>

namespace at::native::ruy_utils {

static thread_local ruy::Context context;

ruy::Context* get_ruy_context() {
  return &context;
}

// Adopted from Ruy:
// https://github.com/google/ruy/blob/2d950b3bfa7ebfbe7a97ecb44b1cc4da5ac1d6f0/ruy/test.h#L1602
void quantize_multiplier(double scale,
                         int* multiplier_fixedpoint,
                         int* multiplier_exponent) {
  TORCH_CHECK(scale > 0, "Quantization scale (", scale, ") must be positive.");
  const double q = std::frexp(scale, multiplier_exponent);
  auto q_fixed = static_cast<std::int64_t>(std::round(q * (1ll << 31)));
  TORCH_CHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*multiplier_exponent;
  }
  TORCH_CHECK(q_fixed <= std::numeric_limits<std::int32_t>::max());
  *multiplier_fixedpoint = static_cast<std::int32_t>(q_fixed);
}

} // namespace at::native::ruy_utils

#endif // USE_RUY_QMATMUL
