#pragma once

#ifdef USE_RUY_QMATMUL

#include <ruy/ruy.h>

namespace at {
namespace native {
namespace ruy_utils {

ruy::Context* get_ruy_context();

void quantize_multiplier(double scale,
                         int* multiplier_fixedpoint,
                         int* multiplier_exponent);

} // namespace ruy_utils
} // namespace native
} // namespace

#endif // USE_RUY_QMATMUL
