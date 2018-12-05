#include "ATen/ATen.h"

namespace at { namespace native {

bool _can_cast(ScalarType from, ScalarType to, Casting casting) {
  return canCast(from, to, casting);
}

}} // namespace at::native
