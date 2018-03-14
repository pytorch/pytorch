#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps) {
  auto diff = abs(x1 - x2 + eps);
  return norm(diff, p, 1);
}
}}  // namespace at::native
