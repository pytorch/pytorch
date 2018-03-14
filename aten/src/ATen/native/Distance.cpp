#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps) {
  return norm(x1 - x2 + eps, p, 1);
}
}}  // namespace at::native
