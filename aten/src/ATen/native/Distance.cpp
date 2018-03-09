#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps) {
  auto diff = abs(x1 - x2);
  auto out = pow(diff + eps, p).sum(1, true);
  return pow(out, 1 / p);
}
}}  // namespace at::native
