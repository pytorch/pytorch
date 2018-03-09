#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#define EPSILON 1e-12


namespace at { namespace native {

Tensor cosine_embedding_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, bool size_average, bool reduce) {
  auto prod_sum = (input1 * input2).sum(1);
  auto mag_square1 = (input1 * input1).sum(1) + EPSILON;
  auto mag_square2 = (input2 * input2).sum(1) + EPSILON;
  auto denom = (mag_square1 * mag_square2).sqrt_();
  auto cos = prod_sum / denom;

  auto zeros = at::zeros_like(target);
  auto pos = 1 - cos;
  auto neg = (cos - margin).clamp_min_(0);
  auto output_pos = at::where(target == 1, pos, zeros);
  auto output_neg = at::where(target == -1, neg, zeros);
  auto output = output_pos + output_neg;

  if (reduce && size_average) {
    return output.sum() / target.numel();
  } else if (reduce) {
    return output.sum();
  }
  return output;
}

Tensor hinge_embedding_loss(const Tensor& self, const Tensor& target, double margin, bool size_average, bool reduce) {
  auto zeros = at::zeros_like(self);
  auto margin_clamp = (margin - self).clamp_min_(0);
  auto output_margin = at::where(target != 1, margin_clamp, zeros);
  auto output_self = at::where(target != -1, self, zeros);
  auto output = output_margin + output_self;

  if (reduce && size_average) {
    return output.sum() / self.numel();
  } else if (reduce) {
    return output.sum();
  }
  return output;
}

}}  // namespace at::native
