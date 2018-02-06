#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor hinge_embedding_loss(const Tensor& self, const Tensor& target, double margin, bool size_average) {
  auto zeros = at::zeros_like(self);
  auto margin_clamp = (margin - self).clamp_min_(0);
  auto output_margin = at::where(target != 1, margin_clamp, zeros);
  auto output_self = at::where(target != -1, self, zeros);
  auto output = (output_margin + output_self).sum();

  if (size_average) {
    output = output / self.numel();
  }
  return output;
}

}}  // namespace at::native
