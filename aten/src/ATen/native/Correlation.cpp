#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

Tensor cov(
    const Tensor& self,
    int64_t correction,
    const c10::optional<Tensor>& fweights,
    const c10::optional<Tensor>& aweights) {
  TORCH_CHECK(
      self.ndimension() <= 2, "cov(): input has more than 2 dimensions");

  // View input tensor as 2D (variables, observations)
  auto in = self.ndimension() < 2 ? self.view({1, -1}) : self;
  const auto num_observations = in.size(1);

  Tensor w;

  if (fweights.has_value()) {
    const auto& fw = fweights.value();
    TORCH_CHECK(
        fw.ndimension() <= 1, "cov(): fweights has more than 1 dimension");
    TORCH_CHECK(
        at::isIntegralType(fw.scalar_type(), false),
        "cov(): fweights must be integral dtype");
    TORCH_CHECK(
        fw.numel() == num_observations,
        "cov(): incompatible number of observations and fweights");
    w = fw;
  }

  if (aweights.has_value()) {
    const auto& aw = aweights.value();
    TORCH_CHECK(
        aw.ndimension() <= 1, "cov(): aweights has more than 1 dimension");
    TORCH_CHECK(
        at::isFloatingType(aw.scalar_type()),
        "cov(): aweights must be floating point dtype");
    TORCH_CHECK(
        aw.numel() == num_observations,
        "cov(): incompatible number of observations and aweights");
    w = w.defined() ? w * aw : aw;
  }

  w = w.defined() ? w.to(kDouble)
                  : at::ones(num_observations, in.options().dtype(kDouble));
  in = in.to(at::promote_types(in.scalar_type(), w.scalar_type()));

  const auto w_sum = w.sum();
  const auto avg = (in * w).sum(1) / w_sum;

  Tensor fact;
  if (correction == 0) {
    fact = w_sum;
  } else if (!aweights.has_value()) {
    fact = w_sum - correction;
  } else {
    fact = w_sum - correction * (w * aweights.value()).sum() / w_sum;
  }
  fact = at::where(fact < 0, at::zeros_like(fact), fact);

  in = in - avg.unsqueeze(1);
  const auto c = at::mm(in, (in * w).t().conj());
  return at::true_divide(c, fact).squeeze();
}

} // namespace native
} // namespace at