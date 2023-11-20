#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor var_dim_IntList(
    const at::Tensor& self_arg,
    const OptionalIntArrayRef opt_dim,
    bool unbiased = true, // correction=1 in version 2.0
    bool keepdim = false) {
  TORCH_CHECK(
      self_arg.dim() >= 2 && self_arg.dim() <= 4,
      "Vulkan var.dim_IntList only supports 2d, 3d, 4d tensors as input!");

  TORCH_CHECK(
      opt_dim.has_value(), "Vulkan var without a dim arg is not implemented");

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();

  std::set<int64_t> dims_set;
  if (opt_dim.has_value()) {
    int sample_size = 1;
    auto dims = opt_dim.value();

    for (const auto& d : dims) {
      TORCH_CHECK(d >= -self.dim() || d < self.dim(), "Dimension out of range");

      int64_t dim_normalized = utils::normalize(d, self.dim());
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      dims_set.insert(dim_normalized);

      sample_size *= self.sizes().vec()[dim_normalized];
    }

    at::Tensor self_mean = self.mean(opt_dim, true);
    at::Tensor self_minus_mean = self.sub(self_mean);
    // We write `self_minus_mean.mul(self_minus_mean)` instead of
    // `self.sub(self_mean).pow(2)` because Vulkan driver on Android doesn't
    // support negative input: "The result is undefined if x<0 or if x=0 and
    // y≤0" see https://registry.khronos.org/OpenGL-Refpages/gl4/html/pow.xhtml
    at::Tensor output =
        self_minus_mean.mul(self_minus_mean).mean(opt_dim, keepdim);
    if (unbiased == true) {
      output = output.mul(sample_size * 1.0 / (sample_size - 1));
    }
    return output;
  }
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::var.dim"), TORCH_FN(var_dim_IntList));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
