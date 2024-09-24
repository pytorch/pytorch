#include <ATen/native/vulkan/ops/Common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/unsqueeze.h>
#endif

#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor stack(const at::TensorList tensors, const int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "Vulkan stack expects at least one tensor");
  const at::Tensor& tensor = tensors[0];
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Vulkan stack only supports up to 3d tensors as input!");

  TORCH_CHECK(
      dim >= -tensor.dim() - 1 && dim <= tensor.dim(),
      "Vulkan stack dimension out of range expected to be in range of [",
      -tensor.dim() - 1,
      ",",
      tensor.dim(),
      "], but got ",
      dim);

  for (const auto& t : tensors) {
    for (const auto d : c10::irange(t.dim())) {
      TORCH_CHECK(
          t.size(d) == tensor.size(d),
          "Vulkan stack inputs must have matching sizes, received ",
          t.size(d),
          tensor.size(d));
    }
  }

  // Unsqueeze each tensor in the list
  std::vector<Tensor> unsqueezed_outputs;
  for (const auto& t : tensors) {
    unsqueezed_outputs.push_back(at::unsqueeze(t, dim));
  }
  // Cat the tensors
  const at::TensorList tensorList = unsqueezed_outputs;
  return at::cat(tensorList, dim);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::stack"), TORCH_FN(stack));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
