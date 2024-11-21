#include <ATen/native/vulkan/ops/Common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/unsqueeze.h>
#endif

#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor repeat(const Tensor& self, const IntArrayRef repeats) {
  TORCH_CHECK(
      self.dim() <= 4, "Vulkan repeat only supports tensors <= 4 dimensions");
  auto in_ndims = safe_downcast<uint32_t>(self.dim());
  auto out_ndims = safe_downcast<uint32_t>(repeats.size());
  TORCH_CHECK(
      out_ndims >= in_ndims,
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
  auto add_ndims = out_ndims - in_ndims;

  at::Tensor tensor_to_repeat = self.clone();

  for (const auto i : c10::irange(add_ndims)) {
    (void)i;
    tensor_to_repeat = at::unsqueeze(tensor_to_repeat, 0);
  }

  std::vector<at::Tensor> tensor_seq_to_concat;
  for (const auto i : c10::irange(out_ndims)) {
    for (const auto k : c10::irange(repeats[i])) {
      (void)k;
      tensor_seq_to_concat.emplace_back(tensor_to_repeat.clone());
    }
    tensor_to_repeat = at::cat(tensor_seq_to_concat, i);
    tensor_seq_to_concat.clear();
  }
  return tensor_to_repeat;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::repeat"), TORCH_FN(repeat));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
