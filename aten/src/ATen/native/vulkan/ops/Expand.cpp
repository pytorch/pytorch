#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/repeat.h>
#endif

#include <ATen/native/vulkan/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor expand(
    const at::Tensor& self,
    const IntArrayRef output_size,
    bool implicit = false) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 4,
      "Vulkan expand supports up to 4d tensors");
  TORCH_CHECK(
      static_cast<size_t>(self.dim()) <= output_size.size(),
      "Vulkan expand: the number of sizes provided (",
      output_size.size(),
      ") must be greater or equal to the number of dimensions in the tensor (",
      self.dim(),
      ").");

  std::vector<int64_t> repeat_size = std::vector<int64_t>(output_size.size());
  std::vector<int64_t> input_size = self.sizes().vec();

  int in_idx = input_size.size() - 1;
  for (int i = output_size.size() - 1; i >= 0; --i) {
    if (in_idx >= 0) {
      TORCH_CHECK(
          input_size[in_idx] == output_size[i] || input_size[in_idx] == 1 ||
              output_size[i] == -1,
          "Vulkan expand: the expanded size of the tensor (",
          output_size[i],
          ") must match the existing size (",
          input_size[in_idx],
          ") at non-singleton dimension ",
          i);

      if (input_size[in_idx] == output_size[i] || output_size[i] == -1) {
        repeat_size[i] = 1;
      } else if (input_size[in_idx] == 1) {
        repeat_size[i] = output_size[i];
      }
      --in_idx;
    } else {
      TORCH_CHECK(
          output_size[i] != -1,
          "Vulkan expand: the expanded size of the tensor (-1) is not allowed in a leading, non-existing dimension 0.");

      repeat_size[i] = output_size[i];
    }
  }

  return self.repeat(repeat_size);
}

Tensor expand_as(const at::Tensor& self, const at::Tensor& other) {
  return expand(self, other.sizes());
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::expand"), TORCH_FN(expand));
  m.impl(TORCH_SELECTIVE_NAME("aten::expand_as"), TORCH_FN(expand_as));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
