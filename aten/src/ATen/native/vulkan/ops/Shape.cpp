#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}

Tensor reshape_copy(const Tensor& self_arg, IntArrayRef shape) {
  api::Context* const context = api::context();

  const vTensor& v_self = convert(self_arg);

  vTensor v_output{
      context,
      shape.vec(),
      self_arg.options(),
  };

  api::Command::Buffer command_buffer =
      api::context()->command().pool.allocate();
  command_buffer.begin();

  command_buffer.copy(
      v_self.buffer(command_buffer),
      v_output.buffer(command_buffer, vTensor::Access::Write));

  command_buffer.end();
  command_buffer.submit(api::context()->gpu().queue);

  return convert(v_output);
}

Tensor cat(const TensorList tensors, int64_t dim) {
  const auto norm_dim = normalize_dim(dim, 4);
  TORCH_INTERNAL_ASSERT(
      norm_dim == 0 || norm_dim == 1,
      "Vulkan cat is implemented only for batch and channels dimensions");

  int64_t cat_dim_size = 0;
  std::vector<vTensor> vTensors{};
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    TORCH_INTERNAL_ASSERT(
        t.dim() == 4, "Vulkan cat expects 4 dimensional inputs");
    TORCH_INTERNAL_ASSERT(t.is_vulkan(), "Vulkan cat expects Vulkan inputs");

    for (int d = 0; d < 4; ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensors[0].size(d),
          "Vulkan cat inputs must have matching sizes except concatenated dimension");
    }
    vTensors.push_back(convert(t));
    cat_dim_size += t.size(dim);
  }

  api::Context* const context = api::context();

  auto result_size = tensors[0].sizes().vec();
  result_size[dim] = cat_dim_size;

  vTensor v_output{
      context,
      result_size,
      vTensors[0].options(),
  };

  api::Command::Buffer command_buffer =
      api::context()->command().pool.allocate();
  command_buffer.begin();

  VkDeviceSize outputOffset = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& tv = vTensors[i];
    command_buffer.copy(
        tv.buffer(command_buffer),
        v_output.buffer(command_buffer, vTensor::Access::Write),
        0,
        outputOffset);
    const auto sizeBytes = sizeof(float) * prod_intlist(tv.sizes());
    outputOffset += sizeBytes;
  }

  command_buffer.end();
  command_buffer.submit(api::context()->gpu().queue);

  return convert(v_output);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  auto sizes = self.sizes().vec();
  sizes.insert(sizes.begin() + dim, 1);
  return reshape_copy(self, sizes);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("view", TORCH_FN(reshape_copy));
  m.impl("cat", TORCH_FN(cat));
  m.impl("unsqueeze", TORCH_FN(unsqueeze));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
