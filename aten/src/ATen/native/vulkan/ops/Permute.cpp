#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>
#include <algorithm>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(self.dim() <= 6, "transpose is implemented only for dim <= 6");
  api::Context* const context = api::context();
  const vTensor& v_self = convert(self);

  auto input_sizes = self.sizes().vec();
  auto output_sizes = input_sizes;
  std::swap(output_sizes[dim0], output_sizes[dim1]);

  for (const auto& osi : output_sizes) {
  }

  std::array<int32_t, 8> input_sizes8;
  input_sizes8.fill(1);
  std::array<int32_t, 8> output_sizes8;
  output_sizes8.fill(1);
  std::copy(input_sizes.cbegin(), input_sizes.cend(), input_sizes8.end() - self.dim());
  std::copy(output_sizes.cbegin(), output_sizes.cend(), output_sizes8.end() - self.dim());
  std::array<int32_t, 8> input_strides;
  input_strides.fill(1);
  std::array<int32_t, 8> output_strides;
  output_strides.fill(1);
  for (int i = 6; i >= 0; --i) {
    input_strides[i] = input_sizes[i + 1] * input_strides[i + 1];
    output_strides[i] = output_sizes[i + 1] * output_strides[i + 1];
  }
  std::swap(input_strides[8 - self.dim() + dim0], input_strides[8 - self.dim() + dim1]);
  vTensor v_output{
    context,
    output_sizes,
    self.options(),
  };

  for (const auto& osi : output_sizes8) {
  }

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    struct {
      uint32_t input_strides[8];
      uint32_t output_strides[8];
      uint32_t output_sizes[8];
      uint32_t storage_offset;
    } block {};
    std::copy(input_strides.cbegin(), input_strides.cend(), std::begin(block.input_strides));
    std::copy(output_strides.cbegin(), output_strides.cend(), std::begin(block.output_strides));
    std::copy(output_sizes.cbegin(), output_sizes.cend(), std::begin(block.output_sizes));
    block.storage_offset = 0;
    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(permute),
        //v_output.extents(),
        {
            output_sizes8[6] * output_sizes8[7],
            output_sizes8[4] * output_sizes8[5],
            output_sizes8[2] * output_sizes8[3]
        },
        // Write-only access bypasses synchronization but inserts appropriate
        // barriers if necessary.
        v_output.buffer(command_buffer, vTensor::Access::Write),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_self.buffer(command_buffer),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);
  return convert(v_output);
}

Tensor slice(
    const Tensor& self,
    int64_t dim,
    int64_t start_arg,
    int64_t end_arg,
    int64_t step) {
  TORCH_CHECK(self.dim() <= 6, "slice is implemented only for dim <= 6");
  api::Context* const context = api::context();
  const vTensor& v_self = convert(self);
  const auto input_sizes = self.sizes().vec();
  auto output_sizes = input_sizes;
  auto start = start_arg;
  auto end = end_arg;
  if (start < 0) {
    start += input_sizes[dim];
  }
  if (end < 0) {
    end += input_sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= input_sizes[dim]) {
    start = input_sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= input_sizes[dim]) {
    end = input_sizes[dim];
  }
  const auto len = end - start;
  output_sizes[dim] = (len + step - 1) / step;
  std::array<int32_t, 8> input_sizes8;
  input_sizes8.fill(1);
  std::copy(input_sizes.cbegin(), input_sizes.cend(), input_sizes8.end() - self.dim());
  std::array<int32_t, 8> input_strides;
  input_strides.fill(1);
  for (int i = 6; i >= 0; --i) {
    input_strides[i] = input_sizes[i + 1] * input_strides[i + 1];
  }
  std::array<int32_t, 8> output_sizes8 = input_sizes8;
  std::array<int32_t, 8> output_strides = input_strides;

  output_strides[8 - self.dim() + dim] *= step;
  vTensor v_output{
    context,
    output_sizes,
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    struct {
      uint32_t input_strides[8];
      uint32_t output_strides[8];
      uint32_t output_sizes[8];
      uint32_t storage_offset;
    } block {};
    std::copy(input_strides.cbegin(), input_strides.cend(), std::begin(block.input_strides));
    std::copy(output_strides.cbegin(), output_strides.cend(), std::begin(block.output_strides));
    std::copy(output_sizes8.cbegin(), output_sizes8.cend(), std::begin(block.output_sizes));
    block.storage_offset = start * input_strides[8 - self.dim() + dim];

    context->dispatch(
        command_buffer,
        {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
        VK_KERNEL(permute),
        {
            output_sizes8[6] * output_sizes8[7],
            output_sizes8[4] * output_sizes8[5],
            output_sizes8[2] * output_sizes8[3]
        },
        // Write-only access bypasses synchronization but inserts appropriate
        // barriers if necessary.
        v_output.buffer(command_buffer, vTensor::Access::Write),
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_self.buffer(command_buffer),
        // Object lifetime is managed by the resource pool.
        // It is OK not to keep track of the handle.
        context->resource().pool.uniform(block).object);
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);
  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("slice.Tensor", TORCH_FN(slice));
  m.impl("transpose.int", TORCH_FN(transpose));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
