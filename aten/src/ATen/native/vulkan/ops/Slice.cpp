#include <ATen/NamedTensorUtils.h>
#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/api/Helper.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor slice_4d(const Tensor& input, const int64_t dim, const int64_t start, const int64_t end,
                const int64_t step, const uvec4& in_tsize, const uvec4& out_tsize, vTensor& v_output) {
  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::slice.Tensor (slice_4d)");

    const Tensor self = input.is_vulkan() ? input : input.vulkan();
    const vTensor& v_self = convert(self);
    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      auto src_image = v_self.image(
              command_buffer,
              vTensor::Stage::Compute);
      auto dst_image = v_output.image(
        command_buffer,
        vTensor::Stage::Compute,
        vTensor::Access::Write);

      const struct Block final {
        uvec3 size;                // output texture size
        uint32_t fill_0;           // dummy
        uvec3 isize;               // input texture size
        uint32_t fill_1;           // dummy
        uvec4 tensor_size;         // output tensor size
        uvec4 itensor_size;        // input tensor size
        uvec4 args;                // input arguments (dim, start, end, step)
      } block {
        v_output.extents(),
        0u,
        v_self.extents(),
        0u,
        out_tsize,
        in_tsize,
        { safe_downcast<uint32_t>(dim),
          safe_downcast<uint32_t>(start),
          safe_downcast<uint32_t>(end),
          safe_downcast<uint32_t>(step) },
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(slice_4d),
          // build up shader operations from the output texture point of view
          // to avoid the nondeterministic order of GPU shader operations between texels
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          dst_image,
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          src_image,
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);
  return convert(v_output);
}

Tensor slice_width(const Tensor& input, const int64_t start, const int64_t end, const int64_t step, vTensor& v_output) {
  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::slice.Tensor (slice_width)");

    const Tensor self = input.is_vulkan() ? input : input.vulkan();
    const vTensor& v_self = convert(self);
    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      auto src_image = v_self.image(
              command_buffer,
              vTensor::Stage::Transfer);
      auto dst_image = v_output.image(
        command_buffer,
        vTensor::Stage::Transfer,
        vTensor::Access::Write);

      uvec3 src_offset{};
      uvec3 dst_offset{};

      if (step == 1) {
        src_offset.data[0u] = start;
        uvec3 copy_extents {safe_downcast<uint32_t>(end - start),
          v_self.extents().data[1u],
          v_self.extents().data[2u]};
        api::helper::copy_texture_to_texture(command_buffer,
          src_image,
          dst_image,
          copy_extents,
          src_offset,
          dst_offset);
      } else {
        uvec3 copy_extents {1u,
          v_self.extents().data[1u],
          v_self.extents().data[2u]};
        const auto x_max = v_self.extents().data[0u];
        for (int64_t x = start, x_new = 0; x < end; x += step, ++x_new) {
          if (x >= x_max) { // out of range
            continue;
          }
          src_offset.data[0u] = x;
          dst_offset.data[0u] = x_new;
          api::helper::copy_texture_to_texture(command_buffer,
            src_image,
            dst_image,
            copy_extents,
            src_offset,
            dst_offset);
        }
      }
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);
  return convert(v_output);
}

Tensor slice_height(const Tensor& input, const int64_t start, const int64_t end, const int64_t step, vTensor& v_output) {
  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::slice.Tensor (slice_height)");

    const Tensor self = input.is_vulkan() ? input : input.vulkan();
    const vTensor& v_self = convert(self);
    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      auto src_image = v_self.image(
              command_buffer,
              vTensor::Stage::Transfer);
      auto dst_image = v_output.image(
        command_buffer,
        vTensor::Stage::Transfer,
        vTensor::Access::Write);

      uvec3 src_offset{};
      uvec3 dst_offset{};

      if (step == 1) {
        src_offset.data[1u] = start;
        uvec3 copy_extents {v_self.extents().data[0u],
          safe_downcast<uint32_t>(end - start),
          v_self.extents().data[2u]};
        api::helper::copy_texture_to_texture(command_buffer,
          src_image,
          dst_image,
          copy_extents,
          src_offset,
          dst_offset);
      } else {
        uvec3 copy_extents {v_self.extents().data[0u],
          1u,
          v_self.extents().data[2u]};
        const auto y_max = v_self.extents().data[1u];
        for (int64_t y = start, y_new = 0; y < end; y += step, ++y_new) {
          if (y >= y_max) { // out of range
            continue;
          }
          src_offset.data[1u] = y;
          dst_offset.data[1u] = y_new;
          api::helper::copy_texture_to_texture(command_buffer,
            src_image,
            dst_image,
            copy_extents,
            src_offset,
            dst_offset);
        }
      }
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);
  return convert(v_output);
}

Tensor slice(
    const Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    const int64_t step) {
  TORCH_CHECK(step > 0, "slice step must be positive");
  auto nDims = safe_downcast<uint32_t>(self.dim());
  dim = maybe_wrap_dim(dim, nDims);
  DimVector newSizes(self.sizes().begin(), self.sizes().end());

  // handle optional parameters
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  // INT64_MAX stands for default value.
  if (start_val == INT64_MAX) {
    start_val = 0;
  }
  if (start_val < 0) {
    start_val += newSizes[dim];
  }
  if (end_val < 0) {
    end_val += newSizes[dim];
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= newSizes[dim]) {
    start_val = newSizes[dim];
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= newSizes[dim]) {
    end_val = newSizes[dim];
  }

  auto len = end_val - start_val;
  newSizes[dim] = (len + step - 1) / step; // round-up
  TORCH_CHECK(len > 0, "Vulkan doesn't support zero-sized slice");

  // generalize into 4D tensor
  uvec4 in_tsize{1u, 1u, 1u, 1u}, out_tsize{1u, 1u, 1u, 1u};
  for (const auto i : c10::irange(nDims)) {
    in_tsize.data[(4u - nDims) + i] = self.sizes()[i];
    out_tsize.data[(4u - nDims) + i] = newSizes[i];
  }
  dim += 4 - nDims;

  vTensor v_output{
    api::context(),
    newSizes,
    self.options()};

  if (dim == 3) {
    slice_width(self, start_val, end_val, step, v_output);
  }
  else if (dim == 2) {
    slice_height(self, start_val, end_val, step, v_output);
  }
  else {
    slice_4d(self, dim, start_val, end_val, step, in_tsize, out_tsize, v_output);
  }

  auto result = convert(v_output);
  namedinference::propagate_names(result, self);
  return result;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::slice.Tensor"), TORCH_FN(slice));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
