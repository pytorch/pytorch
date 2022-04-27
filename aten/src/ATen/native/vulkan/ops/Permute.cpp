#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor permute_4d(const Tensor& input, const uvec4& in_size, const uvec4& out_size, const uvec4& out_dims, vTensor& v_output) {
  api::Context* const context = api::context();
  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::permute (permute_4d)");

    auto dst_image = v_output.image(
      command_buffer,
      vTensor::Stage::Compute,
      vTensor::Access::Read | vTensor::Access::Write);

    const Tensor self = input.is_vulkan() ? input : input.vulkan();
    const vTensor& v_self = convert(self);
    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      auto src_image = v_self.image(
              command_buffer,
              vTensor::Stage::Compute);

      const struct Block final {
        uvec3 size;                // output texture size
        uint32_t fill_0;           // dummy
        uvec3 isize;               // input texture size
        uint32_t fill_1;           // dummy
        uvec4 tensor_size;         // output tensor size
        uvec4 itensor_size;        // input tensor size
        uvec4 dims;                // output dims
      } block {
        v_output.extents(),
        0u,
        v_self.extents(),
        0u,
        out_size,
        in_size,
        out_dims,
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(permute_4d),
          // build up shader operations from the output texture point of view
          // to avoid the nondeterministic order of GPU shader operations between texels
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Read/Write access bypasses synchronization but inserts appropriate
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

Tensor permute(const Tensor& self, IntArrayRef dims) {
  auto nDims = safe_downcast<uint32_t>(self.dim());
  TORCH_CHECK(dims.size() == (size_t)nDims,
           "number of dims don't match in permute");

  uvec4 in_size{1u, 1u, 1u, 1u}, out_size{1u, 1u, 1u, 1u};
  uvec4 out_dims{0u, 1u, 2u, 3u};

  auto oldSizes = self.sizes();
  DimVector newSizes(nDims);
  bool sameDims = true;
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = safe_downcast<uint32_t>(maybe_wrap_dim(dims[i], nDims));
    TORCH_CHECK(!seen[dim],
             "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    if (dim != i) {
      sameDims = false;
    }
    // generalize into 4D tensor
    in_size.data[(4u - nDims) + i] = self.sizes()[i];
    out_size.data[(4u - nDims) + i] = self.sizes()[dim];
    out_dims.data[(4u - nDims) + i] = dim + (4u - nDims);
  }

  if (sameDims) {
    return self;
  }

  vTensor v_output{
    api::context(),
    newSizes,
    self.options()};

  return permute_4d(self, in_size, out_size, out_dims, v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::permute"), TORCH_FN(permute));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
