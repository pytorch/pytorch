#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
Tensor select_batch_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[1], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (c, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, c / 4)[c % 4]
  */
  const struct Block final {
    ivec2 batch_info;
  } block{
      {static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index)}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_batch_4d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select_depth_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_depth_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select_depth_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_depth_4d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select_height_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2]},
      v_input.dtype(),
  };
  // Input tensor is a (c, h, w)
  // Output tensor is a (c, w)
  // In shader, the input texture's coordinate is (w, h, c)
  // In shader, the output texture's coordinate is (w, c, 1)
  uint32_t w = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];
  uint32_t z = 1;
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(w),
       static_cast<int32_t>(c),
       static_cast<int32_t>(z),
       static_cast<int32_t>(index)}};

  // Encoding of c-channel is packed into texel, hence we only call ceil(c/4)
  // times to minimize invocation and read.
  // For the last dimension, it is the selected height. Shader will do a direct
  // lookup based on block.index.
  uvec3 global_workgroup_size{w, api::utils::div_up(c, 4u), z};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_height_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_size,
      // local work group size
      adaptive_work_group_size(global_workgroup_size),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select_height_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, c, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, c, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_height_4d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select_width_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1]},
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 width_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  // Input tensor is a (c, h, w)
  // Output tensor is a (c, h)
  // In shader, the input texture's coordinate is (w, h, c)
  // In shader, the output texture's coordinate is (h, c, 1)
  uint32_t h = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];

  // Encoding of c-channel is packed into texel, hence we only call ceil(c/4)
  // times to minimize invocation and read.
  // For the last dimension, it is the selected width. Shader will do a direct
  // lookup based on block.index.
  uvec3 global_workgroup_size{h, api::utils::div_up(c, 4u), 1};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_width_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_size,
      // local work group size
      adaptive_work_group_size(global_workgroup_size),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select_width_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, c, h)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (h, c, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 width_info;
  } block{
      static_cast<int32_t>(v_input_sizes[0]),
      static_cast<int32_t>(std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
      static_cast<int32_t>(index),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_width_4d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  TORCH_CHECK(
      self.dim() == 3 || self.dim() == 4,
      "Vulkan select only supports 3d and 4d tensors!");

  const int64_t size = self.size(dim);

  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.dim() == 3) {
    if (dim == 0) {
      return select_depth_3d(self, index);
    } else if (dim == 1) {
      return select_height_3d(self, index);
    } else {
      return select_width_3d(self, index);
    }
  } else { // self.dim() == 4
    if (dim == 0) {
      return select_batch_4d(self, index);
    } else if (dim == 1) {
      return select_depth_4d(self, index);
    } else if (dim == 2) {
      return select_height_4d(self, index);
    } else {
      return select_width_4d(self, index);
    }
  }
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::select.int"), TORCH_FN(select));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
