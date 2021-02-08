#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

at::Tensor roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned,
    c10::optional<std::vector<at::Tensor>>) {
  std::cout << "XXX roi_align Vulkan" << std::endl;
  std::cout << "XXX roi_align Vulkan scores.sizes():" << input.sizes() << std::endl;
  std::cout << "XXX roi_align Vulkan rois.sizes():" << rois.sizes() << std::endl;
  api::Context* const context = api::context();

  const vTensor& v_input = convert(input);
  const vTensor& v_rois = convert(rois);

  const auto num_rois = rois.size(0);
  const auto roi_cols = rois.size(1);
  const auto channels = input.size(1);
  const auto width = input.size(3);
  const auto height = input.size(2);
  const float offset = aligned ? 0.5f : 0.0f;

  vTensor v_output{
    context,
    {
      num_rois,
      channels,
      pooled_height,
      pooled_width
    },
    input.options(),
  };

  std::cout << "XXX num_rois:" << num_rois << " channels:" << channels << std::endl;

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if C10_LIKELY(v_input.has_image()) {
      const struct Block final {
        ivec2 input_size;
        ivec2 pooled_size;
        int num_rois;
        int roi_cols;
        int channels;
        int sampling_ratio;
        float spatial_scale;
        float offset;
      } block {
        {
          width,
          height
        },
        {
          pooled_width,
          pooled_height
        },
        num_rois,
        roi_cols,
        channels,
        sampling_ratio,
        spatial_scale,
        offset,
      };

      std::cout << "XXX v_output.extents():"
        << v_output.extents().data[0] << "," 
        << v_output.extents().data[1] << "," 
        << v_output.extents().data[2] <<  std::endl;

      std::cout << "XXX dispatch:"
        << pooled_width << ","
        << pooled_height << ","
        << num_rois * channels << std::endl;
      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(roi_align),
          {
            pooled_width,
            pooled_height,
            num_rois * channels
          },
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.buffer(command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.buffer(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_rois.buffer(command_buffer, vTensor::Stage::Compute),
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

//#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(_caffe2, Vulkan, m) {
  m.impl("roi_align", TORCH_FN(roi_align));
}

//#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
