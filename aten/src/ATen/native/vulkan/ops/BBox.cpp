#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

  namespace at {
  namespace native {
  namespace vulkan {
  namespace ops {
  namespace {

  //using namespace api::utils;

  std::tuple<at::Tensor, at::Tensor> bbox_transform(
      const at::Tensor& rois,
      const at::Tensor& deltas,
      const at::Tensor& iminfos,
      float* weights,
      bool apply_scale,
      bool rotated,
      bool angle_bound_on,
      int64_t angle_bound_lo,
      int64_t angle_bound_hi,
      double clip_angle_thresh,
      bool legacy_plus_one) {
    static const float BBOX_XFORM_CLIP_DEFAULT = std::log(1000.0 / 16.0);
    const auto box_dim = rotated ? 5 : 4;
    const auto N = rois.size(0);
    TORCH_CHECK(rois.dim() == 2);
    TORCH_CHECK(rois.size(1) == box_dim || rois.size(1) == box_dim + 1);

    TORCH_CHECK(deltas.dim() == 2);
    TORCH_CHECK(deltas.size(0) == N);
    TORCH_CHECK(deltas.size(1) % box_dim == 0);
    const auto num_classes = deltas.size(1) / box_dim;

    TORCH_CHECK(iminfos.dim() == 2);
    TORCH_CHECK(iminfos.size(1) == 3);
    const auto batch_size = iminfos.size(0);
    
    api::Context* const context = api::context();

    const vTensor& v_rois = convert(rois);
    const vTensor& v_deltas = convert(deltas);
    const vTensor& v_iminfos = convert(iminfos);

    vTensor v_output{
      context,
      {
        N,
        box_dim * num_classes
      },
      deltas.options(),
    };

    api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if C10_LIKELY(v_rois.has_image()) {
      const struct Block final {
        float legacy_plus_one;
        vec4 weights;
        int num_classes;
        float bbox_xform_clip;
      } block {
        legacy_plus_one ? 1.0f : 0.0f,
        {
          weights[0],
          weights[1],
          weights[2],
          weights[3]
        },
        num_classes,
        BBOX_XFORM_CLIP_DEFAULT
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(bbox_transform),
          {
            N,
            num_classes,
            batch_size
          },
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.buffer(command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_rois.buffer(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_deltas.buffer(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_iminfos.buffer(command_buffer, vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  auto roi_batch_splits = at::zeros({batch_size}, rois.options());
  return std::make_tuple(convert(v_output), roi_batch_splits);
}

//#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("bbox_transform", TORCH_FN(bbox_transform));
}

//#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
