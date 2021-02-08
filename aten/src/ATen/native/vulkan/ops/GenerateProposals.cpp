#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

at::Tensor generate_proposals_in_progress(
    const at::Tensor& scores,
    const at::Tensor& bbox_deltas,
    const at::Tensor& im_infos,
    const at::Tensor& anchors,
    double spatial_scale,
    int64_t rpn_pre_nms_topN,
    int64_t post_nms_topN,
    double nms_thresh,
    double rpn_min_size,
    bool angle_bound_on,
    int64_t angle_bound_lo,
    int64_t angle_bound_hi,
    double clip_angle_thresh,
    bool legacy_plus_one) {
  std::cout << "XXX gen_proposals Vulkan" << std::endl;
  const float feat_stride = 1.0 / spatial_scale;
  static const float BBOX_XFORM_CLIP_DEFAULT = std::log(1000.0 / 16.0);

  const auto N = scores.size(0);
  const auto A = scores.size(1);
  const auto height = scores.size(2);
  const auto width = scores.size(3);
  const auto box_dim = anchors.size(1);

  // scores {N, A, H, W}
  // bbox_deltas {N, A * box_dim, H, W}
  // im_info {N, 3}
  // anchors {A, box_dim}

  api::Context* const context = api::context();

  const vTensor& v_scores = convert(scores);
  const vTensor& v_bbox_deltas = convert(bbox_deltas);
  const vTensor& v_im_infos = convert(im_infos);
  const vTensor& v_anchors = convert(anchors);

  // v_output - proposals
  vTensor v_output{
    context,
    {
      N, A * box_dim, height, width
    },
    scores.options(),
  };
  vTensor v_output_keep{
    context,
    {
      N, A, height, width
    },
    scores.options(),
  };
  //TODO: check min_size
  float min_size = 16.0f;

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if C10_LIKELY(v_scores.has_image()) {
      const struct Block final {
        ivec2 size;
        float feat_stride;
        float legacy_plus_one;
        float bbox_xform_clip;
        float min_size;
      } block {
        {
          width,
          height
        },
        feat_stride,
        legacy_plus_one ? 1.0f : 0.0f,
        BBOX_XFORM_CLIP_DEFAULT,
        min_size
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
          VK_KERNEL(gen_proposals),
          {
            width,
            height,
            A * N
          },
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.buffer(command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output_keep.buffer(command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_im_infos.buffer(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_anchors.buffer(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_bbox_deltas.buffer(command_buffer, vTensor::Stage::Compute),
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

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("gen_proposals", TORCH_FN(generate_proposals_in_progress));
}

//#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
