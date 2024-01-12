#include <ATen/native/vulkan/impl/Arithmetic.h>
#include <ATen/native/vulkan/impl/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace arithmetic {

api::ShaderInfo get_shader(const OpType type) {
  switch (type) {
    case OpType::ADD:
      return VK_KERNEL(add);
    case OpType::SUB:
      return VK_KERNEL(sub);
    case OpType::MUL:
      return VK_KERNEL(mul);
    case OpType::DIV:
      return VK_KERNEL(div);
  }
  VK_THROW("Invalid OpType");
}

struct Params final {
  api::utils::ivec4 outputSizes;
  api::utils::ivec4 input1Sizes;
  api::utils::ivec4 input2Sizes;
  float alpha;
};

void record_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    vTensor& v_in1,
    vTensor& v_in2,
    vTensor& v_dst,
    const float alpha) {
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  Params block{
      api::utils::make_ivec4(
          {dim_at<Dim4D::Width>(v_dst),
           dim_at<Dim4D::Height>(v_dst),
           dim_at<Dim4D::Channel>(v_dst),
           dim_at<Dim4D::Batch>(v_dst)}),
      api::utils::make_ivec4(
          {dim_at<Dim4D::Width>(v_in1),
           dim_at<Dim4D::Height>(v_in1),
           dim_at<Dim4D::Channel>(v_in1),
           dim_at<Dim4D::Batch>(v_in1)}),
      api::utils::make_ivec4(
          {dim_at<Dim4D::Width>(v_in2),
           dim_at<Dim4D::Height>(v_in2),
           dim_at<Dim4D::Channel>(v_in2),
           dim_at<Dim4D::Batch>(v_in2)}),
      alpha,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_in1.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_in2.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
}

} // namespace arithmetic
} // namespace vulkan
} // namespace native
} // namespace at
