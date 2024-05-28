#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

static api::ShaderInfo get_quantize_per_tensor_shader(
    const c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::QUInt8:
      return VK_KERNEL(quantize_per_tensor_quint8);
    case c10::ScalarType::QInt8:
      return VK_KERNEL(quantize_per_tensor_qint8);
    case c10::ScalarType::QInt32:
      return VK_KERNEL(quantize_per_tensor_qint32);
    default:
      TORCH_CHECK(
          false,
          "Vulkan quantization currently not supported for dtype ",
          dtype);
  }
}

Tensor quantize_per_tensor(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype) {
  api::ShaderInfo compute_shader = get_quantize_per_tensor_shader(dtype);

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      v_input.sizes(),
      scale,
      zero_point,
      convert_dtype(dtype),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    float scale;
    float _1;
    int32_t zero_point;
    int32_t _2;
  } block{
      v_output.extents(),
      0u,
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // barrier
      pipeline_barrier,
      // global work group size
      v_input.extents(),
      // local work group size
      adaptive_work_group_size(v_input.extents()),
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

  return convert_quantized(v_output);
}

Tensor quantize_per_tensor_tensor_qparams(
    const at::Tensor& input_arg,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    const c10::ScalarType dtype) {
  TORCH_CHECK(
      (scale.numel() == 1 && zero_point.numel() == 1),
      "Only 1 element expected in scale and zero_point");
  return quantize_per_tensor(
      input_arg, scale.item().toDouble(), zero_point.item().toLong(), dtype);
}

// helper for dequantize function to use scale and zero_point
Tensor dequantize_helper(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype) {
  TORCH_CHECK(dtype == kFloat, "Expected type Float");

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      v_input.sizes(),
      api::kFloat,
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    float scale;
    float _1;
    int32_t zero_point;
    int32_t _2;
  } block{
      v_output.extents(),
      0u,
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(dequantize),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_input.extents(),
      // local work group size
      adaptive_work_group_size(v_input.extents()),
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

double q_scale(const Tensor& self) {
  TORCH_CHECK(self.is_vulkan(), "Expecting a vulkan tensor for q_scale");
  const vTensor& v_input = convert(self);
  return v_input.get_scale();
}

int64_t q_zero_point(const Tensor& self) {
  TORCH_CHECK(self.is_vulkan(), "Expecting a vulkan tensor for q_zero_point");
  const vTensor& v_input = convert(self);
  return v_input.get_zero_point();
}

Tensor dequantize(const Tensor& self) {
  double q_scale = convert(self).get_scale();
  int64_t zero_point = convert(self).get_zero_point();
  return dequantize_helper(self, q_scale, zero_point, kFloat);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::quantize_per_tensor"), quantize_per_tensor);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::quantize_per_tensor.tensor_qparams"),
      quantize_per_tensor_tensor_qparams);
  m.impl(TORCH_SELECTIVE_NAME("aten::q_scale"), q_scale);
  m.impl(TORCH_SELECTIVE_NAME("aten::q_zero_point"), q_zero_point);
  m.impl(TORCH_SELECTIVE_NAME("aten::dequantize.self"), dequantize);
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
