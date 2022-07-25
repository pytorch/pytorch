#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

bool broadcast_input(const Tensor& input1, const Tensor& input2) {
  return ((height_size(input1) > 1 && height_size(input2) == 1) ||
          (height_size(input2) > 1 && height_size(input1) == 1) ||
          (height_size(input1) == height_size(input2))) &&
      ((width_size(input1) > 1 && width_size(input2) == 1) ||
       (width_size(input2) > 1 && width_size(input1) == 1) ||
       (width_size(input1) == width_size(input2)));
}

void check_inputs(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      channels_size(input1) == channels_size(input2),
      "Vulkan binary elementwise ops require channel dimension to be equal!");
  if (batch_size(input1) != batch_size(input2)) {
    TORCH_CHECK(
        channels_size(input1) % 4 == 0,
        "Vulkan binary elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
  }

  const std::string broadcast_error_msg =
      "Incompatible input dimensions for broadcasting for Vulkan binary elementwise op!";

  TORCH_CHECK(broadcast_input(input1, input2), broadcast_error_msg);
}

std::vector<int64_t> broadcast_size(
    const Tensor& input1,
    const Tensor& input2) {
  std::vector<int64_t> out = {};
  int input1_size = input1.sizes().size();
  int input2_size = input2.sizes().size();
  if (input1_size > input2_size) {
    for (int i = 0; i < input1_size; i++) {
      out.push_back(input1.sizes()[i]);
    }
  } else {
    for (int i = 0; i < input2_size; i++) {
      out.push_back(input2.sizes()[i]);
    }
  }

  if (width_size(input1) > 1 && width_size(input2) == 1) {
    out[out.size() - 1] = width_size(input1);
  } else if (width_size(input2) > 1 && width_size(input1) == 1) {
    out[out.size() - 1] = width_size(input2);
  }

  if (out.size() > 1) {
    if (height_size(input1) > 1 && height_size(input2) == 1) {
      out[out.size() - 2] = height_size(input1);
    } else if (height_size(input2) > 1 && height_size(input1) == 1) {
      out[out.size() - 2] = height_size(input2);
    }
  }

  return out;
}
} // namespace
using namespace api::utils;

Tensor arithmetic_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const c10::optional<Scalar>& alpha_arg,
    const api::ShaderSource& shader_descriptor) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.options(),
  };

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    float other;
  } block{
      v_self.extents(),
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& arithmetic_scalar_(
    Tensor& self_arg,
    const Scalar& other,
    const c10::optional<Scalar>& alpha_arg,
    const api::ShaderSource& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    float other;
  } block{
      v_self.extents(),
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor arithmetic_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const c10::optional<Scalar>& alpha_arg,
    const api::ShaderSource& shader_descriptor) {
  check_inputs(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      broadcast_size(self_arg, other_arg),
      v_self.options(),
  };

  const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0;
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    float alpha;
  } block{
      v_output.extents(),
      0u,
      v_self.extents(),
      0u,
      v_other.extents(),
      alpha,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor quantized_arithmetic_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point,
    const api::ShaderSource& shader_descriptor) {
  check_inputs(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);
  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  TORCH_CHECK(v_self.is_quantized(), "Input tensor is not quantized");
  TORCH_CHECK(v_other.is_quantized(), "Input tensor is not quantized");

  vTensor v_output{
      context,
      broadcast_size(self_arg, other_arg),
      self.options().dtype(c10::kQUInt8),
      scale,
      zero_point};

  const double scale1 = v_self.get_scale();
  const double scale2 = v_other.get_scale();
  const int64_t zero_point1 = v_self.get_zero_point();
  const int64_t zero_point2 = v_other.get_zero_point();
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    uint32_t fill_2;
    float scale1;
    float scale2;
    int32_t zero_point1;
    int32_t zero_point2;
    float scale;
    float _1;
    int32_t zero_point;
    int32_t _2;
  } block{
      v_output.extents(),
      0u,
      v_self.extents(),
      0u,
      v_other.extents(),
      0u,
      safe_downcast<float>(scale1),
      safe_downcast<float>(scale2),
      safe_downcast<int32_t>(zero_point1),
      safe_downcast<int32_t>(zero_point2),
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert_quantized(v_output);
}

Tensor& arithmetic_tensor_(
    Tensor& self_arg,
    const Tensor& other_arg,
    const c10::optional<Scalar>& alpha_arg,
    const api::ShaderSource& shader_descriptor) {
  check_inputs(self_arg, other_arg);

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0;
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input_extents;
    float alpha;
  } block{
      v_self.extents(),
      0u,
      v_other.extents(),
      alpha,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return arithmetic_scalar(
      self_arg, other, c10::optional<Scalar>(alpha), VK_KERNEL(add_scalar));
}

Tensor& add_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return arithmetic_scalar_(
      self, other, c10::optional<Scalar>(alpha), VK_KERNEL(add_scalar_));
}

Tensor quantized_add(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_arithmetic_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_add));
}

Tensor quantized_sub(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_arithmetic_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_sub));
}

Tensor quantized_mul(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_arithmetic_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_mul));
}

Tensor quantized_div(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_arithmetic_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_div));
}

Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        other_arg.item<float>(),
        c10::optional<Scalar>(alpha.to<float>()),
        VK_KERNEL(add_scalar));
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(add));
}

Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(add_));
}

Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return arithmetic_scalar(
      self_arg,
      other,
      c10::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar));
}

Tensor& sub_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return arithmetic_scalar_(
      self,
      other,
      c10::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar_));
}

Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        other_arg.item<float>(),
        c10::optional<Scalar>(-1 * alpha.to<float>()),
        VK_KERNEL(add_scalar));
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(sub));
}

Tensor& sub_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(sub_));
}

Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  return arithmetic_scalar(
      self_arg, other, c10::optional<Scalar>(), VK_KERNEL(mul_scalar));
}

Tensor& mul_scalar_(Tensor& self, const Scalar& other) {
  return arithmetic_scalar_(
      self, other, c10::optional<Scalar>(), VK_KERNEL(mul_scalar_));
}

Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        other_arg.item<float>(),
        c10::optional<Scalar>(),
        VK_KERNEL(mul_scalar));
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(), VK_KERNEL(mul));
}

Tensor& mul_tensor_(Tensor& self, const Tensor& other_arg) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(), VK_KERNEL(mul_));
}

Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  return arithmetic_scalar(
      self_arg,
      1.0 / other.to<float>(),
      c10::optional<Scalar>(),
      VK_KERNEL(mul_scalar));
}

Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  return arithmetic_scalar_(
      self,
      1.0 / other.to<float>(),
      c10::optional<Scalar>(),
      VK_KERNEL(mul_scalar_));
}

Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        1.0 / other_arg.item<float>(),
        c10::optional<Scalar>(),
        VK_KERNEL(mul_scalar));
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(), VK_KERNEL(div));
}

Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(), VK_KERNEL(div_));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
