#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor _clamp(
    const Tensor& self_arg,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(min || max, "At least one of 'min' or 'max' must not be None");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self_arg);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    vec2 clamp;
  } block{
      v_output.extents(),
      0u,
      {
          min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
          max ? max->to<float>() : std::numeric_limits<float>::infinity(),
      },
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

Tensor clamp(
    const Tensor& self_arg,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  return _clamp(self_arg, min, max, VK_KERNEL(clamp));
}

Tensor& _clamp_(
    Tensor& self_arg,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(min || max, "At least one of 'min' or 'max' must not be None");

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place clamp is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    vec2 clamp;
  } block{
      v_self.extents(),
      0u,
      {
          min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
          max ? max->to<float>() : std::numeric_limits<float>::infinity(),
      },
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

Tensor threshold(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  return _clamp(self, threshold, value, VK_KERNEL(threshold));
}

Tensor& clamp_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  return _clamp_(self, min, max, VK_KERNEL(clamp_));
}

Tensor activation(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
  } block{
      v_output.extents(),
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
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& activation_(
    Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const struct Block final {
    uvec3 extents;
    uint32_t _;
  } block{
      v_self.extents(),
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

Tensor hardtanh(const Tensor& self, const Scalar& min, const Scalar& max) {
  return ops::_clamp(self, min, max, VK_KERNEL(clamp));
}

Tensor& hardtanh_(Tensor& self, const Scalar& min, const Scalar& max) {
  return ops::_clamp_(self, min, max, VK_KERNEL(clamp_));
}

Tensor relu(const Tensor& self) {
  return ops::_clamp(self, 0, c10::nullopt, VK_KERNEL(clamp));
}

Tensor& relu_(Tensor& self) {
  return ops::_clamp_(self, 0, c10::nullopt, VK_KERNEL(clamp_));
}

Tensor hardswish(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardswish));
}

Tensor& hardswish_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardswish_));
}

Tensor hardsigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardsigmoid));
}

Tensor& hardsigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardsigmoid_));
}

Tensor activation_scalar(
    const Tensor& self_arg,
    const std::vector<Scalar>& scalar_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  api::UniformParamsBuffer params;

  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  if (scalar_arg.size() == 1) {
    if (v_self.is_quantized()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
        float scale;
        int zero_point;
      } block{
          v_output.extents(),
          0u,
          scalar_arg[0].to<float>(),
          safe_downcast<float>(v_self.get_scale()),
          safe_downcast<int32_t>(v_self.get_zero_point()),
      };
      params = api::UniformParamsBuffer(context, block);
    } else {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block{
          v_output.extents(),
          0u,
          scalar_arg[0].to<float>(),
      };
      params = api::UniformParamsBuffer(context, block);
    }
  } else {
    const struct Block final {
      uvec3 extents;
      uint32_t _;
      float scalar_value1;
      float scalar_value2;
    } block{
        v_output.extents(),
        0u,
        scalar_arg[0].to<float>(),
        scalar_arg[1].to<float>(),
    };
    params = api::UniformParamsBuffer(context, block);
  }

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

Tensor& activation_scalar_(
    Tensor& self_arg,
    const std::vector<Scalar>& scalar_arg,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  api::UniformParamsBuffer params;

  if (scalar_arg.size() == 1) {
    if (v_self.is_quantized()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
        float scale;
        int zero_point;
      } block{
          v_self.extents(),
          0u,
          scalar_arg[0].to<float>(),
          safe_downcast<float>(v_self.get_scale()),
          safe_downcast<int32_t>(v_self.get_zero_point()),
      };
      params = api::UniformParamsBuffer(context, block);
    } else {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block{
          v_self.extents(),
          0u,
          scalar_arg[0].to<float>(),
      };
      params = api::UniformParamsBuffer(context, block);
    }
  } else {
    const struct Block final {
      uvec3 extents;
      uint32_t _;
      float scalar_value1;
      float scalar_value2;
    } block{
        v_self.extents(),
        0u,
        scalar_arg[0].to<float>(),
        scalar_arg[1].to<float>(),
    };
    params = api::UniformParamsBuffer(context, block);
  }

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

Tensor gelu(const Tensor& self, c10::string_view approximate) {
  TORCH_CHECK(
      approximate == "tanh", "Vulkan: gelu only supported for tanh type");
  Scalar kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5;
  std::vector<Scalar> scalar;
  scalar.push_back(kBetaVec);

  if (self.scalar_type() == at::kQUInt8) {
    return ops::activation_scalar(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_quint8));
  }

  if (self.scalar_type() == at::kQInt8) {
    return ops::activation_scalar(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_qint8));
  }

  return ops::activation_scalar(self, scalar, VK_KERNEL(gelu_tanh));
}

Tensor& gelu_(Tensor& self, c10::string_view approximate) {
  TORCH_CHECK(
      approximate == "tanh", "Vulkan: gelu only supported for tanh type");
  Scalar kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5;
  std::vector<Scalar> scalar;
  scalar.push_back(kBetaVec);

  if (self.scalar_type() == at::kQUInt8) {
    return ops::activation_scalar_(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_quint8_));
  }

  if (self.scalar_type() == at::kQInt8) {
    return ops::activation_scalar_(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_qint8_));
  }

  return ops::activation_scalar_(self, scalar, VK_KERNEL(gelu_tanh_));
}

Tensor hardshrink(const Tensor& self_arg, const Scalar& lambd) {
  float abs_lambd = std::abs(lambd.to<float>());
  std::vector<Scalar> scalar;
  scalar.push_back(abs_lambd);
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(hardshrink));
}

Tensor& hardshrink_(Tensor& self, const Scalar& lambd) {
  float abs_lambd = std::abs(lambd.to<float>());
  std::vector<Scalar> scalar;
  scalar.push_back(abs_lambd);
  return ops::activation_scalar_(self, scalar, VK_KERNEL(hardshrink_));
}

Tensor leaky_relu(const Tensor& self_arg, const Scalar& negative_slope) {
  std::vector<Scalar> scalar;
  scalar.push_back(negative_slope);
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(leaky_relu));
}

Tensor& leaky_relu_(Tensor& self, const Scalar& negative_slope) {
  std::vector<Scalar> scalar;
  scalar.push_back(negative_slope);
  return ops::activation_scalar_(self, scalar, VK_KERNEL(leaky_relu_));
}

Tensor sigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(sigmoid));
}

Tensor& sigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(sigmoid_));
}

Tensor tanh(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(tanh));
}

Tensor& tanh_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(tanh_));
}

Tensor abs(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(abs));
}

Tensor& abs_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(abs_));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::clamp"), TORCH_FN(clamp));
  m.impl(TORCH_SELECTIVE_NAME("aten::clamp_"), TORCH_FN(clamp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::gelu"), gelu);
  m.impl(TORCH_SELECTIVE_NAME("aten::gelu_"), gelu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid"), hardsigmoid);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid_"), hardsigmoid_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink"), hardshrink);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink_"), hardshrink_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardswish"), hardswish);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardswish_"), hardswish_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh"), hardtanh);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh_"), hardtanh_);
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu"), leaky_relu);
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu_"), leaky_relu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid"), sigmoid);
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid_"), sigmoid_);
  m.impl(TORCH_SELECTIVE_NAME("aten::tanh"), tanh);
  m.impl(TORCH_SELECTIVE_NAME("aten::tanh_"), tanh_);
  m.impl(TORCH_SELECTIVE_NAME("aten::abs"), abs);
  m.impl(TORCH_SELECTIVE_NAME("aten::abs_"), abs_);
  m.impl(TORCH_SELECTIVE_NAME("aten::relu"), relu);
  m.impl(TORCH_SELECTIVE_NAME("aten::relu_"), relu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::threshold"), threshold);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
