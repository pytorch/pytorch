#include <ATen/InferSize.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor view_internal(const Tensor& self_arg, const IntArrayRef shape) {
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);

  at::DimVector inferred_size = at::infer_size_dv(shape, self.numel());

  vTensor v_output{
      context,
      inferred_size,
      self.options(),
  };

  api::StorageBuffer buffer(context, at::kFloat, v_self.numcells(), true);

  utils::pack_vtensor_to_staging(v_self, buffer.buffer());

  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      buffer.buffer(),
      // Previous access
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::WRITE,
      // Next access
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::READ);

  utils::pack_buffer_to_vtensor(buffer.buffer(), v_output, pipeline_barrier);

  return convert(v_output);
}

inline Tensor view(const Tensor& self_arg, const SymIntArrayRef sym_shape) {
  auto shape = c10::asIntArrayRefSlow(sym_shape);
  return view_internal(self_arg, shape);
}

Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  return view_internal(self_arg, shape);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
