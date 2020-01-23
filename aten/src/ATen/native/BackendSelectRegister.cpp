#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorOptions.h>

// TODO:
// 1) add all factory functions
// 2) autogen this

namespace at {
namespace native {
  Tensor empty_backend(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
    DispatchKey key = options.computeDispatchKey();
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty", "memory_format");
    return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>>(key, size, options, memory_format);
  }

  Tensor empty_strided_backend(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) {
    DispatchKey key = options.computeDispatchKey();
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty_strided", "");
    return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, IntArrayRef, const TensorOptions &>(key, size, stride, options);
  }

  Tensor empty_affine_quantized_backend(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) {
    DispatchKey key = options.computeDispatchKey();
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_empty_affine_quantized", "");
    return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>>(key, size, options, scale, zero_point, memory_format);
  }

  Tensor empty_per_channel_affine_quantized_quantized_backend(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
    DispatchKey key = options.computeDispatchKey();
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_empty_per_channel_affine_quantized", "");
    return op.callUnboxedWithDispatchKey<Tensor, IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(key, size, scales, zero_points, axis, options, memory_format);
  }

  static auto registry = torch::RegisterOperators()
    .op(torch::RegisterOperators::options()
      .schema("aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(empty_backend), &empty_backend>(DispatchKey::BackendSelectId)
      .aliasAnalysis(AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options()
      .schema("aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(empty_strided_backend), &empty_strided_backend>(DispatchKey::BackendSelectId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options()
      .schema("aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(empty_affine_quantized_backend), &empty_affine_quantized_backend>(DispatchKey::BackendSelectId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    .op(torch::RegisterOperators::options()
      .schema("aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor")
      .impl_unboxedOnlyKernel<decltype(empty_per_channel_affine_quantized_quantized_backend), &empty_per_channel_affine_quantized_quantized_backend>(DispatchKey::BackendSelectId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
}
}
