#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

Tensor empty_symint_zendnn(c10::SymIntArrayRef sizes, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return at::native::empty_zendnn(c10::asIntArrayRefSlow(sizes), dtype, layout, device, pin_memory, optional_memory_format);
}

#if AT_ZENDNN_ENABLED()

Tensor empty_zendnn(
    IntArrayRef sizes,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "empty_zendnn: 'memory_format' argument is incompatible with zendnn tensor");
  // NOTE: int32_t dims from zendnn::tensor but sizes needs int64_t
  // TODO: support int64_t dims in zendnn::tensor to avoid extra conversion
  zendnn::tensor::dims dst_dims(sizes.begin(), sizes.end());
  auto data_type = dtype.has_value() ? get_zendnn_dtype(dtype.value())
                                     : zendnn::tensor::data_type::f32;
  zendnn::tensor it{dst_dims, data_type};
  return new_with_itensor_zendnn(std::move(it), dtype, device);
}

#else

Tensor empty_zendnn(
    IntArrayRef sizes,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "empty_zendnn: ATen not compiled with ZENDNN support");
}

#endif // AT_ZENDNN_ENABLED()

} // namespace native
} // namespace at
