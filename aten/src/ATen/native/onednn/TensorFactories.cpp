#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/onednn/ONEDNNCommon.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#endif

namespace at::native {

#if AT_MKLDNN_ENABLED()

Tensor empty_mkldnn(IntArrayRef sizes, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with mkldnn tensor");
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  auto data_type = dtype.has_value() ? get_mkldnn_dtype(dtype.value()) : ideep::tensor::data_type::f32;
  ideep::tensor it {dst_dims, data_type};
  return new_with_itensor_mkldnn(std::move(it), dtype, device);
}

#else

Tensor empty_mkldnn(IntArrayRef sizes, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "empty_mkldnn: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}
