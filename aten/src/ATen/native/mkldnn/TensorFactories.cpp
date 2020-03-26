#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor empty_mkldnn(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
     !options.has_memory_format(),
     "'memory_format' argument is incompatible with mkldnn tensor");
  TORCH_CHECK(
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with mkldnn tensor");
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  ideep::tensor it;
  it.resize<AllocForMKLDNN>(dst_dims, ideep::tensor::data_type::f32);
  return new_with_itensor_mkldnn(std::move(it), options);
}

#else

Tensor empty_mkldnn(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR("empty_mkldnn: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}}
