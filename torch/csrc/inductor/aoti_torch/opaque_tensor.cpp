#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/opaque_tensor.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ideep.hpp>
#endif

namespace torch {
namespace aot_inductor {

#if AT_MKLDNN_ENABLED()

void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor) {
  return reinterpret_cast<void*>(
      at::native::data_ptr_from_mkldnn(*mkldnn_tensor));
}

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  std::vector<uint8_t> vector_serialized_md{
      opaque_metadata, opaque_metadata + opaque_metadata_size};
  ideep::tensor::desc deserialized_ideep_desc;
#if IDEEP_PREREQ(3, 4, 1, 2)
  // groups is needed for grouped conv
  deserialized_ideep_desc = ideep::tensor::desc(vector_serialized_md);
#else
  TORCH_CHECK(false, "Unexpected IDeep version to do weight deserialization.");
#endif

  auto a = ideep::tensor(deserialized_ideep_desc, data_ptr);
  return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);
}

#else

void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

#endif

} // namespace aot_inductor
} // namespace torch
