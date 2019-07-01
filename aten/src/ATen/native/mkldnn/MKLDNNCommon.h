#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

// Custom allocator using c10 CPU allocator for `ideep::tensor`
struct AllocForMKLDNN {
  static char* malloc(size_t size) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    return (char*)allocator->raw_allocate(size);
  }

  static void free(void* p) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    allocator->raw_deallocate(p);
  }
};

inline ideep::tensor::data_type get_mkldnn_dtype(ScalarType type, int64_t zero_point = 0) {
  switch (type) {
    case kFloat:
      return ideep::tensor::data_type::f32;
    case kQInt32:
      return ideep::tensor::data_type::s32;
    case kQUInt8:
      if (zero_point == 0)
        return ideep::tensor::data_type::u8;
      else
        return ideep::tensor::data_type::s8;
    case kQInt8:
      return ideep::tensor::data_type::s8;
    default:
      AT_ERROR("get_mkldnn_dtype: unsupported data type");
  }
}

// Construct aten MKL-DNN tensor given an ideep tensor
Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options);

// Retrieve `ideep::tensor` from MKL-DNN tensor
ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(const Tensor& tensor);

// Convert zero_point scales to min_max scales
// NOTE:
//  The scales in operator is saved in pytorch format,
//  while pytorch scales are the reciprocals of MKL-DNN scales.
//  This function is provided to convert scales from pytorch to MKL-DNN
inline ideep::scale_t ConvertScales(
    const std::vector<float>& scales_z) {
  ideep::scale_t scales (scales_z);
  for (auto it = scales.begin(); it != scales.end(); it++) {
    *it = 1.0f / *it;
  }
  return scales;
}

}}

#endif // AT_MKLDNN_ENABLED
