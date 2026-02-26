#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>
#include <dnnl.hpp>

#ifndef DNNL_PREREQ
// oneDNN version check macro
// Checks if the current oneDNN version is >= the specified version
#if defined(DNNL_VERSION_MAJOR) && defined(DNNL_VERSION_MINOR) && \
  defined(DNNL_VERSION_PATCH)
#define DNNL_PREREQ(major, minor, patch) \
  (((DNNL_VERSION_MAJOR << 16) + (DNNL_VERSION_MINOR << 8) + \
   (DNNL_VERSION_PATCH << 0)) >= \
   ((major << 16) + (minor << 8) + (patch << 0)))
#else
#define DNNL_PREREQ(major, minor, patch) 0
#endif
#endif

namespace at::native {

// Mapping ScalarType to oneDNN memory data_type
TORCH_API dnnl::memory::data_type get_mkldnn_dtype(ScalarType type);
static inline dnnl::memory::data_type get_mkldnn_dtype(const Tensor& t) {
  return get_mkldnn_dtype(t.scalar_type());
}

TORCH_API int64_t data_ptr_from_mkldnn(const Tensor& mkldnn_tensor);

TORCH_API at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

// Construct aten MKL-DNN tensor given an ideep tensor
TORCH_API Tensor new_with_itensor_mkldnn(ideep::tensor&& it, std::optional<ScalarType> dtype, std::optional<Device> device);

// Retrieve `ideep::tensor` from MKL-DNN tensor
TORCH_API ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

TORCH_API int64_t nbytes_from_mkldnn(const Tensor& mkldnn_tensor);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
TORCH_API ideep::tensor itensor_view_from_dense(const Tensor& tensor, bool from_const_data_ptr=false);

// Construct an `ideep::tensor` "view" from dense tensor using given desc, note
// the ideep::tensor will share the underlying buffer
TORCH_API ideep::tensor itensor_view_from_dense(
    const at::Tensor& tensor,
    const ideep::tensor::desc& desc);

// Helper function for getting an ideep tensor out of an aten Tensor or MKL-DNN tensor.
TORCH_API ideep::tensor itensor_from_tensor(const Tensor& tensor, bool from_const_data_ptr=false);

// Set MKLDNN verbose level
TORCH_API int set_verbose(int level);

}

#endif // AT_MKLDNN_ENABLED
