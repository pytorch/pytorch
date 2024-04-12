#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

#ifndef IDEEP_PREREQ
// Please find definitions of version numbers in ideep.hpp
#if defined(IDEEP_VERSION_MAJOR) && defined(IDEEP_VERSION_MINOR) && \
  defined(IDEEP_VERSION_PATCH) && defined(IDEEP_VERSION_REVISION)
#define IDEEP_PREREQ(major, minor, patch, revision) \
  (((IDEEP_VERSION_MAJOR << 16) + (IDEEP_VERSION_MINOR << 8) + \
   (IDEEP_VERSION_PATCH << 0)) >= \
   ((major << 16) + (minor << 8) + (patch << 0)) && \
   (IDEEP_VERSION_REVISION >= revision))
#else
#define IDEEP_PREREQ(major, minor, patch, revision) 0
#endif
#endif

namespace at { namespace native {

// Mapping ScalarType to ideep tensor data_type
TORCH_API ideep::tensor::data_type get_mkldnn_dtype(ScalarType type);
static inline ideep::tensor::data_type get_mkldnn_dtype(const Tensor& t) {
  return get_mkldnn_dtype(t.scalar_type());
}

// Construct aten MKL-DNN tensor given an ideep tensor
TORCH_API Tensor new_with_itensor_mkldnn(ideep::tensor&& it, c10::optional<ScalarType> dtype, c10::optional<Device> device);

// Retrieve `ideep::tensor` from MKL-DNN tensor
TORCH_API ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

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

}}

#endif // AT_MKLDNN_ENABLED
