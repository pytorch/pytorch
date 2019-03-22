#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

// Construct MKL-DNN tensor given `sizes` for allocation
Tensor new_with_sizes_mkldnn(IntArrayRef sizes, const TensorOptions& options);

// Construct MKL-DNN tensor from an initialized `ideep::tensor`
Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options);

// Retrieve `ideep::tensor` from MKL-DNN tensor
ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

}}

#endif // AT_MKLDNN_ENABLED
