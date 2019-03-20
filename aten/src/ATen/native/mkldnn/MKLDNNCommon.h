#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

Tensor new_mkldnn_with_itensor(ideep::tensor&& ideep_tensor, const TensorOptions& options);

ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

}}

#endif // AT_MKLDNN_ENABLED
