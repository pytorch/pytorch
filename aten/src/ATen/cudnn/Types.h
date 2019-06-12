#pragma once

#include "cudnn-wrapper.h"
#include <ATen/Tensor.h>

namespace at { namespace native {

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);

int64_t cudnn_version();

}}  // namespace at::cudnn
