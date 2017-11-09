#pragma once

#include "cudnn-wrapper.h"
#include <ATen/Tensor.h>

#include <cstddef>
#include <string>

namespace torch { namespace cudnn {

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);
void _cudnn_assertContiguous(const at::Tensor& tensor, const std::string& name);
#define cudnn_assertContiguous(tensor) \
_cudnn_assertContiguous(tensor, #tensor " tensor")

}}  // namespace torch::cudnn
