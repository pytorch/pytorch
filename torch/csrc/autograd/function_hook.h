#pragma once

#include <ATen/Tensor.h>

namespace torch { namespace autograd {

using variable_list = std::vector<at::Tensor>;

}}
