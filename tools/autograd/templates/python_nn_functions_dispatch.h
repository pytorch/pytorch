#pragma once

// ${generated_comment}

#include "torch/csrc/utils/auto_gil.h"

#include <ATen/ATen.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using namespace at;
using at::Generator;

${py_method_dispatch}

}} // namespace torch::autograd
