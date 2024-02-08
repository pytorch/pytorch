#include "torch/csrc/autograd/FunctionsManual.h"
#include "torch/csrc/dynamo/compiled_autograd.h"

// ${generated_comment}

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace torch::autograd::generated::details;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd::generated {

${autograd_function_definitions}

} // namespace torch::autograd::generated
