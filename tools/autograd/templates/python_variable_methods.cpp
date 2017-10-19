// ${generated_comment}

#include <Python.h>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"

#include "python_variable_methods_dispatch.h"

using at::Tensor;
using at::Scalar;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

${py_methods}

PyMethodDef variable_methods[] = {
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
