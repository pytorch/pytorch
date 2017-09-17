// ${generated_comment}

#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/Exceptions.h"

using at::Tensor;
using at::Scalar;

namespace torch { namespace autograd {

${py_methods}

PyMethodDef variable_methods[] = {
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
