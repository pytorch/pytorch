#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::distributed::autograd {

PyMethodDef* python_functions();

} // namespace torch::distributed::autograd
