#pragma once
#include <torch/csrc/utils/python_stub.h>

// see [Note: Compiled Autograd]
namespace torch::dynamo::autograd {
PyObject* torch_c_dynamo_compiled_autograd_init();
} // namespace torch::dynamo::autograd
