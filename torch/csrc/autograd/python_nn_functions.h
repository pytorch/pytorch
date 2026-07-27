#pragma once
#include <torch/csrc/utils/python_compat.h>
namespace torch::autograd {

void initNNFunctions(PyObject* module);

}
