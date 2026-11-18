#pragma once
#include <torch/csrc/utils/pythoncapi_compat.h>
namespace torch::autograd {

void initSpecialFunctions(PyObject* module);

}
