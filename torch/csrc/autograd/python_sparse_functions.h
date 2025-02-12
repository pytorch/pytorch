#pragma once
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace torch::autograd {

void initSparseFunctions(PyObject* module);

}
