#pragma once

#include "torch/csrc/utils/python_stub.h"

namespace torch { namespace jit { namespace python {

void initCompilerMixin(PyObject *module);

}}}
