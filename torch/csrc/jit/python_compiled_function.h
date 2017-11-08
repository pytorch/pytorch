#pragma once

#include "torch/csrc/jit/pybind.h"

#include <tuple>

namespace torch { namespace jit { namespace python {

void initCompilerMixin(PyObject *module);

}}}

