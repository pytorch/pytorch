#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void initPythonIRBindings(PyObject* module);

}}
