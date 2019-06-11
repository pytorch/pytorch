#pragma once

#include <functional>
#include <iostream>

namespace torch {
namespace jit {

void initJITBindings(PyObject* module);

}
} // namespace torch
