#pragma once

#include <Python.h>

namespace torch {
namespace fx {
void initFx(PyObject* module);
} // namespace fx
} // namespace torch
