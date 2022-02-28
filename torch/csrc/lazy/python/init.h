#pragma once
#include <c10/macros/Export.h>
#include <pybind11/pybind11.h>

namespace torch {
namespace lazy {

TORCH_API void initLazyBindings(PyObject* module);

}  // namespace lazy
}  // namespace torch
