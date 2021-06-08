#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {
// Initialize Python bindings for JIT to_<backend> functions.
void initJitBackendBindings(PyObject* module);
Module codegen_func(
    const std::string& backend_name,
    const Module& orig_module,
    const py::dict& method_compile_spec);
} // namespace jit
} // namespace torch
