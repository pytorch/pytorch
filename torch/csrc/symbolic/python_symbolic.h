#pragma once
#include <torch/csrc/python_headers.h>
#include <torch/csrc/symbolic/NativeShapeEnv.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::symbolic {

void initSymbolicBindings(PyObject* module);

py::object hint_to_py(const Hint& h);
// The Python node of `n`: the SymNode of a PythonSymNodeImpl, else the binding.
py::object node_to_py(const c10::SymNode& n);

// python_symint_glue.cpp
void initGlueBindings(py::module_& sm);
// cls(node) for cls SymInt or SymBool, without running cls.__init__ while the
// sealed glue is intact.
py::object make_sym_object(py::handle cls, py::handle node);
// Whether op(*args) on SymInt/SymBool operands runs the sealed native glue
// without logging MAGIC.
bool sym_glue_native();

} // namespace torch::symbolic
