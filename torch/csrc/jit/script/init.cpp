#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch {
namespace jit {
namespace script {

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<CompilationUnit>(m, "CompilationUnit")
      .def(
          "get_graph",
          &CompilationUnit::getGraph,
          py::return_value_policy::reference)
      .def(py::init<>())
      .def("define", &CompilationUnit::define);
  m.def("_jit_script_compile", jitScriptCompile);
}

} // namespace script
} // namespace jit
} // namespace torch
