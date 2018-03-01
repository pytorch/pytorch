#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(std::string)>;

struct PythonResolver : public Resolver {
  PythonResolver(ResolutionCallback rcb) : rcb(rcb) {}

  Node* resolveCall(SourceRange location, Node* n) const override {
    AutoGIL ag;
    py::function func;
    func = rcb(n->kind().toString());
    auto* py_func = func.ptr();
    if (py_func == Py_None) {
      throw ErrorReport(location)
          << "Unknown function " << n->kind().toString();
    }
    // Release the function object so we can wrap it in a PythonOp
    auto fn_ptr = THPObjectPtr(func.release().ptr());
    std::string cconv(n->inputs().size(), 't');
    Node* new_node = n->owningGraph()->createPythonOp(
        std::move(fn_ptr), cconv, false, {}, {}, false);
    return new_node;
  }

  ResolutionCallback rcb;
};

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<CompilationUnit>(m, "CompilationUnit")
      .def(
          "get_graph",
          &CompilationUnit::getGraph,
          py::return_value_policy::reference)
      .def(py::init<>())
      .def(
          "define",
          [](CompilationUnit* self,
             const std::string& script,
             ResolutionCallback rcb) {
            PythonResolver r(rcb);
            return self->define(script, r);
          });
  m.def("_jit_script_compile", [](Def def, ResolutionCallback rcb) {
    PythonResolver r(rcb);
    return jitScriptCompile(def, r);
  });
}

} // namespace script
} // namespace jit
} // namespace torch
