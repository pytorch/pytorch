#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(Graph*, std::string)>;

struct PythonResolver : public Resolver {
  PythonResolver(ResolutionCallback rcb) : rcb(rcb) {}

  std::vector<Value*> resolveCall(SourceRange location, Node* n) override {
    AutoGIL ag;
    py::function func;
    try {
      func = rcb(n->owningGraph(), n->kind().toString());
    } catch (std::exception e) {
      throw ErrorReport(location)
          << "Unknown function " << n->kind().toString();
    }
    auto* py_func = func.ptr();
    if (py_func == Py_None) {
      throw ErrorReport(location)
          << "Unknown function " << n->kind().toString();
    }
    // Release the function object so we can wrap it in a PythonOp
    pybind11::handle h = func.release();
    std::string cconv;
    for (const auto& i : n->inputs()) {
      cconv += "t";
    }
    Node* new_node = n->owningGraph()->createPythonOp(
        THPObjectPtr(h.ptr()), cconv, false, {}, {}, false);
    new_node->insertBefore(n);
    for (const auto i : n->inputs()) {
      new_node->addInput(i);
    }
    for (const auto o : n->outputs()) {
      new_node->addOutput();
    }
    n->replaceAllUsesWith(new_node);
    n->destroy();
    return new_node->outputs();
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
            return self->define(script, &r);
          });
  m.def("_jit_script_compile", [](Def def, ResolutionCallback rcb) {
    PythonResolver r(rcb);
    return jitScriptCompile(def, &r);
  });
}

} // namespace script
} // namespace jit
} // namespace torch
