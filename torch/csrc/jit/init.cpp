#include "torch/csrc/utils/pybind.h"

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/python_ir.h"
#include "torch/csrc/jit/python_arg_flatten.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/jit/python_compiled_function.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/onnx.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/canonicalize.h"
#include "torch/csrc/jit/passes/onnx/peephole.h"



namespace torch  { namespace jit {

namespace {

bool loadPythonClasses() {
  // Leaving this code here, because it will likely be useful at some point
  //PyObject *jit_module = PyImport_ImportModule("torch.jit");
  //THPUtils_assert(jit_module, "class loader couldn't access "
          //"torch.jit module");
  //PyObject *jit_dict = PyModule_GetDict(jit_module);

  return true;
}

template<void (*F)(std::shared_ptr<Graph>& graph)>
void graph_pass(const std::shared_ptr<tracer::TracingState>& state) {
  return F(state->graph);
}

} // anonymous namespace

extern void runJITCPPTests();

void initJITBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_jit_init", loadPythonClasses)
   .def("_jit_pass_onnx", ToONNX)
   .def("_jit_pass_onnx_peephole", graph_pass<PeepholeOptimizeONNX>)
   .def("_jit_pass_fuse", graph_pass<FuseGraph>)
   .def("_jit_pass_dce", graph_pass<EliminateDeadCode>)
   .def("_jit_pass_cse", graph_pass<EliminateCommonSubexpression>)
   .def("_jit_pass_peephole", graph_pass<PeepholeOptimize>)
   .def("_jit_pass_canonicalize", graph_pass<Canonicalize>)
   .def("_jit_pass_lint", graph_pass<LintGraph>)
   .def("_jit_run_cpp_tests", runJITCPPTests)
   .def("_jit_flatten", [](py::handle& obj) {
     return python::flatten(obj).vars;
   });

  initPythonIRBindings(module);
  initPythonTracerBindings(module);
  python::initCompilerMixin(module);
}

}}
