#include "torch/csrc/utils/pybind.h"

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/python_ir.h"
#include "torch/csrc/jit/python_arg_flatten.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/jit/python_compiled_function.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/onnx.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/canonicalize.h"
#include "torch/csrc/jit/passes/onnx/peephole.h"
#include "torch/csrc/jit/passes/onnx/fixup_onnx_loop.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/python_tree_views.h"


namespace torch  { namespace jit {

namespace {

using autograd::variable_list;

bool loadPythonClasses() {
  // Leaving this code here, because it will likely be useful at some point
  //PyObject *jit_module = PyImport_ImportModule("torch.jit");
  //THPUtils_assert(jit_module, "class loader couldn't access "
          //"torch.jit module");
  //PyObject *jit_dict = PyModule_GetDict(jit_module);

  return true;
}

// we cannot use the default py:cast<autograd::Variable> because it currently
// unwraps the data tensor in the conversion process
// TODO: replace with bs type
variable_tensor_list createVariableTensorList(py::tuple tuple, size_t reserve_extra_space = 0) {
  variable_tensor_list result;
  result.reserve(tuple.size() + reserve_extra_space);
  for(auto e : tuple) {
    result.push_back(py::cast<autograd::Variable>(e));
  }
  return result;
}

} // anonymous namespace

extern std::string runJITCPPTests();

void initJITBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<python::IODescriptor>(m, "IODescriptor");

  m.def("_jit_init", loadPythonClasses)
   .def("_jit_pass_onnx", ToONNX)
   .def("_jit_pass_onnx_peephole", PeepholeOptimizeONNX)
   .def("_jit_pass_fuse", FuseGraph)
   .def("_jit_pass_dce", [](std::shared_ptr<Graph>& g){
     return EliminateDeadCode(g); // overload resolution
   })
   .def("_jit_pass_cse", EliminateCommonSubexpression)
   .def("_jit_pass_peephole", PeepholeOptimize)
   .def("_jit_pass_canonicalize", [](const std::shared_ptr<Graph>& g) {
     return Canonicalize(g);
   })
   .def("_jit_pass_lint", LintGraph)
   .def("_jit_pass_shape_analysis", [](Graph& graph, py::tuple inputs, bool with_grad) {
     auto tensor_inputs = createVariableTensorList(inputs);
     PropagateInputShapes(graph, ArgumentSpec(with_grad, tensor_inputs));
   })
   .def("_jit_run_cpp_tests", runJITCPPTests)
   .def("_jit_flatten", [](py::handle& obj) {
     auto res =  python::flatten(obj);
     return std::make_pair(res.vars, res.desc);
   })
   .def("_jit_unflatten", [](autograd::variable_list vars, python::IODescriptor& desc) {
     return py::reinterpret_steal<py::object>(python::unflatten(vars, desc));
   })
   .def("_jit_pass_onnx_block", BlockToONNX)
   .def("_jit_pass_fixup_onnx_loops", FixupONNXLoops);

  py::class_<GraphExecutor>(m, "GraphExecutor")
      .def(
          py::init([](py::function func,
                      variable_list inputs,
                      bool optimize) {
              size_t num_inputs = inputs.size();
              auto graph = tracer::createGraphByTracing(func, std::move(inputs), num_inputs);
              return GraphExecutor(graph, optimize);
          }),
          py::arg("func"),
          py::arg("inputs"),
          py::arg("optimize") = true)
      .def(
          py::init([](std::shared_ptr<Graph> graph, bool optimize) {
            return GraphExecutor(std::move(graph), optimize);
          }),
          py::arg("graph"),
          py::arg("optimize") = true)
      .def_property_readonly("graph", [](GraphExecutor& ge) {
        return ge.graph();
      })
      .def("__call__", [](GraphExecutor& ge, py::args args) -> py::object {
        auto inputs = createVariableTensorList(args);
        auto outputs = ge.run(std::move(inputs));
        // if we don't tell pybind these are variables it chokes on the
        // conversion.
        // TODO: fix conversions to be sane and make sure this works.
        if (outputs.size() == 0) {
          return py::none();
        } else if (outputs.size() == 1) {
          return py::cast(static_cast<autograd::Variable&>(outputs[0]));
        } else {
          py::tuple tuple(outputs.size());
          for(size_t i = 0; i < outputs.size(); i++) {
            tuple[i] = py::cast(static_cast<autograd::Variable&>(outputs[i]));
          }
          return tuple;
        }
      });
  initPythonIRBindings(module);
  initPythonTracerBindings(module);
  python::initCompilerMixin(module);
  script::initTreeViewBindings(module);
  script::initJitScriptBindings(module);
}

}}
