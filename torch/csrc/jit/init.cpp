#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/utils/auto_gil.h"

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/python_ir.h"
#include "torch/csrc/jit/python_arg_flatten.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/jit/import.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/passes/remove_expands.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/onnx.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/erase_number_types.h"
#include "torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_pooling.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/canonicalize.h"
#include "torch/csrc/jit/passes/onnx/peephole.h"
#include "torch/csrc/jit/passes/onnx/fixup_onnx_loop.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/canonicalize_ops.h"
#include "torch/csrc/jit/passes/remove_inplace_ops.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/python_tree_views.h"
#include "torch/csrc/jit/batched/BatchTensor.h"
#include "torch/csrc/jit/pybind_utils.h"
#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/script/jit_exception.h"

#include "caffe2/serialize/inline_container.h"

#include <pybind11/functional.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch  { namespace jit {

// TODO: make a fake future for python
namespace detail {
class Future {

};
}

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

} // anonymous namespace

#if defined(_WIN32)
std::string runJITCPPTests() {
  AT_ERROR("JIT tests not yet supported on Windows");
}
#else
std::string runJITCPPTests();
#endif

void initJITBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();

  py::register_exception<JITException>(m, "JITException");

  py::class_<python::IODescriptor>(m, "IODescriptor");

  m.def("_jit_init", loadPythonClasses)
   .def("_jit_pass_onnx", ToONNX)
   .def("_jit_pass_lower_all_tuples", LowerAllTuples)
   .def("_jit_pass_onnx_peephole", PeepholeOptimizeONNX)
   .def("_jit_pass_fuse", FuseGraph)
   .def("_jit_pass_dce", [](std::shared_ptr<Graph>& g) {
     return EliminateDeadCode(g); // overload resolution
   })
   .def("_jit_pass_cse", [](std::shared_ptr<Graph>& g) {
     return EliminateCommonSubexpression(g); // overload resolution
   })
   .def("_jit_pass_remove_inplace_ops", [](std::shared_ptr<Graph> g) {
      return RemoveInplaceOps(g);
   })
   .def("_jit_pass_constant_pooling", ConstantPooling)
   .def("_jit_pass_peephole", [](const std::shared_ptr<Graph>& g, bool addmm_fusion_enabled) {
     return PeepholeOptimize(g, addmm_fusion_enabled);
   }, py::arg("graph"), py::arg("addmm_fusion_enabled") = false)
   .def("_jit_pass_canonicalize", [](const std::shared_ptr<Graph>& g) {
     return Canonicalize(g);
   })
   .def("_jit_pass_lint", LintGraph)
   .def("_jit_pass_shape_analysis", [](Graph& graph, std::vector<at::Tensor> inputs, bool with_grad) {
     setInputTypes(graph, ArgumentSpec(with_grad, fmap<IValue>(inputs), inputs.size()));
     PropagateInputShapes(graph);
   })
   .def("_jit_pass_complete_shape_analysis", [](Graph& graph, py::tuple inputs, bool with_grad) {
     CompleteArgumentSpec spec(with_grad, evilDeprecatedBadCreateStackDoNotUse(inputs, graph.inputs()));
     auto graph_inputs = graph.inputs();
     JIT_ASSERT(spec.size() == graph_inputs.size());
     for (size_t i = 0; i < graph_inputs.size(); ++i) {
       graph_inputs[i]->setType(spec.at(i));
     }
     PropagateInputShapes(graph);
   })
   .def("_jit_pass_remove_expands", RemoveExpands)
   .def("_jit_pass_erase_number_types", EraseNumberTypes)
   .def("_jit_pass_prepare_division_for_onnx", PrepareDivisionForONNX)
   .def("_jit_pass_loop_unrolling", UnrollLoops)
   .def("_jit_pass_constant_propagation", [](std::shared_ptr<Graph>& g) {
     return ConstantPropagation(g);
   })
   .def("_jit_pass_erase_shape_information", EraseShapeInformation)
   .def("_jit_pass_create_autodiff_subgraphs", [](Graph& graph) {
     CreateAutodiffSubgraphs(graph);
   })
   .def("_jit_run_cpp_tests", [] {
     // We have to release the GIL inside this method, because if we happen to
     // initialize the autograd engine in these tests, the newly spawned worker threads will
     // try to initialize their PyThreadState*, and they need the GIL for this.
     AutoNoGIL _no_gil;
     return runJITCPPTests();
   })
   .def("_jit_flatten", [](py::handle& obj) {
     auto res =  python::flatten(obj);
     return std::make_pair(res.vars, res.desc);
   })
   .def("_jit_unflatten", [](autograd::variable_list vars, python::IODescriptor& desc) {
     return py::reinterpret_steal<py::object>(python::unflatten(vars, desc));
   })
   .def("_jit_pass_onnx_block", BlockToONNX)
   .def("_jit_pass_fixup_onnx_loops", FixupONNXLoops)
   .def("_jit_pass_canonicalize_ops", CanonicalizeOps)
   .def("_jit_pass_specialize_undef", specializeUndef)
   .def("_jit_override_can_fuse_on_cpu", &overrideCanFuseOnCPU)
   .def("_jit_differentiate", [](Graph &g) {
       // the python binding slightly differs in semantics
       // it makes a copy of the input Graph, and works on that
       // jit::differentiate mutates the input Graph
       auto g_clone = g.copy();
       return differentiate(g_clone);
   });

  py::class_<CompleteArgumentSpec>(m, "CompleteArgumentSpec")
      .def("__repr__", [](CompleteArgumentSpec& self) {
        std::ostringstream s;
        s << self;
        return s.str();
      });
  py::class_<ArgumentSpec>(m, "ArgumentSpec");
  py::class_<Code>(m, "Code")
      .def("grad_executors", [](Code& c) {
        return py::make_iterator(c.grad_executors().begin(), c.grad_executors().end());
      });

  py::class_<ExecutionPlanState>(m, "ExecutionPlanState")
    .def_property_readonly("graph", [](ExecutionPlanState& s) {
      return s.graph;
    })
    .def_property_readonly("code", [](ExecutionPlanState& s) {
      return s.code;
    });

  py::class_<Gradient>(m, "Gradient")
    .def_property_readonly("f", [](Gradient& m) {
      return m.f;
    })
    .def_property_readonly("df", [](Gradient& m) {
      return m.df;
    })
    .def_property_readonly("f_real_outputs", [](Gradient& m) {
      return m.f_real_outputs;
    })
    .def_property_readonly("df_input_vjps", [](Gradient& m) {
      return m.df_input_vjps;
    })
    .def_property_readonly("df_input_captured_inputs", [](Gradient& m) {
      return m.df_input_captured_inputs;
    })
    .def_property_readonly("df_input_captured_outputs", [](Gradient& m) {
      return m.df_input_captured_outputs;
    })
    .def_property_readonly("df_output_vjps", [](Gradient& m) {
      return m.df_output_vjps;
    });

  py::class_<GraphExecutorState>(m, "GraphExecutorState")
    .def_property_readonly("graph", [](GraphExecutorState& s) {
      return s.graph;
    })
    .def_property_readonly("execution_plans", [](GraphExecutorState& s) {
      return s.execution_plans;
    })
    .def_property_readonly("fallback", [](GraphExecutorState& s) {
      return s.fallback;
    });

  py::class_<GraphExecutor>(m, "GraphExecutor", py::dynamic_attr())
      .def(
          py::init([](py::function func,
                      py::tuple inputs,
                      py::function var_name_lookup_fn,
                      bool optimize) {
              auto graph = tracer::createGraphByTracing(func, toStack(inputs), var_name_lookup_fn);
              return GraphExecutor(graph, optimize);
          }),
          py::arg("func"),
          py::arg("inputs"),
          py::arg("var_name_lookup_fn"),
          py::arg("optimize") = true)
      .def(
          py::init([](std::shared_ptr<Graph> graph, bool optimize) {
            return GraphExecutor(std::move(graph), optimize);
          }),
          py::arg("graph"),
          py::arg("optimize") = true)
      .def("graph_for", [](GraphExecutor& ge, py::args args) {
        return ge.graphFor(evilDeprecatedBadCreateStackDoNotUse(args, ge.graph()->inputs()));
      })
      .def_property_readonly("graph", [](GraphExecutor& ge) {
        return ge.graph();
      })
      .def("get_debug_state", [](GraphExecutor& ge) {
        return ge.getDebugState();
      })
      .def("__call__", [](GraphExecutor& ge, py::args args) -> py::object {
        const auto & graph = ge.graph();
        auto stack = evilDeprecatedBadCreateStackDoNotUse(args, graph->inputs());
        {
          AutoNoGIL no_gil_guard;
          ge.run(stack);
        }
        return createPyObjectForStack(std::move(stack));
      });

  py::class_<PyTorchFileWriter>(m, "PyTorchFileWriter")
      .def(py::init<std::string>())
      .def(
          "write_record",
          [](PyTorchFileWriter& self, const char* data, size_t size) {
            return self.writeRecord(data, size);
          })
      .def("write_end_of_file", &PyTorchFileWriter::writeEndOfFile);

  py::class_<PyTorchFileReader>(m, "PyTorchFileReader")
      .def(py::init<std::string>())
      .def(
          "get_record_with_key",
          [](PyTorchFileReader& self, uint64_t key) {
            at::DataPtr data;
            size_t size;
            std::tie(data, size) = self.getRecordWithKey(key);
            return py::bytes(reinterpret_cast<const char*>(data.get()), size);
          })
      .def("get_last_record", [](PyTorchFileReader& self) {
        at::DataPtr data;
        size_t size;
        std::tie(data, size) = self.getLastRecord();
        return py::bytes(reinterpret_cast<const char*>(data.get()), size);
      });

  m.def("_jit_get_operation", [](const std::string& qualified_name) {
    try {
      auto symbol = Symbol::fromQualString(qualified_name);
      auto operations = getAllOperatorsFor(std::move(symbol));
      AT_CHECK(!operations.empty(), "No such operator ", qualified_name);
      AT_CHECK(
          operations.size() == 1,
          "Found ", operations.size(), " overloads for operator ",
          qualified_name, "! Overloads are not supported from Python.");
      std::shared_ptr<Operator> op = operations[0];
      AT_ASSERT(op != nullptr);
      std::ostringstream docstring;
      docstring << "Automatically bound operator '" << qualified_name
                << "' with schema: " << op->schema();
      return py::cpp_function([op](py::args args, py::kwargs kwargs) {
        return invokeOperatorFromPython(
            *op, std::move(args), std::move(kwargs));
      }, py::name(qualified_name.c_str()), py::doc(docstring.str().c_str()));
    } catch (const c10::Error& error) {
      throw std::runtime_error(error.what_without_backtrace());
    }
  }, py::arg("qualified_name"));

  py::class_<FunctionSchema>(m, "FunctionSchema")
  .def_property_readonly("name", [](FunctionSchema& self) { return self.name(); })
  .def_property_readonly("arguments", [](FunctionSchema& self) { return self.arguments(); })
  .def_property_readonly("returns", [](FunctionSchema& self) { return self.returns(); });
  py::class_<Argument>(m, "Argument")
  .def_property_readonly("name", [](Argument& self) { return self.name(); })
  .def_property_readonly("type", [](Argument& self) { return self.type(); })
  .def_property_readonly("N", [](Argument& self) -> py::object {
    return (self.N()) ? py::cast(*self.N()) :  py::none();
  })
  .def_property_readonly("default_value", [](Argument& self) -> py::object {
    if(!self.default_value())
      return py::none();
    IValue v = *self.default_value();
    return toPyObject(std::move(v));
  });
  m.def("_jit_get_schemas_for_operator", [](const std::string& qualified_name) {
    auto symbol = Symbol::fromQualString(qualified_name);
    auto operations = getAllOperatorsFor(std::move(symbol));
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
        return op->schema();
      });
  });

  py::class_<detail::Future>(m, "Future");

  m.def("fork", [](script::Module &sm, py::args args) {
    // TODO: this is a fake stub
    return detail::Future();
  });

  m.def("wait", [](detail::Future &fut) {
    // TODO: this is a fake stub
  });


  initPythonIRBindings(module);
  tracer::initPythonTracerBindings(module);
  script::initTreeViewBindings(module);
  script::initJitScriptBindings(module);
  initBatchTensorBindings(module);
  initRegisterBatchOpsBindings(module);
}

}}
