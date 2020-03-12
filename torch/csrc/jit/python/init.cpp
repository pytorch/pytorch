#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/cuda_graph_fuser.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/fixup_onnx_conditionals.h>
#include <torch/csrc/jit/passes/onnx/fixup_onnx_loop.h>
#include <torch/csrc/jit/passes/onnx/peephole.h>
#include <torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/prepare_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/python/python_tree_views.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/tensorexpr/execution_counter.h>

#include <c10/macros/Export.h>
#include <caffe2/serialize/inline_container.h>

#include <ATen/core/function_schema.h>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch {
namespace jit {

using ::c10::Argument;
using ::c10::FunctionSchema;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;

namespace {

using autograd::variable_list;

bool loadPythonClasses() {
  // Leaving this code here, because it will likely be useful at some point
  // PyObject *jit_module = PyImport_ImportModule("torch.jit");
  // THPUtils_assert(jit_module, "class loader couldn't access "
  //"torch.jit module");
  // PyObject *jit_dict = PyModule_GetDict(jit_module);

  return true;
}
} // anonymous namespace

#if !defined(_WIN32) && !defined(__HIP_PLATFORM_HCC__)
TORCH_API void runJITCPPTests(bool runCuda);
#endif

void initJITBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::register_exception<JITException>(m, "JITException");

  py::class_<python::IODescriptor> iodescriptor(
      m, "IODescriptor"); // NOLINT(bugprone-unused-raii)

  m.def("_jit_init", loadPythonClasses)
      .def(
          "_jit_debug_fuser_num_cached_kernel_specs",
          torch::jit::fuser::debugNumCachedKernelSpecs)
      .def("_jit_pass_onnx_remove_print", RemovePrintOps)
      .def("_jit_pass_onnx_preprocess_caffe2", PreprocessCaffe2Ops)
      .def("_jit_pass_onnx", ToONNX)
      .def("_jit_pass_lower_all_tuples", LowerAllTuples)
      .def(
          "_jit_pass_onnx_peephole",
          [](std::shared_ptr<Graph>& graph,
             int opset_version,
             bool fixed_batch_size) {
            return PeepholeOptimizeONNX(graph, opset_version, fixed_batch_size);
          })
      .def(
          "_jit_pass_onnx_cast_all_constant_to_floating",
          CastAllConstantToFloating)
      .def(
          "_jit_pass_onnx_constant_fold",
          [](std::shared_ptr<Graph>& graph,
             std::map<std::string, at::Tensor>& paramsDict,
             int opset_version) {
            ConstantFoldONNX(
                graph->block(),
                paramsDict,
                opset_version); // overload resolution
            return paramsDict;
          },
          pybind11::return_value_policy::move)
      .def("_jit_pass_onnx_scalar_type_analysis", ScalarTypeAnalysisForONNX)
      .def(
          "_jit_pass_onnx_prepare_inplace_ops_for_onnx",
          PrepareInplaceOpsForONNX)
      .def("_jit_pass_fuse", FuseGraph)
      .def(
          "_jit_pass_dce",
          [](std::shared_ptr<Graph>& g) {
            return EliminateDeadCode(g->block()); // overload resolution
          })
      .def(
          "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
          [](std::shared_ptr<Graph>& g) {
            return EliminateDeadCode(
                g->block(),
                true,
                DCESideEffectPolicy::
                    ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS); // overload
                                                             // resolution
          })
      .def(
          "_jit_pass_cse",
          [](std::shared_ptr<Graph>& g) {
            return EliminateCommonSubexpression(g); // overload resolution
          })
      .def(
          "_jit_pass_insert_observers",
          [](Module& module,
             const std::string& method_name,
             const py::dict& qconfig_dict,
             bool inplace) {
            auto dict = py::cast<std::unordered_map<
                std::string,
                std::tuple<Module, Module>>>(qconfig_dict);
            return InsertObservers(module, method_name, dict, inplace);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("qconfig_dict"),
          py::arg("inplace") = false)
      .def(
          "_jit_pass_insert_quant_dequant",
          [](Module& module,
             const std::string& method_name,
             bool inplace) {
            return InsertQuantDeQuant(module, method_name, inplace);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("inplace") = false)
      .def(
          "_jit_pass_insert_prepack_unpack",
          [](std::shared_ptr<Graph>& g) { return InsertPrepackUnpack(g); })
      .def(
          "_jit_pass_insert_prepack_unpack",
          [](Module& module) { return InsertPrepackUnpack(module); })
      .def(
          "_jit_pass_quant_fusion",
          [](std::shared_ptr<Graph>& g) { return QuantFusion(g); })
      .def("_jit_pass_fold_convbn", &FoldConvBatchNorm2d)
      .def("_freeze_module",
          [](Module& module) {
            return freeze_module(module);
          },
          py::arg("module"))
      .def("_jit_pass_fuse_linear", &FuseLinear)
      .def(
          "_jit_pass_fold_quantize",
          [](Module& module, const std::string& method_name) {
            FoldQuantizeCallIntoBuffer(module, method_name);
          })
      .def("_jit_pass_fold_prepack", &FoldPrepackedWeightIntoModule)
      .def("_jit_pass_dedup_module_uses", &DedupModuleUses)
      .def("_jit_pass_replicate_dequantize", &ReplicateDeQuant)
      .def("_jit_pass_swap_dequantize", &SwapDeQuant)
      .def("_jit_pass_swap_functional_linear",
           [](std::shared_ptr<Graph>& graph) {
             SwapFunctionalLinear(graph);
           })
      .def("_jit_pass_swap_functional_linear",
           [](script::Module& module) {
             SwapFunctionalLinear(module);
           })
      .def(
          "_jit_pass_pattern_based_rewrite",
          [](const Module& m) { return PatternBasedRewrite(m); })
      .def(
          "_jit_pass_custom_pattern_based_rewrite",
          [](const std::string& pattern,
             const std::string& fused_node_name,
             const Module& m) {
            SubgraphRewriter subgraph_rewriter;
            subgraph_rewriter.RegisterRewritePattern(pattern, fused_node_name);
            subgraph_rewriter.runOnModule(m);
          })
      .def(
          "_jit_pass_custom_pattern_based_rewrite_graph",
          [](const std::string& pattern,
             const std::string& fused_node_name,
             std::shared_ptr<Graph> g) {
            SubgraphRewriter subgraph_rewriter;
            subgraph_rewriter.RegisterRewritePattern(pattern, fused_node_name);
            subgraph_rewriter.runOnGraph(g);
          })
      .def(
          "_jit_pass_fold_quant_inputs",
          [](std::shared_ptr<Graph>& g) {
            return FoldQuantNodesIntoInputsOutputs(g);
          })
      .def(
          "_jit_pass_remove_inplace_ops",
          [](std::shared_ptr<Graph> g) { return RemoveInplaceOps(g); })
      .def("_jit_pass_constant_pooling", ConstantPooling)
      .def(
          "_jit_pass_peephole",
          [](const std::shared_ptr<Graph>& g, bool addmm_fusion_enabled) {
            return PeepholeOptimize(g, addmm_fusion_enabled);
          },
          py::arg("graph"),
          py::arg("addmm_fusion_enabled") = false)
      .def(
          "_jit_pass_canonicalize",
          [](const std::shared_ptr<Graph>& g) { return Canonicalize(g); })
      .def("_jit_pass_lint", LintGraph)
      .def(
          "_jit_pass_complete_shape_analysis",
          [](std::shared_ptr<Graph> graph, py::tuple inputs, bool with_grad) {
            ArgumentSpecCreator arg_spec_creator(*graph);
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            ArgumentSpec spec = arg_spec_creator.create(with_grad, stack);
            arg_spec_creator.specializeTypes(*graph, spec);
            // We only get partial specialization from the arg_spec_creator, but
            // we want full shape specialization. The alternative would be to
            // have a "complete type inference" function in ArguemntSpecCreator.
            auto g_inputs = graph->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            PropagateInputShapes(graph);
          })
      .def("_jit_pass_remove_expands", RemoveExpands)
      .def("_jit_pass_erase_number_types", EraseNumberTypes)
      .def("_jit_pass_inline_fork_wait", InlineForkWait)
      .def("_jit_pass_inline", Inline)
      .def("_jit_pass_prepare_division_for_onnx", PrepareDivisionForONNX)
      .def(
          "_jit_pass_lower_graph",
          [](std::shared_ptr<Graph>& graph, const Module& self) {
            return LowerGraph(*graph, self._ivalue());
          })
      .def("_jit_pass_loop_unrolling", UnrollLoops)
      .def(
          "_jit_pass_constant_propagation",
          [](std::shared_ptr<Graph>& g) { return ConstantPropagation(g); })
      .def("_jit_pass_erase_shape_information", EraseShapeInformation)
      .def(
          "_jit_pass_create_autodiff_subgraphs",
          [](std::shared_ptr<Graph> graph) { CreateAutodiffSubgraphs(graph); })
#if defined(BUILDING_TESTS) && !defined(_WIN32) && !defined(__HIP_PLATFORM_HCC__)
      .def(
          "_jit_run_cpp_tests",
          [](bool runCuda) {
            // We have to release the GIL inside this method, because if we
            // happen to initialize the autograd engine in these tests, the
            // newly spawned worker threads will try to initialize their
            // PyThreadState*, and they need the GIL for this.
            pybind11::gil_scoped_release _no_gil;
            return runJITCPPTests(runCuda);
          },
          py::arg("run_cuda"))
      .def("_jit_has_cpp_tests", []() { return true; })
#else
      .def("_jit_run_cpp_tests", []() { throw std::exception(); })
      .def("_jit_has_cpp_tests", []() { return false; })
#endif
      .def(
          "_jit_flatten",
          [](py::handle& obj) {
            auto res = python::flatten(obj);
            return std::make_pair(res.vars, res.desc);
          })
      .def(
          "_jit_unflatten",
          [](autograd::variable_list vars, python::IODescriptor& desc) {
            return py::reinterpret_steal<py::object>(
                python::unflatten(vars, desc));
          })
      .def("_jit_pass_onnx_block", BlockToONNX)
      .def("_jit_pass_fixup_onnx_loops", FixupONNXLoops)
      .def("_jit_pass_fixup_onnx_conditionals", FixupONNXConditionals)
      .def("_jit_pass_canonicalize_ops", CanonicalizeOps)
      .def("_jit_pass_decompose_ops", DecomposeOps)
      .def("_jit_pass_specialize_autogradzero", specializeAutogradZero)
      .def("_jit_override_can_fuse_on_cpu", &overrideCanFuseOnCPU)
      .def("_jit_override_can_fuse_on_gpu", &overrideCanFuseOnGPU)
      .def("_jit_register_tensorexpr_fuser", &registerTensorExprFuser)
      .def(
          "_jit_differentiate",
          [](Graph& g) {
            // the python binding slightly differs in semantics
            // it makes a copy of the input Graph, and works on that
            // jit::differentiate mutates the input Graph
            auto g_clone = g.copy();
            return differentiate(g_clone);
          })
      .def(
          "_jit_check_alias_annotation",
          [](std::shared_ptr<Graph> g,
             py::tuple args,
             const std::string& unqualified_op_name) {
            auto stack = toTraceableStack(args);
            checkAliasAnnotation(g, std::move(stack), unqualified_op_name);
          })
      .def(
          "_jit_register_cuda_fuser", &registerCudaFuseGraph)
      .def(
          "_jit_set_profiling_mode",
          [](bool profiling_flag) {
            bool oldState = getProfilingMode();
            getProfilingMode() = profiling_flag;
            return oldState;
          })
      .def(
          "_jit_set_profiling_executor",
          [](bool profiling_flag) {
            bool oldState = getExecutorMode();
            getExecutorMode() = profiling_flag;
            return oldState;
          })
      .def(
          "_jit_set_num_profiled_runs",
          [](size_t num) {
            size_t old_num = getNumProfiledRuns();
            getNumProfiledRuns() = num;
            return old_num;
          })
      .def(
          "_jit_set_bailout_depth",
          [](size_t depth) {
            size_t old_depth = getBailoutDepth();
            getBailoutDepth() = depth;
            return old_depth;
          })
      .def(
          "_jit_set_inline_everything_mode",
          [](bool enabled) { getInlineEverythingMode() = enabled; })
      .def(
          "_jit_get_inline_everything_mode",
          []() { return getInlineEverythingMode(); })
      .def(
          "_jit_try_infer_type",
          [](py::object obj) -> TypePtr {
            auto match = tryToInferType(obj);
            if (match.success()) {
              return match.type();
            }
            return nullptr;
          })
      .def(
          "_jit_get_trigger_value",
          [](const std::string& trigger_name) {
            using namespace torch::jit::tensorexpr;
            ExecutionTrigger* trigger =
                ExecutionTriggerList::GetInstance().FindByName(trigger_name);
            return trigger->value();
          })
      .def("_jit_set_texpr_fuser_enabled", &setTensorExprFuserEnabled)
      .def(
          "_jit_fuser_get_fused_kernel_code",
          [](Graph& g, std::vector<at::Tensor> inps) {
            return debugGetFusedKernelCode(g, inps);
          })
      .def(
          "_jit_pass_onnx_unpack_quantized_weights",
          [](std::shared_ptr<Graph>& graph,
             std::map<std::string, at::Tensor>& paramsDict) {
            UnpackQuantizedWeights(graph, paramsDict);
            return paramsDict;
          },
          pybind11::return_value_policy::move)
      .def(
          "_jit_pass_onnx_quantization_insert_permutes",
          [](std::shared_ptr<Graph>& graph,
             std::map<std::string, at::Tensor>& paramsDict) {
            insertPermutes(graph, paramsDict);
            return paramsDict;
          },
          pybind11::return_value_policy::move);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<CompleteArgumentSpec>(m, "CompleteArgumentSpec")
      .def("__repr__", [](CompleteArgumentSpec& self) {
        std::ostringstream s;
        s << self;
        return s.str();
      });
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ArgumentSpec>(m, "ArgumentSpec");
  py::class_<Code>(m, "Code")
      .def(
          "grad_executor_states",
          [](Code& c) {
            std::vector<GraphExecutorState> states;
            for (auto& e : c.grad_executors()) {
              states.emplace_back(e->getDebugState());
            }
            return states;
          })
      .def("num_bailouts", [](Code& c) { return c.num_bailouts(); })
      .def("request_bailout", [](Code& c, size_t index) {
        c.request_bailout(index);
      });

  py::class_<ExecutionPlan>(m, "ExecutionPlan")
      .def_property_readonly("graph", [](ExecutionPlan& s) { return s.graph; })
      .def_property_readonly("code", [](ExecutionPlan& s) { return s.code; });

  py::class_<Gradient>(m, "Gradient")
      .def_property_readonly("f", [](Gradient& m) { return m.f; })
      .def_property_readonly("df", [](Gradient& m) { return m.df; })
      .def_property_readonly(
          "f_real_outputs", [](Gradient& m) { return m.f_real_outputs; })
      .def_property_readonly(
          "df_input_vjps", [](Gradient& m) { return m.df_input_vjps; })
      .def_property_readonly(
          "df_input_captured_inputs",
          [](Gradient& m) { return m.df_input_captured_inputs; })
      .def_property_readonly(
          "df_input_captured_outputs",
          [](Gradient& m) { return m.df_input_captured_outputs; })
      .def_property_readonly(
          "df_output_vjps", [](Gradient& m) { return m.df_output_vjps; });

  py::class_<GraphExecutorState>(m, "GraphExecutorState")
      .def_property_readonly(
          "graph", [](GraphExecutorState& s) { return s.graph; })
      .def_property_readonly(
          "execution_plans",
          [](GraphExecutorState& s) { return s.execution_plans; })
      .def_property_readonly(
          "fallback", [](GraphExecutorState& s) { return s.fallback; });

  py::class_<PyTorchStreamWriter>(m, "PyTorchFileWriter")
      .def(py::init<std::string>())
      .def(py::init([](const py::object &buffer) {
        auto writer_func = [=](const void *data, size_t size) {
          auto bytes = py::bytes(reinterpret_cast<const char *>(data), size);
          buffer.attr("write")(std::move(bytes));
          return size;
        };
        return std::make_unique<PyTorchStreamWriter>(std::move(writer_func));
      }))
      .def(py::init<const std::function<size_t(const void *, size_t)> &>())
      .def("write_record",
           [](PyTorchStreamWriter &self, const std::string &name,
              const char *data,
              size_t size) { return self.writeRecord(name, data, size); })
      .def("write_end_of_file", &PyTorchStreamWriter::writeEndOfFile)
      .def("write_record",
           [](PyTorchStreamWriter &self, const std::string &name,
              uintptr_t data, size_t size) {
             return self.writeRecord(name, reinterpret_cast<const char *>(data),
                                     size);
           });

  // This allows PyTorchStreamReader to read from a Python buffer. It requires
  // that the buffer implement `seek()`, `tell()`, and `read()`.
  class BufferAdapter : public caffe2::serialize::ReadAdapterInterface {
   public:
    BufferAdapter(const py::object& buffer) : buffer_(buffer) {
      // Jump to the end of the buffer to get its size
      auto current = buffer.attr("tell")();
      start_offset_ = py::cast<size_t>(current);
      buffer.attr("seek")(current, py::module::import("os").attr("SEEK_END"));
      size_ = py::cast<size_t>(buffer.attr("tell")()) - start_offset_;
      buffer.attr("seek")(current);

      // If we can read directly into a buffer, do that instead of an extra copy
      use_readinto_ = py::hasattr(buffer, "readinto");
    }

    size_t size() const override {
      return size_;
    }

    THPObjectPtr getMemview(void* buf, size_t n) const {
#if PY_MAJOR_VERSION >= 3
      THPObjectPtr memview(PyMemoryView_FromMemory(
          reinterpret_cast<char*>(buf), n, PyBUF_WRITE));
#else
      THPObjectPtr memview(PyBuffer_FromReadWriteMemory(buf, n));
#endif
      if (!memview) {
        throw python_error();
      }
      return memview;
    }

    size_t read(uint64_t pos, void* buf, size_t n, const char* what)
        const override {
      // Seek to desired position (NB: this has to be a Py_ssize_t or Python
      // throws a weird error)
      Py_ssize_t absolute_pos = start_offset_ + pos;
      buffer_.attr("seek")(absolute_pos);

      if (use_readinto_) {
        auto memview = getMemview(buf, n);
        auto res =
            PyObject_CallMethod(buffer_.ptr(), "readinto", "O", memview.get());
        if (res) {
          int i = PyInt_AsLong(res);
          if (i > 0) {
            return i;
          }
        }
      }

      // Read bytes into `buf` from the buffer
      std::string bytes = py::cast<std::string>(buffer_.attr("read")(n));
      std::copy(
          bytes.data(),
          bytes.data() + bytes.size(),
          reinterpret_cast<char*>(buf));
      return bytes.size();
    }

    py::object buffer_;
    size_t size_;
    size_t start_offset_;
    bool use_readinto_;
  };

  py::class_<PyTorchStreamReader>(m, "PyTorchFileReader")
      .def(py::init<std::string>())
      .def(py::init([](const py::object& buffer) {
        auto adapter = std::make_unique<BufferAdapter>(std::move(buffer));
        return std::make_unique<PyTorchStreamReader>(std::move(adapter));
      }))
      .def("get_record", [](PyTorchStreamReader& self, const std::string& key) {
        at::DataPtr data;
        size_t size;
        std::tie(data, size) = self.getRecord(key);
        return py::bytes(reinterpret_cast<const char*>(data.get()), size);
      })
      .def("get_all_records", [](PyTorchStreamReader& self) {
        return self.getAllRecords();
      });

  m.def(
      "_jit_get_operation",
      [](const std::string& op_name) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          TORCH_CHECK(!operations.empty(), "No such operator ", op_name);
          TORCH_CHECK(
              operations.size() == 1,
              "Found ",
              operations.size(),
              " overloads for operator ",
              op_name,
              "! Overloads are not supported from Python.");
          std::shared_ptr<Operator> op = operations[0];
          AT_ASSERT(op != nullptr);
          std::ostringstream docstring;
          docstring << "Automatically bound operator '" << op_name
                    << "' with schema: " << op->schema();
          return py::cpp_function(
              [op](py::args args, py::kwargs kwargs) {
                return invokeOperatorFromPython(
                    *op, std::move(args), std::move(kwargs));
              },
              py::name(symbol.toUnqualString()),
              py::doc(docstring.str().c_str()));
        } catch (const c10::Error& error) {
          throw std::runtime_error(error.what_without_backtrace());
        }
      },
      py::arg("qualified_name"));

  m.def("parse_ir", [](const std::string& input) {
    auto graph = std::make_shared<Graph>();
    parseIR(input, &*graph);
    return graph;
  });
  m.def("parse_schema", parseSchema);

  py::class_<FunctionSchema>(m, "FunctionSchema")
      .def_property_readonly(
          "name", [](FunctionSchema& self) { return self.name(); })
      .def_property_readonly(
          "overload_name",
          [](FunctionSchema& self) { return self.overload_name(); })
      .def_property_readonly(
          "arguments", [](FunctionSchema& self) { return self.arguments(); })
      .def_property_readonly(
          "returns", [](FunctionSchema& self) { return self.returns(); })
      .def("is_backward_compatible_with",
          [](const FunctionSchema& self, const FunctionSchema& old_schema) {
            return self.isBackwardCompatibleWith(old_schema);
          })
      .def("__eq__", [](const FunctionSchema& self,
            const FunctionSchema& other) {
          return self == other;
        })
      .def("__str__", [](FunctionSchema& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });
  py::class_<Argument>(m, "Argument")
      .def_property_readonly("name", [](Argument& self) { return self.name(); })
      .def_property_readonly("type", [](Argument& self) { return self.type(); })
      .def_property_readonly(
          "N",
          [](Argument& self) -> py::object {
            return (self.N()) ? py::cast(*self.N()) : py::none();
          })
      .def_property_readonly("default_value", [](Argument& self) -> py::object {
          if (!self.default_value())
            return py::none();
          IValue v = *self.default_value();
          return toPyObject(std::move(v));
        });
  m.def(
      "_jit_get_all_schemas", []() {
    const std::vector<std::shared_ptr<Operator>>& operations = getAllOperators();
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });
  m.def("_jit_get_schemas_for_operator", [](const std::string& qualified_name) {
    auto symbol = Symbol::fromQualString(qualified_name);
    auto operations = getAllOperatorsFor(symbol);
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });

  struct PythonFutureWrapper {
    explicit PythonFutureWrapper(c10::intrusive_ptr<c10::ivalue::Future> fut)
        : fut(std::move(fut)) {}

    c10::intrusive_ptr<c10::ivalue::Future> fut;
  };

  py::class_<PythonFutureWrapper>(m, "Future");

  m.def("fork", [](py::args args) {
    AT_ASSERT(args.size() >= 1);

    py::function f = py::cast<py::function>(args[0]);
    py::tuple args_tup(args.size() - 1);

    for (size_t i = 1; i < args.size(); ++i) {
      args_tup[i - 1] = args[i];
    }

    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;
      auto fork_node = graph->insertNode(graph->create(prim::TracedFork, 1));
      auto body_block = fork_node->addBlock();

      Value* node_output;
      py::object py_func_output;
      // Insert new trace ops into the fork op's sub-block
      WithInsertPoint guard(body_block);
      IValue output_ivalue;
      {
        tracer::WithNestedTracingFrame env_guard;

        // Run the user-supplied function
        py_func_output = f(*args_tup);

        // Convert the output of the user-supplied function to IValue. The type
        // information of this IValue is used both to record the correct type in
        // the trace.
        output_ivalue = toTypeInferredIValue(py_func_output);
        Value* out_val = jit::tracer::getValueTrace(output_ivalue);
        body_block->registerOutput(out_val);
        node_output =
            fork_node->output()->setType(FutureType::create(out_val->type()));
      }

      auto retval =
          c10::make_intrusive<c10::ivalue::Future>(output_ivalue.type());

      // Record the ivalue in the tracer
      jit::tracer::setValueTrace(retval, node_output);

      // stuff the ivalue output in the Future
      retval->markCompleted(output_ivalue);

      return PythonFutureWrapper(retval);
    } else {
      auto result = toTypeInferredIValue(f(*args_tup));
      auto retval = c10::make_intrusive<c10::ivalue::Future>(result.type());
      retval->markCompleted(std::move(result));
      return PythonFutureWrapper(retval);
    }
  });

  m.def("wait", [](PythonFutureWrapper& fut) {
    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;

      Value* fut_val = jit::tracer::getValueTrace(fut.fut);
      auto output = graph->insert(aten::wait, {fut_val});
      jit::tracer::setValueTrace(fut.fut->value(), output);
    }
    return fut.fut->value();
  });

  m.def("_jit_assert_is_instance", [](py::object obj, TypePtr type) {
    toIValue(obj, type);
  });

  initPythonCustomClassBindings(module);
  initPythonIRBindings(module);
  tracer::initPythonTracerBindings(module);
  initTreeViewBindings(module);
  initJitScriptBindings(module);

  setPrintHandler([](const std::string& str) {
    py::gil_scoped_acquire acquire;
    try {
      auto _stdout = py::module::import("sys").attr("stdout");
      _stdout.attr("write")(str);
    } catch (py::error_already_set& e) {
      throw std::runtime_error(e.what());
    }
  });
}
} // namespace jit
} // namespace torch
