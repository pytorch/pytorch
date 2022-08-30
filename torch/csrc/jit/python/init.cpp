#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/schema_info.h>

#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
#include <torch/csrc/jit/codegen/onednn/interface.h>
#endif
#include <c10/core/SymIntNodeImpl.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/create_functional_graphs.h>
#include <torch/csrc/jit/passes/cuda_graph_fuser.h>
#include <torch/csrc/jit/passes/dbr_quantization/remove_redundant_aliases.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/device_type_analysis.h>
#include <torch/csrc/jit/passes/dtype_analysis.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/integer_value_refinement.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/metal_rewrite.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/quantization/dedup_module_uses.h>
#include <torch/csrc/jit/passes/quantization/finalize.h>
#include <torch/csrc/jit/passes/quantization/fusion_passes.h>
#include <torch/csrc/jit/passes/quantization/insert_observers.h>
#include <torch/csrc/jit/passes/quantization/insert_quant_dequant.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>
#include <torch/csrc/jit/passes/refine_tuple_types.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/python/python_tree_views.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/csrc/jit/runtime/static/init.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <c10/util/signal_handler.h>
#include <caffe2/serialize/inline_container.h>

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>

#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch {
namespace jit {

using c10::AliasInfo;
using c10::Argument;
using c10::FunctionSchema;
using c10::SchemaArgType;
using c10::SchemaArgument;
using c10::SymIntNode;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using torch::utils::SchemaInfo;

static c10::SymIntNode toSymIntNode(c10::SymIntNode a, py::object b) {
  return torch::is_symint_node(b) ? b.cast<c10::SymIntNode>()
                                  : a->wrap(b.cast<int64_t>());
}

class PythonSymIntNodeImpl : public c10::SymIntNodeImpl {
 public:
  PythonSymIntNodeImpl(py::object pyobj) : c10::SymIntNodeImpl() {
    pyobj_ = std::make_shared<c10::SafePyObject>(
        pyobj.release().ptr(), getPyInterpreter());
  };

  virtual SymIntNode wrap(int64_t num) override {
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr("wrap")(num);
    return c10::make_intrusive<PythonSymIntNodeImpl>(r);
  }

  virtual bool bool_() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("__bool__")().is(py::handle(Py_True));
  }

  virtual int64_t int_() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("__int__")().cast<int64_t>();
  }

  virtual std::string str() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("__str__")().cast<std::string>();
  }

  virtual SymIntNode dispatch_common_(
      const char* fname,
      const SymIntNode& other) {
    auto pother = dynamic_cast<PythonSymIntNodeImpl*>(other.get());
    TORCH_CHECK(pother);
    py::gil_scoped_acquire acquire;
    auto r = getPyObj().attr(fname)(pother->getPyObj());
    return c10::make_intrusive<PythonSymIntNodeImpl>(r);
  }

  virtual SymIntNode add(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode sub(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode mul(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode truediv(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode floordiv(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode mod(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode eq(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode gt(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode lt(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode le(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  virtual SymIntNode ge(const SymIntNode& other) override {
    return dispatch_common_(__FUNCTION__, other);
  }

  py::handle getPyObj() {
    return py::handle(pyobj_.get()->ptr(getPyInterpreter()));
  }
  std::shared_ptr<c10::SafePyObject> pyobj_ = nullptr;
};

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

c10::optional<IValue> toTypeInferredIValueOptional(py::handle input) {
  // Errors need to be caught here because toTypeInferredIValue errors out
  // on various object types, but we want it to work with all types.
  try {
    return toTypeInferredIValue(input);
  } catch (const c10::Error& e) {
    return c10::nullopt;
  }
}
} // anonymous namespace

#if !defined(USE_ROCM)
TORCH_API void runJITCPPTests();
#endif

void initJITBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_jit");

  static py::exception<JITException> exc(m, "JITException");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const JITException& e) {
      // special handling of JITException, to set its python class name and msg
      py::gil_scoped_acquire acquire;
      const auto& className = e.getPythonClassName();
      const auto& originalMsg = e.getOriginalMsg();
      JITException::setCaughtOriginalMsg(originalMsg.value_or(""));
      JITException::setCaughtPythonClassName(className.value_or(""));
      exc(e.what());
    }
  });

  m.def(
      "_get_caught_jit_exception_class_name",
      JITException::getCaughtPythonClassName);
  m.def(
      "_get_caught_jit_exception_original_msg",
      JITException::getCaughtOriginalMsg);

  py::class_<python::IODescriptor> iodescriptor(
      m,
      "IODescriptor"); // NOLINT(bugprone-unused-raii)

  m.def("_jit_init", loadPythonClasses)
      .def(
          "_jit_debug_fuser_num_cached_kernel_specs",
          torch::jit::fuser::debugNumCachedKernelSpecs)
      .def("_jit_pass_lower_all_tuples", LowerAllTuples)
      .def(
          "_new_symbolic_shape_symbol",
          []() { return c10::ShapeSymbol::newSymbol().value(); })
      .def(
          "_jit_shape_compute_graph_for_node",
          [](Node* n) -> c10::optional<std::shared_ptr<Graph>> {
            if (!n->maybeSchema()) {
              return c10::nullopt;
            }
            return shapeComputeGraphForSchema(n->schema());
          })
      .def(
          "_jit_decomposition_graph_for_node",
          [](Node* n) -> c10::optional<std::shared_ptr<Graph>> {
            if (!n->maybeSchema()) {
              return c10::nullopt;
            }
            return GetDecomposition(n->schema());
          })
      .def("_jit_pass_run_decompositions", RunDecompositions)
      // using Node* here instead of Schema because looking up the schema
      // and passing it in from Python will have a different pointer than the
      // schema that is globally used for caching
      .def(
          "_jit_register_shape_compute_graph_for_node",
          [](Node* n, std::shared_ptr<Graph>& graph) {
            if (n->maybeSchema()) {
              const FunctionSchema& schema = n->schema();
              RegisterShapeComputeGraphForSchema(schema, graph);
            } else {
              TORCH_INTERNAL_ASSERT(false, "Expected schema", n);
            }
          })
      .def(
          "_jit_register_decomposition_for_schema",
          [](const FunctionSchema& s, std::shared_ptr<Graph>& graph) {
            // because this is invoked by python, the function schema *
            // becomes different, and we need to find and reuse the
            // one that is used for caching
            auto op =
                findOperatorFor(c10::OperatorName(s.name(), s.overload_name()));
            RegisterDecomposition(op->schema(), graph);
          })
      .def("_jit_pass_propagate_shapes_on_graph", PropagateShapesOnGraph)
      .def(
          "_jit_pass_propagate_shapes_on_graph_and_build_compute",
          [](std::shared_ptr<Graph>& graph) {
            return PropagateShapesAndBuildLargeShapeComputeGraph(
                graph, *graph->nodes().begin(), *graph->nodes().end());
          })
      .def(
          "_jit_pass_propagate_shapes_on_graph_and_build_compute",
          [](std::shared_ptr<Graph>& graph, Node* beg) {
            return PropagateShapesAndBuildLargeShapeComputeGraph(
                graph, beg, *graph->nodes().end());
          })
      .def(
          "_jit_pass_propagate_shapes_on_graph_and_build_compute",
          PropagateShapesAndBuildLargeShapeComputeGraph)
      .def("_jit_pass_integer_value_refinement", RefineIntegerValues)
      .def(
          "_jit_set_symbolic_shapes_test_mode",
          &setSymbolicShapeAnalysisTestMode)
      .def(
          "_jit_symbolic_shapes_test_mode_enabled",
          &symbolicShapeAnalysisTestModeEnabled)
      .def("_jit_pass_autocast", Autocast)
      .def("_jit_set_autocast_mode", &setAutocastMode)
      .def("_jit_pass_fuse", FuseGraph)
      .def(
          "_jit_pass_replace_old_ops_with_upgraders",
          [](std::shared_ptr<Graph>& g) {
            return ReplaceOldOperatorsWithUpgraders(g);
          })
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
          "_jit_pass_fuse_quantized_add_relu",
          [](std::shared_ptr<Graph>& g) {
            return FuseQuantizedAddRelu(g); // overload resolution
          })
      .def(
          "_jit_pass_insert_observers",
          [](Module& module,
             const std::string& method_name,
             const py::dict& qconfig_dict,
             bool inplace,
             int quant_type_int) {
            auto dict = py::cast<std::unordered_map<
                std::string,
                c10::optional<std::tuple<Module, Module>>>>(qconfig_dict);
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertObservers(
                module, method_name, dict, inplace, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("qconfig_dict"),
          py::arg("inplace"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_observer_method_for_ondevice_ptq",
          [](Module& module,
             const std::string& method_name,
             const py::dict& qconfig_dict,
             bool inplace,
             int quant_type_int) {
            auto dict = py::cast<std::unordered_map<
                std::string,
                c10::optional<std::tuple<Module, Module>>>>(qconfig_dict);
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertObserversForOnDevicePTQ(
                module, method_name, dict, inplace, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("qconfig_dict"),
          py::arg("inplace"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_quant_dequant",
          [](Module& module,
             const std::string& method_name,
             bool inplace,
             bool debug,
             int quant_type_int) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertQuantDeQuant(
                module, method_name, inplace, debug, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("inplace"),
          py::arg("debug"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_quant_dequant_for_ondevice_ptq",
          [](Module& module,
             const std::string& method_name,
             bool inplace,
             bool debug,
             int quant_type_int) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return InsertQuantDeQuantOnDevicePTQ(
                module, method_name, inplace, debug, quant_type);
          },
          py::arg("module"),
          py::arg("method_name"),
          py::arg("inplace"),
          py::arg("debug"),
          py::arg("quant_type_int") = 1)
      .def(
          "_jit_pass_insert_prepack_unpack",
          [](std::shared_ptr<Graph>& g) { return InsertPrepackUnpack(g); })
      .def(
          "_jit_pass_insert_prepack_unpack",
          [](Module& module) { return InsertPrepackUnpack(module); })
      .def(
          "_jit_pass_quant_fusion",
          [](std::shared_ptr<Graph>& g) { return QuantFusion(g); })
      .def(
          "_jit_pass_fold_convbn",
          [](Module& module) { return FoldConvBatchNorm(module); })
      .def(
          "_jit_pass_dbr_quant_remove_redundant_aliases",
          [](Module& module) { return DBRQuantRemoveRedundantAliases(module); })
      .def(
          "_freeze_module",
          [](Module& module,
             std::vector<std::string>& preservedAttrs,
             bool freezeInterfaces,
             bool preserveParameters) {
            return freeze_module(
                module, preservedAttrs, freezeInterfaces, preserveParameters);
          },
          py::arg("module"),
          py::arg("preservedAttrs") = std::vector<std::string>(),
          py::arg("freezeInterfaces") = true,
          py::arg("preserveParameters") = false)
      .def("_jit_pass_concat_frozen_linear", &FrozenConcatLinear)
      .def("_jit_pass_fold_frozen_conv_bn", &FoldFrozenConvBatchnorm)
      .def("_jit_pass_fold_frozen_conv_add_or_sub", &FoldFrozenConvAddOrSub)
      .def("_jit_pass_fold_frozen_conv_mul_or_div", &FoldFrozenConvMulOrDiv)
      .def("_jit_pass_convert_frozen_ops_to_mkldnn", &ConvertFrozenOpsToMKLDNN)
      .def("_jit_pass_fuse_frozen_conv_add_relu", &FuseFrozenConvAddRelu)
      .def("_jit_pass_transpose_frozen_linear", &FrozenLinearTranspose)
      .def("_jit_pass_optimize_frozen_graph", &OptimizeFrozenGraph)
      .def(
          "_jit_pass_optimize_for_inference",
          [](Module& module, std::vector<std::string> other_methods) {
            optimize_for_inference(module, other_methods);
          },
          py::arg("module"),
          py::arg("other_methods") = std::vector<std::string>())
      .def("_jit_pass_fuse_linear", &FuseLinear)
      .def(
          "_jit_pass_fuse_add_relu",
          [](std::shared_ptr<Graph>& graph) { FuseAddRelu(graph); })
      .def("_jit_pass_dedup_module_uses", &DedupModuleUses)
      .def("_jit_pass_replicate_dequantize", &ReplicateDeQuant)
      .def(
          "_jit_pass_swap_functional_linear",
          [](std::shared_ptr<Graph>& graph) { SwapFunctionalLinear(graph); })
      .def(
          "_jit_pass_swap_functional_linear",
          [](Module& module) { SwapFunctionalLinear(module); })
      .def(
          "_jit_pass_quant_finalize",
          [](Module& module,
             int quant_type_int,
             const std::vector<std::string>& preserved_attrs) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return Finalize(module, quant_type, preserved_attrs);
          },
          py::arg("module"),
          py::arg("quant_type_int") = 1,
          py::arg("preserved_attrs") = std::vector<std::string>())
      .def(
          "_jit_pass_quant_finalize_for_ondevice_ptq",
          [](Module& module,
             int quant_type_int,
             const std::string& method_name) {
            auto quant_type = static_cast<QuantType>(quant_type_int);
            return FinalizeOnDevicePTQ(module, quant_type, method_name);
          },
          py::arg("module"),
          py::arg("quant_type_int") = 1,
          py::arg("preserved_attrs") = std::vector<std::string>())
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
             std::shared_ptr<Graph> g,
             const std::vector<std::pair<std::string, std::string>>&
                 value_name_pairs) {
            SubgraphRewriter subgraph_rewriter;
            subgraph_rewriter.RegisterRewritePattern(
                pattern, fused_node_name, value_name_pairs);
            subgraph_rewriter.runOnGraph(g);
          },
          py::arg("pattern"),
          py::arg("fused_node_name"),
          py::arg("g"),
          py::arg("value_name_pairs") =
              std::vector<std::pair<std::string, std::string>>())
      .def("_jit_pass_constant_pooling", ConstantPooling)
      // RemoveInplaceOps is used by CoreML so it must be removed with care.
      .def("_jit_pass_propagate_dtype", DtypePropagation)
      .def("_jit_pass_propagate_device", DeviceTypePropagation)
      .def(
          "_jit_pass_remove_inplace_ops",
          [](const std::shared_ptr<Graph>& g) { return RemoveInplaceOps(g); })
      .def(
          "_jit_pass_create_functional_graphs",
          [](std::shared_ptr<Graph>& g) { return CreateFunctionalGraphs(g); })
      .def(
          "_jit_pass_remove_mutation",
          [](std::shared_ptr<Graph>& g) {
            RemoveListMutation(g);
            return RemoveTensorMutation(g);
          })
      .def(
          "_jit_pass_functional_to_inplace_activation",
          [](std::shared_ptr<Graph>& g) {
            return FunctionalToInplaceActivation(g);
          })
      .def(
          "_jit_pass_inplace_to_functional_activation",
          [](std::shared_ptr<Graph>& g) {
            return InplaceToFunctionalActivation(g);
          })
      .def(
          "_jit_pass_inline_functional_graphs",
          [](std::shared_ptr<Graph>& g) { return InlineFunctionalGraphs(g); })
      .def(
          "_jit_pass_peephole",
          [](const std::shared_ptr<Graph>& g, bool disable_shape_peepholes) {
            return PeepholeOptimize(g, disable_shape_peepholes);
          },
          py::arg("graph"),
          py::arg("disable_shape_peepholes") = false)
      .def(
          "_jit_pass_peephole_list_idioms",
          [](const std::shared_ptr<Graph>& g, bool refine_list_len) {
            return PeepholeOptimizeListIdioms(g, refine_list_len);
          },
          py::arg("graph"),
          py::arg("refine_list_len") = false)
      .def(
          "_jit_pass_refine_integer_values",
          [](std::shared_ptr<Graph>& g) { return RefineIntegerValues(g); })
      .def(
          "_jit_pass_fuse_addmm",
          [](std::shared_ptr<Graph>& g) { return FuseAddMM(g); })
      .def(
          "_jit_pass_canonicalize",
          [](const std::shared_ptr<Graph>& g, bool keep_unique_names = true) {
            return Canonicalize(g, keep_unique_names);
          },
          py::arg("graph"),
          py::arg("keep_unique_names") = true)
      .def("_jit_pass_lint", LintGraph)
      .def(
          "_jit_pass_complete_shape_analysis",
          [](const std::shared_ptr<Graph>& graph,
             const py::tuple& inputs,
             bool with_grad) {
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
            for (const auto i : c10::irange(inputs.size())) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            PropagateInputShapes(graph);
          })
      .def(
          "_jit_interpret_graph",
          [](std::shared_ptr<Graph>& graph, const py::tuple& inputs) {
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto g_inputs = graph->inputs();
            for (const auto i : c10::irange(inputs.size())) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            Code code(graph, "<on-demand-func>");
            InterpreterState(code).run(stack);
            return createPyObjectForStack(std::move(stack));
          },
          py::doc(
              "Interpret a JIT graph with given inputs without running any optimization passes on it"))
      .def(
          "_jit_trace_graph",
          [](std::shared_ptr<Graph>& graph, const py::tuple& inputs) {
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto g_inputs = graph->inputs();
            for (const auto i : c10::irange(inputs.size())) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            return TraceGraph(graph, stack);
          })
      .def(
          "_jit_trace_module",
          [](Module& model, const py::tuple& inputs) {
            auto graph = model.get_method("forward").graph();
            Stack stack;
            stack.reserve(inputs.size() + 1); // captures?
            push(stack, model._ivalue());
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto traced = TraceGraph(graph, stack);
            GRAPH_DUMP("Traced Graph", traced);

            // the easiest way to replace a graph in a module is
            // to remove all the nodes in the original graph
            // clone everything from the traced one
            graph->block()->clear();
            graph->block()->cloneFrom(traced->block(), nullptr);
            GRAPH_DUMP("Copied Graph", graph);
          })
      .def("_jit_pass_remove_expands", RemoveExpands)
      .def("_jit_pass_erase_number_types", EraseNumberTypes)
      .def("_jit_pass_inline_fork_wait", InlineForkWait)
      .def("_jit_pass_inline", Inline)
      .def(
          "_jit_pass_lower_graph",
          [](std::shared_ptr<Graph>& graph, const Module& self) {
            return LowerGraph(*graph, self._ivalue());
          })
      .def("_jit_pass_loop_unrolling", UnrollLoops)
      .def("_jit_pass_constant_loop_unrolling", UnrollConstantLoops)
      .def(
          "_jit_pass_constant_propagation_immutable_types",
          [](std::shared_ptr<Graph>& g) {
            return ConstantPropagationImmutableTypes(g);
          })
      .def(
          "_jit_pass_constant_propagation",
          [](std::shared_ptr<Graph>& g) { return ConstantPropagation(g); },
          py::arg("graph"))
      .def("_jit_pass_erase_shape_information", EraseShapeInformation)
      .def(
          "_jit_object_is_non_holding",
          [](Node& n) {
            return toIValue(n.output())->toObject()->is_weak_compilation_ref();
          })
      .def(
          "_jit_erase_non_input_shape_information",
          [](std::shared_ptr<Graph>& g) {
            std::vector<TypePtr> input_types;
            for (Value* v : g->inputs()) {
              if (auto tt = v->type()->cast<TensorType>()) {
                input_types.push_back(tt);
              } else {
                input_types.push_back(nullptr);
              }
            }
            EraseShapeInformation(g);
            for (size_t i = 0; i < input_types.size(); ++i) {
              if (input_types[i]) {
                g->inputs().at(i)->setType(input_types[i]);
              }
            }
          })
      .def(
          "_jit_pass_create_autodiff_subgraphs",
          [](const std::shared_ptr<Graph>& graph, py::object threshold) {
            if (threshold.is(py::none())) {
              CreateAutodiffSubgraphs(graph);
            } else {
              CreateAutodiffSubgraphs(graph, py::cast<int>(threshold));
            }
          },
          py::arg("graph"),
          py::arg("threshold") = py::none())
#if defined(BUILDING_TESTS) && !defined(USE_ROCM)
      .def(
          "_jit_run_cpp_tests",
          []() {
            // We have to release the GIL inside this method, because if we
            // happen to initialize the autograd engine in these tests, the
            // newly spawned worker threads will try to initialize their
            // PyThreadState*, and they need the GIL for this.
            pybind11::gil_scoped_release _no_gil;
            return runJITCPPTests();
          })
      .def("_jit_has_cpp_tests", []() { return true; })
      .def("_has_tensorexpr_cpp_tests", []() { return true; })
#else
      .def("_jit_run_cpp_tests", []() { throw std::exception(); })
      .def("_jit_has_cpp_tests", []() { return false; })
      .def("_run_tensorexpr_cpp_tests", []() { throw std::exception(); })
      .def("_has_tensorexpr_cpp_tests", []() { return false; })
#endif
      .def(
          "_jit_flatten",
          [](py::handle& obj) {
            auto res = python::flatten(obj);
            return std::make_pair(res.vars, res.desc);
          })
      .def(
          "_jit_unflatten",
          [](const autograd::variable_list& vars, python::IODescriptor& desc) {
            return py::reinterpret_steal<py::object>(
                python::unflatten(vars, desc));
          })
      .def("_jit_pass_canonicalize_graph_fuser_ops", CanonicalizeOps)
      .def("_jit_pass_decompose_ops", DecomposeOps)
      .def("_jit_pass_specialize_autogradzero", specializeAutogradZero)
      .def("_jit_override_can_fuse_on_cpu", &overrideCanFuseOnCPU)
      .def("_jit_override_can_fuse_on_gpu", &overrideCanFuseOnGPU)
      .def("_jit_can_fuse_on_cpu", &canFuseOnCPU)
      .def("_jit_can_fuse_on_gpu", &canFuseOnGPU)
      .def("_jit_can_fuse_on_cpu_legacy", &canFuseOnCPULegacy)
      .def("_jit_override_can_fuse_on_cpu_legacy", &overrideCanFuseOnCPULegacy)
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
          [](const std::shared_ptr<Graph>& g,
             const py::tuple& args,
             const std::string& unqualified_op_name) {
            auto stack = toTraceableStack(args);
            checkAliasAnnotation(g, std::move(stack), unqualified_op_name);
          })
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
      .def("_jit_set_llga_enabled", &RegisterLlgaFuseGraph::setEnabled)
      .def("_jit_llga_enabled", &RegisterLlgaFuseGraph::isEnabled)
#endif
      .def(
          "_jit_set_tracer_state_warn",
          [](bool new_warn) {
            jit::tracer::getTracerStateWarnMode() = new_warn;
          })
      .def(
          "_jit_get_tracer_state_warn",
          []() {
            bool current_tracer_warn = jit::tracer::getTracerStateWarnMode();
            return current_tracer_warn;
          })
      .def(
          "_jit_set_nvfuser_skip_node_kind",
          // Args:
          //     `op_name`: Symbol of op;
          //     `flip`: flag indicating whether to flip the given op in the
          //             skip list.
          // Returns:
          //     a bool flag indicating if `op_name` was already in the skip
          //     list.
          [](const std::string& op_name, bool flip = true) {
            return fuser::cuda::skipNode(op_name, flip);
          })
      .def("_jit_set_nvfuser_enabled", &fuser::cuda::setEnabled)
      .def("_jit_nvfuser_can_be_enabled", &fuser::cuda::canBeEnabled)
      .def(
          "_jit_set_nvfuser_single_node_mode",
          [](bool flag) { return fuser::cuda::setSingletonFusion(flag); })
      .def(
          "_jit_nvfuser_single_node_mode",
          []() { return fuser::cuda::getSingletonFusion(); })
      .def(
          "_jit_set_nvfuser_horizontal_mode",
          [](bool flag) { return fuser::cuda::setHorizontalFusion(flag); })
      .def(
          "_jit_nvfuser_horizontal_mode",
          []() { return fuser::cuda::getHorizontalFusion(); })
      .def(
          "_jit_set_nvfuser_guard_mode",
          [](bool profiling_flag) {
            bool oldState = fuser::cuda::getCudaFusionGuardMode();
            fuser::cuda::getCudaFusionGuardMode() = profiling_flag;
            return oldState;
          })
      .def("_jit_nvfuser_enabled", &fuser::cuda::isEnabled)
      .def(
          "_jit_nvfuser_set_comparison_callback",
          [](bool run_fallback, py::function fn) {
            // If set, then the callback will be run after each nvfuser fusion
            // group is executed. Can be used for testing accuracy.
            // If run_fallback == True, then a fallback will be run and
            // unfused_outputs will be nonempty, showing the result if the
            // fusion didn't take place. Otherwise, unfused_outputs will
            // be empty
            auto fn_ptr = std::make_shared<py::function>(fn);
            auto callback_lambda = [fn_ptr](
                                       const Stack& fused_outputs,
                                       const Stack& unfused_outputs,
                                       const std::string& graph_ir) {
              py::gil_scoped_acquire acquire{};
              (*fn_ptr)(fused_outputs, unfused_outputs, graph_ir);
            };
            setCudaFuserComparisonCallback({run_fallback, callback_lambda});
          })
      .def(
          "_jit_nvfuser_clear_comparison_callback",
          []() {
            setCudaFuserComparisonCallback({false, nullptr});
          })
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
          "_jit_get_num_profiled_runs",
          [] {
            // pybind can't automatically bind to atomic size_t
            size_t num_runs = getNumProfiledRuns();
            return num_runs;
          })
      .def(
          "_jit_set_bailout_depth",
          [](size_t depth) {
            TORCH_WARN(
                "Use _jit_set_fusion_strategy, bailout depth is deprecated. Setting to (STATIC, ",
                depth,
                ")");
            size_t old_depth = getBailoutDepth();
            FusionStrategy strat = {{FusionBehavior::STATIC, depth}};
            setFusionStrategy(strat);
            return old_depth;
          })
      .def(
          "_jit_set_fusion_strategy",
          [](std::vector<std::pair<std::string, size_t>> strategy) {
            FusionStrategy vec_conv;
            for (const auto& pair : strategy) {
              if (pair.first == "STATIC") {
                vec_conv.emplace_back(FusionBehavior::STATIC, pair.second);
              } else if (pair.first == "DYNAMIC") {
                vec_conv.emplace_back(FusionBehavior::DYNAMIC, pair.second);
              } else {
                TORCH_INTERNAL_ASSERT(
                    false,
                    "FusionBehavior only supported 'STATIC' or 'DYNAMIC', got: ",
                    pair.first);
              }
            }
            auto old_strategy = getFusionStrategy();
            auto strat =
                fmap(old_strategy, [](std::pair<FusionBehavior, size_t> behav) {
                  return std::pair<std::string, size_t>(
                      behav.first == FusionBehavior::STATIC ? "STATIC"
                                                            : "DYNAMIC",
                      behav.second);
                });
            setFusionStrategy(vec_conv);
            return strat;
          })
      .def(
          "_jit_set_inline_everything_mode",
          [](bool enabled) { getInlineEverythingMode() = enabled; })
      .def(
          "_jit_get_inline_everything_mode",
          []() { return getInlineEverythingMode(); })
      .def(
          "_jit_get_logging_option",
          []() { return ::torch::jit::get_jit_logging_levels(); })
      .def(
          "_jit_set_logging_option",
          [](std::string loggingOption) -> void {
            ::torch::jit::set_jit_logging_levels(loggingOption);
          })
      .def(
          "_jit_set_logging_stream",
          [](std::string stream_name) -> void {
            if (stream_name == "stdout") {
              ::torch::jit::set_jit_logging_output_stream(std::cout);
            } else if (stream_name == "stderr") {
              ::torch::jit::set_jit_logging_output_stream(std::cerr);
            } else {
              std::cerr << "ERROR: only `stdout` and `stderr`"
                        << "are supported as output options" << std::endl;
            }
          })
      .def(
          "_storage_id",
          [](const at::Tensor& ten) -> int64_t {
            return reinterpret_cast<int64_t>(
                ten.storage().unsafeGetStorageImpl());
          })
      .def(
          "_jit_try_infer_type",
          [](py::object obj) -> InferredType {
            return tryToInferType(std::move(obj));
          })
      .def(
          "_jit_get_te_cuda_pointwise_loop_levels",
          []() -> int {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseLoopLevels();
          })
      .def(
          "_jit_set_te_cuda_pointwise_loop_levels",
          [](int level) {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseLoopLevels() = level;
          })
      .def(
          "_jit_get_te_cuda_pointwise_block_count",
          []() -> int {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockCount();
          })
      .def(
          "_jit_set_te_cuda_pointwise_block_count",
          [](int block_count) {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockCount() = block_count;
          })
      .def(
          "_jit_get_te_cuda_pointwise_block_size",
          []() -> int {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockSize();
          })
      .def(
          "_jit_set_te_cuda_pointwise_block_size",
          [](int block_size) {
            using namespace torch::jit::tensorexpr;
            return getTECudaPointwiseBlockSize() = block_size;
          })
      .def("_jit_set_texpr_fuser_enabled", &setTensorExprFuserEnabled)
      .def("_jit_texpr_fuser_enabled", &tensorExprFuserEnabled)
      .def("_jit_texpr_fallback_allowed", &tensorexpr::fallbackAllowed)
      .def("_jit_texpr_set_fallback_allowed", &tensorexpr::setFallbackAllowed)
      .def("_jit_set_texpr_reductions_enabled", &setTexprReductionsEnabled)
      .def(
          "_jit_set_texpr_dynamic_shape_enabled",
          &setTensorExprDynamicShapeFusionEnabled)
      .def(
          "_jit_texpr_dynamic_shape_enabled",
          &tensorExprDynamicShapeFusionEnabled)
      .def("_jit_texpr_reductions_enabled", &texprReductionsEnabled)
      .def(
          "_jit_set_te_generate_block_code",
          [](bool gen_block_code) {
            using namespace torch::jit::tensorexpr;
            return getTEGenerateBlockCode() = gen_block_code;
          })
      .def(
          "_jit_get_te_generate_block_code",
          []() -> bool {
            using namespace torch::jit::tensorexpr;
            return getTEGenerateBlockCode();
          })
      .def(
          "_jit_get_te_must_use_llvm_cpu",
          []() -> bool {
            using namespace torch::jit::tensorexpr;
            return getTEMustUseLLVMOnCPU();
          })
      .def(
          "_jit_set_te_must_use_llvm_cpu",
          [](bool use_llvm) {
            using namespace torch::jit::tensorexpr;
            getTEMustUseLLVMOnCPU() = use_llvm;
          })
      .def(
          "_jit_cat_wo_conditionals",
          [](bool optimize_cat) {
            using namespace torch::jit::tensorexpr;
            getCatWoConditionals() = optimize_cat;
          })
      .def(
          "_jit_opt_conditionals",
          [](bool opt_conds) {
            using namespace torch::jit::tensorexpr;
            getOptConditionals() = opt_conds;
          })
      .def(
          "_llvm_enabled",
          []() {
#ifdef TORCH_ENABLE_LLVM
            return true;
#else
        return false;
#endif
          })
      .def(
          "_jit_pass_fuse_tensorexprs",
          [](std::shared_ptr<Graph>& g) {
            FuseTensorExprs(g);
            RemoveTensorTypeSpecializations(g);
          })
      .def(
          "_jit_fuser_get_fused_kernel_code",
          [](Graph& g, const std::vector<at::Tensor>& inps) {
            return debugGetFusedKernelCode(g, inps);
          })
      .def(
          "_jit_pass_remove_dropout",
          [](script::Module& module) { return removeDropout(module); })
      .def(
          "_jit_pass_refine_tuple_types",
          [](std::shared_ptr<Graph>& graph) { return RefineTupleTypes(graph); })
      .def(
          "_jit_pass_transform_conv1d_to_conv2d",
          [](std::shared_ptr<Graph>& graph) {
            return transformConv1dToConv2d(graph);
          })
      .def(
          "_jit_pass_transform_conv1d_to_conv2d",
          [](script::Module& module) {
            return transformConv1dToConv2d(module);
          })
      .def(
          "_jit_pass_insert_prepacked_ops",
          [](std::shared_ptr<Graph>& graph) {
            return insertPrePackedOps(graph);
          })
      .def(
          "_jit_pass_insert_prepacked_ops",
          [](script::Module& module) { return insertPrePackedOps(module); })
      .def(
          "_jit_pass_fuse_clamp_w_prepacked_linear_conv",
          [](script::Module& module) {
            return fusePrePackedLinearConvWithClamp(module);
          })
      .def(
          "_jit_pass_fold_prepacking_ops",
          [](script::Module& module) { return FoldPrePackingOps(module); })
      .def(
          "_jit_pass_optimize_for_mobile",
          [](script::Module& module,
             std::set<MobileOptimizerType>& optimization_blocklist,
             std::vector<std::string>& preserved_methods) {
            return optimizeForMobile(
                module, optimization_blocklist, preserved_methods);
          })
      .def(
          "_hack_do_not_use_clone_module_with_class",
          [](script::Module& module,
             std::vector<std::string>& ignored_methods,
             std::vector<std::string>& ignored_attributes) {
            const bool inplace = false;
            const std::unordered_set<std::string> ignored_methods_set(
                ignored_methods.begin(), ignored_methods.end());
            const std::unordered_set<std::string> ignored_attributes_set(
                ignored_attributes.begin(), ignored_attributes.end());
            return module.clone(
                inplace, ignored_methods_set, ignored_attributes_set);
          })
      .def(
          "_jit_pass_vulkan_insert_prepacked_ops",
          [](std::shared_ptr<Graph>& graph) {
            return vulkanInsertPrePackedOps(graph);
          })
      .def(
          "_jit_pass_vulkan_insert_prepacked_ops",
          [](script::Module& module) {
            return vulkanInsertPrePackedOps(module);
          })
      .def(
          "_jit_pass_vulkan_fuse_clamp_w_prepacked_conv",
          [](script::Module& module) {
            return vulkanFusePrePackedConvWithClamp(module);
          })
      .def(
          "_jit_pass_vulkan_fold_prepacking_ops",
          [](script::Module& module) {
            return vulkanFoldPrePackingOps(module);
          })
      .def(
          "_jit_pass_vulkan_optimize_for_mobile",
          [](script::Module& module,
             std::vector<std::string>& preserved_methods) {
            return vulkanOptimizeForMobile(module, preserved_methods);
          })
      .def(
          "_jit_pass_metal_insert_prepacked_ops",
          [](std::shared_ptr<Graph>& graph) {
            return metalInsertPrePackedOps(graph);
          })
      .def(
          "_jit_pass_metal_insert_prepacked_ops",
          [](script::Module& module) {
            return metalInsertPrePackedOps(module);
          })
      .def(
          "_jit_pass_metal_fuse_clamp_w_prepacked_conv",
          [](script::Module& module) {
            return metalFusePrePackedConvWithClamp(module);
          })
      .def(
          "_jit_pass_metal_fold_prepacking_ops",
          [](script::Module& module) { return metalFoldPrePackingOps(module); })
      .def(
          "_jit_pass_metal_optimize_for_mobile",
          [](script::Module& module,
             std::vector<std::string>& preserved_methods) {
            return metalOptimizeForMobile(module, preserved_methods);
          })
      .def(
          "_jit_pass_filter_non_tensor_arguments",
          [](std::map<std::string, IValue> params) {
            std::map<std::string, at::Tensor> retval;
            for (auto& kv : params) {
              if (kv.second.isTensor()) {
                retval[kv.first] = std::move(kv.second).toTensor();
              }
            }
            return retval;
          })
      .def("_jit_pass_batch_mm", BatchMM)
      .def("_jit_decay_packed_param_input_types", [](Graph& g) {
        for (Value* i : g.inputs()) {
          if (i->type() ==
                  getCustomClass(
                      "__torch__.torch.classes.quantized.Conv2dPackedParamsBase") ||
              i->type() ==
                  getCustomClass(
                      "__torch__.torch.classes.quantized.Conv3dPackedParamsBase") ||
              i->type() ==
                  getCustomClass(
                      "__torch__.torch.classes.quantized.LinearPackedParamsBase")) {
            // Dummy CompleteTensorType to appease ONNX validator.
            i->setType(TensorType::create(
                at::kQInt8,
                c10::kCPU,
                std::vector<int64_t>{1},
                std::vector<int64_t>{1},
                c10::nullopt));
          }
        }
      });

  py::class_<c10::SymIntNodeImpl, c10::SymIntNode>(m, "SymIntNode")
      .def_static(
          "new_symint",
          [](py::object obj) -> c10::SymIntNode {
            return c10::make_intrusive<PythonSymIntNodeImpl>(obj);
          })
      .def(
          "get_pyobj",
          [](c10::SymIntNode a) -> py::object {
            if (auto* psn = dynamic_cast<PythonSymIntNodeImpl*>(a.get())) {
              return py::reinterpret_borrow<py::object>(psn->getPyObj());
            }
            return py::none();
          })
      .def(
          "__add__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->add(snb);
          })
      .def(
          "__radd__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->add(snb);
          })
      .def(
          "__sub__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->sub(snb);
          })
      .def(
          "__mul__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->mul(snb);
          })
      .def(
          "__rmul__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->mul(snb);
          })
      .def(
          "__truediv__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->truediv(snb);
          })
      .def(
          "__rtruediv__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return snb->truediv(a);
          })
      .def(
          "__floordiv__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->floordiv(snb);
          })
      .def(
          "__rfloordiv__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return snb->floordiv(a);
          })
      .def(
          "__mod__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->mod(snb);
          })
      .def(
          "__eq__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->eq(snb);
          })
      .def(
          "__gt__",
          [](c10::SymIntNode a, py::object b) {
            auto snb = toSymIntNode(a, b);
            return a->gt(snb);
          })
      .def(
          "__lt__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->lt(snb);
          })
      .def(
          "__le__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->le(snb);
          })
      .def(
          "__ge__",
          [](c10::SymIntNode a, py::object b) -> c10::SymIntNode {
            auto snb = toSymIntNode(a, b);
            return a->ge(snb);
          })
      .def("__bool__", [](c10::SymIntNode a) { return a->bool_(); })
      .def("__int__", [](c10::SymIntNode a) { return a->int_(); })
      .def("__str__", [](c10::SymIntNode a) { return a->str(); });

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
      .def(
          "differentiable_op_executor_states",
          [](Code& c) {
            std::vector<GraphExecutorState> states;
            for (auto& e : c.diff_graph_op_executors()) {
              if (e->isOptimized()) {
                states.emplace_back(e->getDebugState());
              } else {
                // we leave an empty entry for node that doesn't have an
                // optimized plan
                states.emplace_back();
              }
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
      .def(py::init([](const py::object& buffer) {
        auto writer_func = [=](const void* data, size_t size) {
          // Writting an empty file is a noop
          if (size == 0) {
            return size;
          }
          auto memory_view = py::memoryview::from_memory(
              reinterpret_cast<const char*>(data), size);
          buffer.attr("write")(std::move(memory_view));
          return size;
        };
        return std::make_unique<PyTorchStreamWriter>(std::move(writer_func));
      }))
      .def(py::init<const std::function<size_t(const void*, size_t)>&>())
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             const char* data,
             size_t size) { return self.writeRecord(name, data, size); })
      .def("write_end_of_file", &PyTorchStreamWriter::writeEndOfFile)
      .def("set_min_version", &PyTorchStreamWriter::setMinVersion)
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             uintptr_t data,
             size_t size) {
            return self.writeRecord(
                name, reinterpret_cast<const char*>(data), size);
          })
      .def("archive_name", &PyTorchStreamWriter::archiveName)
      .def(
          "get_all_written_records",
          &PyTorchStreamWriter::getAllWrittenRecords);

  py::enum_<MobileOptimizerType>(m, "MobileOptimizerType")
      .value("CONV_BN_FUSION", MobileOptimizerType::CONV_BN_FUSION)
      .value(
          "INSERT_FOLD_PREPACK_OPS",
          MobileOptimizerType::INSERT_FOLD_PREPACK_OPS)
      .value("REMOVE_DROPOUT", MobileOptimizerType::REMOVE_DROPOUT)
      .value("FUSE_ADD_RELU", MobileOptimizerType::FUSE_ADD_RELU)
      .value(
          "HOIST_CONV_PACKED_PARAMS",
          MobileOptimizerType::HOIST_CONV_PACKED_PARAMS)
      .export_values();

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
      THPObjectPtr memview(PyMemoryView_FromMemory(
          reinterpret_cast<char*>(buf), n, PyBUF_WRITE));
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
          int64_t i = static_cast<int64_t>(PyLong_AsLongLong(res));
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

  py::class_<PyTorchStreamReader, std::shared_ptr<PyTorchStreamReader>>(
      m, "PyTorchFileReader")
      .def(py::init<std::string>())
      .def(py::init([](const py::object& buffer) {
        auto adapter = std::make_unique<BufferAdapter>(buffer);
        return std::make_shared<PyTorchStreamReader>(std::move(adapter));
      }))
      .def(
          "get_record",
          [](PyTorchStreamReader& self, const std::string& key) {
            at::DataPtr data;
            size_t size = 0;
            std::tie(data, size) = self.getRecord(key);
            return py::bytes(reinterpret_cast<const char*>(data.get()), size);
          })
      .def(
          "has_record",
          [](PyTorchStreamReader& self, const std::string& key) {
            return self.hasRecord(key);
          })
      .def(
          "get_storage_from_record",
          [](PyTorchStreamReader& self,
             const std::string& key,
             size_t numel,
             py::object data_type_obj) {
            at::DataPtr data(std::get<0>(self.getRecord(key)));
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;

            c10::Storage storage(
                c10::Storage::use_byte_size_t(),
                numel * elementSize(scalar_type),
                std::move(data),
                /*allocator=*/nullptr,
                /*resizable=*/false);
            auto ptr =
                c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                    std::move(storage),
                    at::DispatchKeySet(),
                    at::CPU(scalar_type).typeMeta());
            return at::Tensor(std::move(ptr));
          })
      .def("get_all_records", [](PyTorchStreamReader& self) {
        return self.getAllRecords();
      });

  // Used by torch.Package to coordinate deserialization of storages across
  // ScriptModules and eager modules
  py::class_<
      DeserializationStorageContext,
      std::shared_ptr<DeserializationStorageContext>>(
      m, "DeserializationStorageContext")
      .def(py::init<>())
      .def(
          "get_storage",
          [](DeserializationStorageContext& self,
             const std::string& name,
             py::object data_type_obj) {
            c10::Storage storage = self.getStorage(name);
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
            auto ptr =
                c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                    std::move(storage),
                    at::DispatchKeySet(),
                    at::CPU(scalar_type).typeMeta());

            return at::Tensor(std::move(ptr));
          })
      .def(
          "add_storage",
          [](DeserializationStorageContext& self,
             const std::string& name,
             const at::Tensor& tensor) {
            return self.addStorage(name, tensor.storage());
          })
      .def("has_storage", &DeserializationStorageContext::hasStorage);

  m.def(
      "_get_schema",
      [](const std::string& op_name, const std::string& overload_name) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          for (const auto& op : operations) {
            if (op->schema().overload_name() == overload_name) {
              return op->schema();
            }
          }
          throw std::runtime_error("Found no matching schema");
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      });

  m.def(
      "_get_operation_overload",
      [](const std::string& op_name, const std::string& overload_name) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          bool allow_numbers_as_tensors = symbol.is_prims() ||
              symbol.is_nvprims() ||
              (symbol.is_aten() &&
               torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));
          for (const auto& op : operations) {
            if (op->schema().overload_name() == overload_name) {
              auto func =
                  py::cpp_function([op, symbol, allow_numbers_as_tensors](
                                       py::args args, py::kwargs kwargs) {
                    ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                    return _get_operation_for_overload_or_packet(
                        {op}, symbol, args, kwargs, /*is_overload*/ true);
                  });
              auto func_dk =
                  py::cpp_function([op, symbol, allow_numbers_as_tensors](
                                       const std::string& str_dk,
                                       py::args args,
                                       py::kwargs kwargs) {
                    c10::optional<c10::DispatchKey> dk =
                        c10::make_optional(c10::parseDispatchKey(str_dk));
                    ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                    return _get_operation_for_overload_or_packet(
                        {op}, symbol, args, kwargs, /*is_overload*/ true, dk);
                  });
              return py::make_tuple(
                  func, func_dk, py::cast(op->getTags().vec()));
            }
          }
          throw std::runtime_error("Found no matching operator overload");
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      });

  m.def(
      "_jit_get_operation",
      [](const std::string& op_name) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          TORCH_CHECK(!operations.empty(), "No such operator ", op_name);
          std::ostringstream docstring;
          docstring << "Automatically bound operator '" << op_name
                    << "' with schema(s):\n";

          for (const auto& op : operations) {
            docstring << "  " << op->schema() << "\n";
          }

          py::list overload_names;
          for (const auto& op : operations) {
            overload_names.append(py::str(op->schema().overload_name()));
          }

          bool allow_numbers_as_tensors = symbol.is_prims() ||
              symbol.is_nvprims() ||
              (symbol.is_aten() &&
               torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));

          auto func = py::cpp_function(
              [operations, symbol, allow_numbers_as_tensors](
                  py::args args, py::kwargs kwargs) {
                ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                return _get_operation_for_overload_or_packet(
                    operations, symbol, args, kwargs, false);
              },
              py::name(symbol.toUnqualString()),
              py::doc(docstring.str().c_str()));
          return py::make_tuple(func, overload_names);
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      },
      py::arg("qualified_name"));

  m.def(
      "parse_ir",
      [](const std::string& input, bool parse_tensor_constants) {
        auto graph = std::make_shared<Graph>();
        parseIR(input, &*graph, parse_tensor_constants);
        return graph;
      },
      py::arg("input"),
      py::arg("parse_tensor_constants") = false);
  m.def("parse_schema", parseSchema);
  m.def("unify_type_list", [](const std::vector<TypePtr>& types) {
    std::ostringstream s;
    auto type = unifyTypeList(types, s);
    if (!type) {
      throw std::runtime_error(s.str());
    }
    return type.value();
  });
  py::enum_<SchemaArgType>(m, "_SchemaArgType")
      .value("input", SchemaArgType::input)
      .value("output", SchemaArgType::output);
  py::class_<SchemaArgument>(m, "_SchemaArgument")
      .def(py::init<SchemaArgType, size_t>())
      .def_readwrite("type", &SchemaArgument::type)
      .def_readwrite("index", &SchemaArgument::index);
  py::class_<SchemaInfo>(m, "_SchemaInfo")
      .def(py::init<FunctionSchema>())
      .def("is_mutable", [](SchemaInfo& self) { return self.is_mutable(); })
      .def(
          "is_mutable",
          [](SchemaInfo& self, const SchemaArgument& argument) {
            return self.is_mutable(argument);
          })
      .def(
          "is_mutable",
          [](SchemaInfo& self, const std::string& name) {
            return self.is_mutable(name);
          })
      .def(
          "may_alias",
          [](SchemaInfo& self,
             const SchemaArgument& lhs,
             const SchemaArgument& rhs) { return self.may_alias(lhs, rhs); })
      .def(
          "may_contain_alias",
          [](SchemaInfo& self,
             const SchemaArgument& lhs,
             const SchemaArgument& rhs) {
            return self.may_contain_alias(lhs, rhs);
          })
      .def(
          "add_argument_value",
          [](SchemaInfo& self,
             const std::string& name,
             const py::object& value) {
            c10::optional<IValue> i_value = toTypeInferredIValueOptional(value);
            if (i_value) {
              // For normalization purposes there is an inconsistency within
              // torch.fx that turns all arguments named "self" into "input".
              // Thus this check ensures that those arguments are checked
              // correctly.
              if (name == "input" && !self.hasInputArgumentNamed("input")) {
                self.addArgumentValue("self", *i_value);
              } else {
                self.addArgumentValue(name, *i_value);
              }
            }
          })
      .def("add_argument_values", [](SchemaInfo& self, const py::dict& values) {
        std::unordered_map<std::string, IValue> value_map;
        for (const auto& key_pair : values) {
          IValue key = toTypeInferredIValue(key_pair.first);
          TORCH_INTERNAL_ASSERT(
              key.isString(),
              "Add argument value keys types should be strings.");
          c10::optional<IValue> value =
              toTypeInferredIValueOptional(key_pair.second);
          if (value) {
            // For normalization purposes there is an inconsistency within
            // torch.fx that
            // turns all arguments named "self" into "input". Thus this check
            // ensures that those arguments are checked correctly.
            if (key.toStringRef() == "input" &&
                !self.hasInputArgumentNamed("input")) {
              self.addArgumentValue("self", *value);
            } else {
              value_map[key.toStringRef()] = *value;
            }
          }
        }
        self.addArgumentValues(value_map);
      });
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
      .def(
          "is_backward_compatible_with",
          [](const FunctionSchema& self, const FunctionSchema& old_schema) {
            return self.isBackwardCompatibleWith(old_schema);
          })
      .def(
          "check_forward_compatible_with",
          [](const FunctionSchema& self, const FunctionSchema& old_schema) {
            std::ostringstream out;
            auto result = self.isForwardCompatibleWith(old_schema, out);
            return std::make_pair(result, out.str());
          })
      .def(
          "__eq__",
          [](const FunctionSchema& self, const FunctionSchema& other) {
            return self == other;
          })
      .def(
          "__str__",
          [](FunctionSchema& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def_property_readonly(
          "is_mutable", [](FunctionSchema& self) { return self.is_mutable(); });
  py::class_<Argument>(m, "Argument")
      .def_property_readonly("name", [](Argument& self) { return self.name(); })
      .def_property_readonly("type", [](Argument& self) { return self.type(); })
      .def_property_readonly(
          "N",
          [](Argument& self) -> py::object {
            return (self.N()) ? py::cast(*self.N()) : py::none();
          })
      .def_property_readonly(
          "default_value",
          [](Argument& self) -> py::object {
            if (!self.default_value()) {
              return py::none();
            }
            IValue v = *self.default_value();
            return toPyObject(std::move(v));
          })
      .def(
          "has_default_value",
          [](Argument& self) -> py::bool_ {
            return self.default_value().has_value();
          })
      .def_property_readonly(
          "alias_info", [](Argument& self) { return self.alias_info(); })
      .def_property_readonly(
          "is_out", [](Argument& self) { return self.is_out(); })
      .def_property_readonly("kwarg_only", [](Argument& self) -> bool {
        return self.kwarg_only();
      });
  py::class_<AliasInfo>(m, "_AliasInfo")
      .def_property_readonly(
          "is_write", [](AliasInfo& self) { return self.isWrite(); })
      .def_property_readonly(
          "before_set",
          [](AliasInfo& self) {
            std::set<py::str> before_set_python;
            for (const auto& set : self.beforeSets()) {
              before_set_python.insert(py::str(set.toUnqualString()));
            }
            return before_set_python;
          })
      .def_property_readonly("after_set", [](AliasInfo& self) {
        std::set<py::str> after_set_python;
        for (const auto& set : self.afterSets()) {
          after_set_python.insert(py::str(set.toUnqualString()));
        }
        return after_set_python;
      });
  m.def("_jit_get_all_schemas", []() {
    const std::vector<std::shared_ptr<Operator>>& operations =
        getAllOperators();
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });
  m.def("_jit_get_custom_class_schemas", customClassSchemasForBCCheck);
  m.def("_jit_get_schemas_for_operator", [](const std::string& qualified_name) {
    auto symbol = Symbol::fromQualString(qualified_name);
    const auto& operations = getAllOperatorsFor(symbol);
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });
  m.def("_is_tracing", []() { return jit::tracer::isTracing(); });

  py::class_<PythonFutureWrapper, std::shared_ptr<PythonFutureWrapper>>(
      m, "Future")
      .def(py::init([](std::vector<c10::Device> devices = {}) {
        return std::make_shared<PythonFutureWrapper>(
            c10::make_intrusive<c10::ivalue::Future>(
                PyObjectType::get(), std::move(devices)));
      }))
      .def(
          "done",
          // Intentionally not releasing GIL
          &PythonFutureWrapper::done)
      .def(
          "value",
          &PythonFutureWrapper::value,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait",
          &PythonFutureWrapper::wait,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "then",
          &PythonFutureWrapper::then,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_done_callback",
          &PythonFutureWrapper::add_done_callback,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_result",
          // Intentionally not releasing GIL
          &PythonFutureWrapper::markCompleted)
      .def(
          "_set_unwrap_func",
          // Intentionally not releasing GIL as this just does an assign
          [](PythonFutureWrapper& self, py::function unwrapFunc) {
            auto functionGuard =
                std::make_shared<torch::jit::PythonFunctionGuard>(
                    std::move(unwrapFunc));

            std::function<void(py::object)> pf =
                [functionGuard(std::move(functionGuard))](
                    const py::object& inp) {
                  return functionGuard->func_(inp);
                };
            self.unwrap_func = std::move(pf);
          })
      .def(
          py::pickle(
              /* __getstate__ */
              [](const PythonFutureWrapper& /* unused */) {
                TORCH_CHECK(false, "Can not pickle torch.futures.Future");
                // Note that this return has no meaning since we always
                // throw, it's only here to satisfy Pybind API's
                // requirement.
                return py::make_tuple();
              },
              /* __setstate__ */
              [](const py::tuple& /* unused */) { // NOLINT
                TORCH_CHECK(false, "Can not unpickle torch.futures.Future");
                // Note that this return has no meaning since we always
                // throw, it's only here to satisfy PyBind's API
                // requirement.
                return nullptr;
              }),
          py::call_guard<py::gil_scoped_release>());
  m.def("_is_alias_of", [](const py::object& self, const py::object& other) {
    c10::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    c10::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // Only return true if we are certain that self and other are aliasing.
    if (!self_value || !other_value) {
      return false;
    }
    return self_value->isAliasOf(*other_value);
  });
  m.def("_overlaps", [](const py::object& self, const py::object& other) {
    c10::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    c10::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // Only return true if we are certain that self and other are overlapping.
    if (!self_value || !other_value) {
      return false;
    }
    return self_value->overlaps(*other_value);
  });
  m.def("fork", [](const py::args& args, const py::kwargs& kwargs) {
    AT_ASSERT(args.size() >= 1);

    py::function f = py::cast<py::function>(args[0]);
    py::tuple args_tup(args.size() - 1);

    for (const auto i : c10::irange(1, args.size())) {
      args_tup[i - 1] = args[i];
    }

    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;
      auto fork_node = graph->insertNode(graph->create(prim::TracedFork, 1));
      auto body_block = fork_node->addBlock();

      Value* node_output = nullptr;
      py::object py_func_output;
      // Insert new trace ops into the fork op's sub-block
      WithInsertPoint guard(body_block);
      IValue output_ivalue;
      {
        tracer::WithNestedTracingFrame env_guard;

        // Run the user-supplied function
        py_func_output = f(*args_tup, **kwargs);

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

      return std::make_shared<PythonFutureWrapper>(retval);
    } else {
      auto result = toTypeInferredIValue(f(*args_tup, **kwargs));
      auto retval = c10::make_intrusive<c10::ivalue::Future>(result.type());
      retval->markCompleted(std::move(result));
      return std::make_shared<PythonFutureWrapper>(retval);
    }
  });

  m.def("wait", [](const std::shared_ptr<PythonFutureWrapper>& fut) {
    return fut->wait();
  });

  m.def(
      "_collect_all",
      [](const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>& futures)
          -> std::shared_ptr<jit::PythonFutureWrapper> {
        auto typePtr =
            futures.empty() ? AnyType::get() : futures[0]->fut->elementType();
        c10::List<c10::intrusive_ptr<c10::ivalue::Future>> asList(
            c10::FutureType::create(typePtr));
        asList.reserve(futures.size());
        for (const auto& f : futures) {
          asList.push_back(f->fut);
        }
        return std::make_shared<jit::PythonFutureWrapper>(
            c10::collectAll(asList),
            /* unwrap_func */ [futures](const py::object& /*unused*/) {
              // Throw errors when calling wait() on the returned Future if
              // any of the original futures would throw.
              // NB: PythonFutureWrapper takes an unwrap_func which serves as a
              // callback to evalute the value in the Future. RPC uses this
              // unwrap_func to check whether the returned py::object is a
              // RemoteException object, and re-throw the exception if it is.
              // By extracting the c10::ivalue::Future from PythonFutureWrapper
              // the unwrap_func on the original PythonFutureWrapper objects are
              // discarded, and hence it will return the RemoteException as an
              // object instead of re-throwing it.
              for (auto& fut : futures) {
                fut->wait();
              }
            });
      },
      py::call_guard<py::gil_scoped_release>());

  m.def("_jit_assert_is_instance", [](py::object obj, const TypePtr& type) {
    toIValue(std::move(obj), type);
  });

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
  m.def("_set_print_stack_traces_on_fatal_signal", [](bool print) {
    c10::FatalSignalHandler::getInstance().setPrintStackTracesOnFatalSignal(
        print);
  });
#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)

  initPythonCustomClassBindings(module);
  initPythonIRBindings(module);
  tracer::initPythonTracerBindings(module);
  initTreeViewBindings(module);
  initJitScriptBindings(module);
  initJitBackendBindings(module);
  initStaticModuleBindings(module);
  initTensorExprBindings(module);
  initNvFuserPythonBindings(module);

  setPrintHandler([](const std::string& str) {
    py::gil_scoped_acquire acquire;
    try {
      auto _stdout = py::module::import("sys").attr("stdout");
      _stdout.attr("write")(str);
    } catch (py::error_already_set& e) {
      throw std::runtime_error(e.what());
    }
  });

  // On exit we need to reset the print handler to default one,
  // because otherwise prim::Print() instruction won't work for JIT modules.
  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(
      py::cpp_function([]() { setPrintHandler(getDefaultPrintHandler()); }));
}
} // namespace jit
} // namespace torch
