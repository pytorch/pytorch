#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/schema_info.h>

#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
// #include <torch/csrc/jit/codegen/cuda/python_frontend/python_bindings.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))
#include <torch/csrc/jit/codegen/onednn/interface.h>
#endif
#include <c10/core/SymNodeImpl.h>
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
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
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
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
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
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>
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

namespace torch::jit {

using c10::AliasInfo;
using c10::Argument;
using c10::FunctionSchema;
using c10::SchemaArgType;
using c10::SchemaArgument;
using c10::SymNode;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using torch::utils::SchemaInfo;

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

static bool opAllowsNumbersAsTensors(c10::Symbol symbol) {
  return symbol.is_prims() || symbol.is_nvprims() ||
      (symbol.is_aten() &&
       torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));
}

std::optional<IValue> toTypeInferredIValueOptional(py::handle input) {
  // Errors need to be caught here because toTypeInferredIValue errors out
  // on various object types, but we want it to work with all types.
  try {
    return toTypeInferredIValue(input);
  } catch (const c10::Error& e) {
    return std::nullopt;
  }
}
} // anonymous namespace

#if !defined(USE_ROCM)
TORCH_API void runJITCPPTests();
#endif

void initJITBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_jit");

  // This is a static object, so we must leak the Python object
  // "release()" is used here to preserve 1 refcount on the
  // object, preventing it from ever being de-allocated by CPython.
  static py::handle exc =
      py::exception<JITException>(m, "JITException").release();

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
      // If we still had the py::exception<JITException> object, we could
      // just call it. But we must get a handle to leak it and there is no
      // way I can find to re-create it from the handle. So setting the
      // exception manually
      PyErr_SetString(exc.ptr(), e.what());
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
          [](Node* n) -> std::optional<std::shared_ptr<Graph>> {
            if (!n->maybeSchema()) {
              return std::nullopt;
            }
            return shapeComputeGraphForSchema(n->schema());
          })
      .def(
          "_jit_decomposition_graph_for_node",
          [](Node* n) -> std::optional<std::shared_ptr<Graph>> {
            if (!n->maybeSchema()) {
              return std::nullopt;
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
                std::optional<std::tuple<Module, Module>>>>(qconfig_dict);
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
                std::optional<std::tuple<Module, Module>>>>(qconfig_dict);
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
      .def("_jit_pass_fold_frozen_linear_bn", &FoldFrozenLinearBatchnorm)
      .def("_jit_pass_convert_frozen_ops_to_mkldnn", &ConvertFrozenOpsToMKLDNN)
      .def("_jit_pass_fuse_frozen_conv_add_relu", &FuseFrozenConvAddRelu)
      .def("_jit_pass_transpose_frozen_linear", &FrozenLinearTranspose)
      .def("_jit_pass_optimize_frozen_graph", &OptimizeFrozenGraph)
      .def(
          "_jit_pass_optimize_for_inference",
          [](Module& module, const std::vector<std::string>& other_methods) {
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
                input_types.emplace_back(tt);
              } else {
                input_types.emplace_back(nullptr);
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
          [](const std::shared_ptr<Graph>& graph, const py::object& threshold) {
            if (threshold.is_none()) {
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
#else
      .def("_jit_set_llga_enabled", [](bool flag) { return false; })
      .def("_jit_llga_enabled", []() { return false; })
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
          [](const std::string& op_name, bool flip = true) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_skip_node_kind is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_enabled",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_can_be_enabled",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_can_be_enabled is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_single_node_mode",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_single_node_mode is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_single_node_mode",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_single_node_mode is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_horizontal_mode",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_horizontal_mode is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_horizontal_mode",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_horizontal_mode is deprecated and a no-op");
          })
      .def(
          "_jit_set_nvfuser_guard_mode",
          [](bool) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_set_nvfuser_guard_mode is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_enabled",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_enabled is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_set_comparison_callback",
          [](bool, py::function) {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_set_comparison_callback is deprecated and a no-op");
          })
      .def(
          "_jit_nvfuser_clear_comparison_callback",
          []() {
            TORCH_WARN(
                "nvfuser is no longer supported in torch script, use _jit_nvfuser_clear_comparison_callback is deprecated and a no-op");
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
          [](const std::vector<std::pair<std::string, size_t>>& strategy) {
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
            ::torch::jit::set_jit_logging_levels(std::move(loggingOption));
          })
      .def(
          "_jit_set_logging_stream",
          [](const std::string& stream_name) -> void {
            if (stream_name == "stdout") {
              ::torch::jit::set_jit_logging_output_stream(std::cout);
            } else if (stream_name == "stderr") {
              ::torch::jit::set_jit_logging_output_stream(std::cerr);
            } else {
              std::cerr << "ERROR: only `stdout` and `stderr`"
                        << "are supported as output options" << '\n';
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
             std::set<MobileOptimizerType>& optimization_blocklist,
             std::vector<std::string>& preserved_methods) {
            return vulkanOptimizeForMobile(
                module, optimization_blocklist, preserved_methods);
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
      .def(
          "_jit_decay_packed_param_input_types",
          [](Graph& g) {
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
                    std::nullopt));
              }
            }
          })
      .def("_jit_set_utf8_decoding_ignore", &setUTF8DecodingIgnore);

  // NB: This isn't actually used for regular PyTorch symbolic tracing;
  // XLA is what needs this
#define SYMNODE_UNARY(n) .def(#n, [](const c10::SymNode& a) { return a->n(); })
#define SYMNODE_BINARY(n) \
  .def(#n, [](const c10::SymNode& a, const c10::SymNode& b) { return a->n(b); })
#define SYMNODE_SIZES_STRIDES(n)                \
  .def(                                         \
      #n,                                       \
      [](const c10::SymNode& a,                 \
         c10::ArrayRef<c10::SymNode> sizes,     \
         c10::ArrayRef<c10::SymNode> strides) { \
        return a->n(sizes, strides);            \
      })
  auto symnode_class =
      py::class_<c10::SymNodeImpl, c10::SymNode>(m, "_SymNode")
      // clang-format off
      // These DO NOT install magic methods; the SymInt/SymFloat wrapper in
      // Python is responsible for this
      SYMNODE_UNARY(clone)
      SYMNODE_UNARY(is_int)
      SYMNODE_UNARY(is_float)
      SYMNODE_UNARY(is_bool)
      SYMNODE_UNARY(bool_)
      SYMNODE_UNARY(int_)
      SYMNODE_UNARY(sym_float)
      SYMNODE_BINARY(add)
      SYMNODE_BINARY(sub)
      SYMNODE_BINARY(mul)
      SYMNODE_BINARY(truediv)
      SYMNODE_BINARY(int_truediv)
      SYMNODE_BINARY(float_truediv)
      SYMNODE_BINARY(pow)
      SYMNODE_BINARY(float_pow)
      SYMNODE_BINARY(pow_by_natural)
      SYMNODE_BINARY(floordiv)
      SYMNODE_BINARY(int_floordiv)
      SYMNODE_BINARY(mod)
      SYMNODE_BINARY(eq)
      SYMNODE_BINARY(ne)
      SYMNODE_BINARY(gt)
      SYMNODE_BINARY(lt)
      SYMNODE_BINARY(le)
      SYMNODE_BINARY(ge)
      SYMNODE_BINARY(sym_min)
      SYMNODE_BINARY(sym_max)
      SYMNODE_BINARY(sym_and)
      SYMNODE_BINARY(sym_or)
      SYMNODE_UNARY(sym_not)
      SYMNODE_UNARY(ceil)
      SYMNODE_UNARY(floor)
      SYMNODE_UNARY(neg)
      SYMNODE_SIZES_STRIDES(is_contiguous)
      SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_2d)
      SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_3d)
      SYMNODE_SIZES_STRIDES(is_channels_last_strides_2d)
      SYMNODE_SIZES_STRIDES(is_channels_last_strides_3d)
      SYMNODE_SIZES_STRIDES(is_non_overlapping_and_dense)
      .def(
          "guard_int",
          [](const c10::SymNode& a, const char* file, int64_t line) {
            return a->guard_int(file, line);
          })
      .def(
          "guard_bool",
          [](const c10::SymNode& a, const char* file, int64_t line) {
            return a->guard_bool(file, line);
          })
      .def(
          "guard_float",
          [](const c10::SymNode& a, const char* file, int64_t line) {
            return a->guard_float(file, line);
          })
      .def(
          "expect_true",
          [](const c10::SymNode& a, const char* file, int64_t line) {
            return a->expect_true(file, line);
          })
      .def(
          "expect_size",
          [](const c10::SymNode& a, const char* file, int64_t line) {
            return a->expect_size(file, line);
          })
      .def(
          "guard_size_oblivious",
          [](const c10::SymNode& a, const char* file, int64_t line) {
            return a->guard_size_oblivious(file, line);
          })
      .def(
          "has_hint",
          [](const c10::SymNode& a) {
            return a->has_hint();
          })
      .def(
          "wrap_int",
          [](const c10::SymNode& a, int64_t b) {
            return a->wrap_int(b);
          })
      .def(
          "wrap_float",
          [](const c10::SymNode& a, double b) {
            return a->wrap_float(b);
          })
      .def(
          "wrap_bool",
          [](const c10::SymNode& a, bool b) {
            return a->wrap_bool(b);
          })
      .def(
          "__str__",
          [](const c10::SymNode& a) { return a->str(); })
      .def(
          "__repr__",
          [](const c10::SymNode& a) { return a->str(); })
      .def(
          "_graph_repr",
          [](const c10::SymNode& a) { return a->_graph_repr(); })
      .def(
          "is_constant",
          [](const c10::SymNode& node){
            return node->is_constant();
          })
      .def(
          "is_nested_int",
          [](const c10::SymNode& node) {
            return node->is_nested_int();
          })
      .def(
          "is_symbolic",
          [](const c10::SymNode& node) {
            return node->is_symbolic();
          })
      .def(
          "nested_int",
          [](const c10::SymNode& node) {
            return node->nested_int();
          })
      .def(
          "nested_int_coeff",
          [](const c10::SymNode& node) {
            return node->nested_int_coeff();
          })
      .def(
          "__deepcopy__",
          [](const c10::SymNode& node, py::handle memo) {
            return node->clone();
          });

  // clang-format on

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
      .def(
          py::init<std::string, bool>(),
          py::arg("file_name"),
          py::arg("compute_crc32") = true)
      .def(
          py::init([](const py::object& buffer, bool compute_crc32 = true) {
            auto writer_func = [=](const void* data, size_t size) {
              // Writing an empty file is a noop
              if (size == 0) {
                return size;
              }
              py::gil_scoped_acquire acquire;
              if (!data) {
                // See [Note: write_record_metadata]
                buffer.attr("seek")(
                    size, py::module::import("os").attr("SEEK_CUR"));
              } else {
                auto memory_view = py::memoryview::from_memory(
                    reinterpret_cast<const char*>(data), size);
                buffer.attr("write")(std::move(memory_view));
              }
              return size;
            };
            return std::make_unique<PyTorchStreamWriter>(
                std::move(writer_func), compute_crc32);
          }),
          py::arg("buffer"),
          py::arg("compute_crc32") = true)
      .def(
          py::init<const std::function<size_t(const void*, size_t)>&, bool>(),
          py::arg("writer_func"),
          py::arg("compute_crc32") = true)
      // [Note: write_record_metadata]
      // The write_record_metadata function is intended to write metadata (i.e.
      // the zipfile header and end of central directory record) for a file
      // while reserving nbytes of space for the file for the bytes of the
      // actual file to be added in later. This functionality is achieved by
      // defining `m_pWrite` to seek instead of write if the buffer passed is a
      // nullptr. This has implications on CRC-32 which will not be written at
      // write_record_metadata time, and will not be combined with the hash in
      // combined_uncomp_crc32_. We define this in `m_pWrite` rather than
      // extending the interface of miniz to have an `m_pSeek` since different
      // versions of miniz are used in fbcode/oss.
      .def(
          "write_record_metadata",
          [](PyTorchStreamWriter& self, const std::string& name, size_t size) {
            return self.writeRecord(name, nullptr, size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             const char* data,
             size_t size) {
            // Since we don't know where the data come from, we cannot
            // release the GIL in this overload
            return self.writeRecord(name, data, size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             py::bytes data,
             size_t size) {
            // It is not clear from the doc but according to CPython own code,
            // it is ok to use the result of PyBytes_AsString without the GIL
            // being held
            // https://github.com/python/cpython/blob/e2a3e4b7488aff6fdc704a0f258bc315e96c1d6e/Objects/stringlib/join.h#L67
            const char* data_str = PyBytes_AsString(data.ptr());
            py::gil_scoped_release release;
            return self.writeRecord(name, data_str, size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             const c10::Storage& data,
             size_t size) {
            // Reading Tensor data is always ok without the GIL held
            py::gil_scoped_release release;
            return self.writeRecord(
                name, reinterpret_cast<const char*>(data.data()), size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             uintptr_t data,
             size_t size) {
            TORCH_WARN_ONCE(
                "write_record(): Passing Storage by data pointer is deprecated and will be an error in ",
                "the future, please pass the Storage object instead.");
            return self.writeRecord(
                name, reinterpret_cast<const char*>(data), size);
          })
      .def("write_end_of_file", &PyTorchStreamWriter::writeEndOfFile)
      .def("set_min_version", &PyTorchStreamWriter::setMinVersion)
      .def("archive_name", &PyTorchStreamWriter::archiveName)
      .def("serialization_id", &PyTorchStreamWriter::serializationId)
      .def(
          "get_all_written_records",
          &PyTorchStreamWriter::getAllWrittenRecords);

  py::enum_<MobileOptimizerType>(m, "_MobileOptimizerType")
      .value("CONV_BN_FUSION", MobileOptimizerType::CONV_BN_FUSION)
      .value(
          "INSERT_FOLD_PREPACK_OPS",
          MobileOptimizerType::INSERT_FOLD_PREPACK_OPS)
      .value("REMOVE_DROPOUT", MobileOptimizerType::REMOVE_DROPOUT)
      .value("FUSE_ADD_RELU", MobileOptimizerType::FUSE_ADD_RELU)
      .value(
          "HOIST_CONV_PACKED_PARAMS",
          MobileOptimizerType::HOIST_CONV_PACKED_PARAMS)
      .value(
          "VULKAN_AUTOMATIC_GPU_TRANSFER",
          MobileOptimizerType::VULKAN_AUTOMATIC_GPU_TRANSFER);

  // This allows PyTorchStreamReader to read from a Python buffer. It requires
  // that the buffer implement `seek()`, `tell()`, and `read()`.
  class BufferAdapter : public caffe2::serialize::ReadAdapterInterface {
   public:
    BufferAdapter(const py::object& buffer) : buffer_(buffer) {
      // Jump to the end of the buffer to get its size
      auto current = buffer.attr("tell")();
      start_offset_ = py::cast<size_t>(current);
      buffer.attr("seek")(0, py::module::import("os").attr("SEEK_END"));
      size_ = py::cast<size_t>(buffer.attr("tell")()) - start_offset_;
      buffer.attr("seek")(current);
      // If we can read directly into a buffer, do that instead of an extra copy
      // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
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
          Py_DECREF(res);
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
    bool use_readinto_{};
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
            auto [data, size] = self.getRecord(key);
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
      .def("serialization_id", &PyTorchStreamReader::serializationId)
      .def(
          "get_all_records",
          [](PyTorchStreamReader& self) { return self.getAllRecords(); })
      .def(
          "get_record_offset",
          [](PyTorchStreamReader& self, const std::string& key) {
            return self.getRecordOffset(key);
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
      [](const std::string& op_name,
         const std::string& overload_name) -> std::optional<py::tuple> {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          bool allow_numbers_as_tensors = opAllowsNumbersAsTensors(symbol);
          for (const auto& op : operations) {
            if (op->schema().overload_name() == overload_name) {
              auto func = py::cpp_function(
                  [op, symbol, allow_numbers_as_tensors](
                      const py::args& args, const py::kwargs& kwargs) {
                    ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                    return _get_operation_for_overload_or_packet(
                        {op}, symbol, args, kwargs, /*is_overload*/ true);
                  });
              auto func_dk =
                  py::cpp_function([op, symbol, allow_numbers_as_tensors](
                                       c10::DispatchKey dk_,
                                       const py::args& args,
                                       const py::kwargs& kwargs) {
                    ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                    return _get_operation_for_overload_or_packet(
                        {op}, symbol, args, kwargs, /*is_overload*/ true, dk_);
                  });
              return py::make_tuple(
                  func, func_dk, py::cast(op->getTags().vec()));
            }
          }
          return std::nullopt;
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      });

  m.def(
      "_check_schema_allow_fake_script_object",
      [](const FunctionSchema& schema,
         const py::args& args,
         const py::kwargs& kwargs) {
        // checkSchemaAllowFakeScriptObject will throw runtime error if there is
        // a schema mismatch. Otherwise, it returns true.
        return checkSchemaAllowFakeScriptObject(schema, args, kwargs);
      });

  m.def(
      "_jit_resolve_packet",
      [](const char* op_name, py::args args, const py::kwargs& kwargs) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          bool allow_numbers_as_tensors = opAllowsNumbersAsTensors(symbol);
          ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
          const auto overloads = getAllSortedOperatorsFor(symbol);
          auto opWithStack = getOpWithStack(overloads, args, kwargs);
          std::shared_ptr<Operator> overload = std::get<0>(opWithStack);
          auto result = overload->schema().overload_name();
          if (result.empty()) {
            result = "default";
          }
          return result;
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
          const auto sortedOps = getAllSortedOperatorsFor(symbol);
          if (sortedOps.empty()) {
            // No such operator
            return py::make_tuple(py::none(), py::none());
          }

          std::ostringstream docstring;
          docstring << "Automatically bound operator '" << op_name
                    << "' with schema(s):\n";

          for (const auto& op : sortedOps) {
            docstring << "  " << op->schema() << "\n";
          }

          py::list overload_names;
          for (const auto& op : sortedOps) {
            overload_names.append(py::str(op->schema().overload_name()));
          }

          bool allow_numbers_as_tensors = opAllowsNumbersAsTensors(symbol);

          auto func = py::cpp_function(
              [sortedOps, symbol, allow_numbers_as_tensors](
                  const py::args& args, const py::kwargs& kwargs) {
                ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                return _get_operation_for_overload_or_packet(
                    sortedOps, symbol, args, kwargs, false);
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
      "_maybe_call_torch_function_for_op_packet",
      [](py::handle op_overload_packet,
         const py::args& args,
         const py::kwargs& kwargs) {
        py::list ns_method =
            op_overload_packet.attr("_qualified_op_name").attr("split")("::");
        auto res = _maybe_handle_torch_function(
            py::cast<std::string>(ns_method[0]),
            py::cast<std::string>(ns_method[1]),
            "",
            false,
            args,
            kwargs);
        if (res) {
          return py::make_tuple(true, *res);
        } else {
          return py::make_tuple(false, py::none());
        }
      });

  m.def(
      "parse_ir",
      [](const std::string& input, bool parse_tensor_constants) {
        auto graph = std::make_shared<Graph>();
        parseIR(input, &*graph, parse_tensor_constants);
        return graph;
      },
      py::arg("input"),
      py::arg("parse_tensor_constants") = false);
  m.def(
      "parse_schema",
      &parseSchema,
      py::arg("schema"),
      py::arg("allow_typevars") = true);
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
          "has_argument",
          [](SchemaInfo& self, const std::string& name) {
            return self.has_argument(name);
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
            std::optional<IValue> i_value = toTypeInferredIValueOptional(value);
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
          std::optional<IValue> value =
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
          "__hash__",
          [](const FunctionSchema& self) {
            return std::hash<FunctionSchema>{}(self);
          })
      .def(
          "__str__",
          [](FunctionSchema& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def(
          "__repr__",
          [](FunctionSchema& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def(py::pickle(
          [](const FunctionSchema& self) { // __getstate__
            std::stringstream ss;
            ss << self;
            return py::str(ss.str());
          },
          [](const py::str& schema) { // __setstate__, note: no `self` argument
            return parseSchema(schema);
          }))
      .def_property_readonly(
          "is_mutable", [](FunctionSchema& self) { return self.is_mutable(); });
  py::class_<Argument>(m, "Argument")
      .def_property_readonly("name", [](Argument& self) { return self.name(); })
      .def_property_readonly("type", [](Argument& self) { return self.type(); })
      .def_property_readonly(
          "real_type", [](Argument& self) { return self.real_type(); })
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
          "is_write",
          [](Argument& self) {
            if (self.alias_info() == nullptr) {
              return false;
            }
            return self.alias_info()->isWrite();
          })
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

  py::class_<PythonAwaitWrapper, std::shared_ptr<PythonAwaitWrapper>>(
      m, "_Await")
      .def(
          "wait",
          &PythonAwaitWrapper::wait,
          py::call_guard<py::gil_scoped_release>())
      .def("fn", &PythonAwaitWrapper::fn)
      .def("args", &PythonAwaitWrapper::args)
      .def("type", &PythonAwaitWrapper::type)
      .def("is_nowait", &PythonAwaitWrapper::is_nowait)
      .def(
          "__getattr__",
          [](PythonAwaitWrapper& self, const std::string& name) -> py::object {
            // In eager mode allow Await[W] to be used as W, redirecting getattr
            // to the result of delayed function.
            return py::getattr(self.wait(), name.c_str(), py::none());
          })
      .def(
          py::pickle(
              /* __getstate__ */
              [](const PythonAwaitWrapper& /* unused */) {
                TORCH_CHECK(false, "Can not pickle torch.jit._Await");
                // Note that this return has no meaning since we always
                // throw, it's only here to satisfy Pybind API's
                // requirement.
                return py::make_tuple();
              },
              /* __setstate__ */
              [](const py::tuple& /* unused */) { // NOLINT
                TORCH_CHECK(false, "Can not unpickle torch.jit._Await");
                // Note that this return has no meaning since we always
                // throw, it's only here to satisfy PyBind's API
                // requirement.
                return nullptr;
              }),
          py::call_guard<py::gil_scoped_release>());

  m.def("_is_alias_of", [](const py::object& self, const py::object& other) {
    std::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    std::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // Only return true if we are certain that self and other are aliasing.
    if (!self_value || !other_value) {
      return false;
    }
    return self_value->isAliasOf(*other_value);
  });
  m.def("_overlaps", [](const py::object& self, const py::object& other) {
    std::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    std::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // Only return true if we are certain that self and other are overlapping.
    if (!self_value || !other_value) {
      return false;
    }
    return self_value->overlaps(*other_value);
  });
  m.def("_awaitable", [](const py::args& args, const py::kwargs& kwargs) {
    AT_ASSERT(!args.empty());
    py::tuple args_tup(args.size() - 1);
    for (const auto i : c10::irange(1, args.size())) {
      args_tup[i - 1] = args[i];
    }
    return std::make_shared<PythonAwaitWrapper>(
        py::cast<py::function>(args[0]), std::move(args_tup));
  });
  m.def("_awaitable_nowait", [](py::handle input) {
    return std::make_shared<PythonAwaitWrapper>(input);
  });
  m.def(
      "_awaitable_wait", [](const std::shared_ptr<PythonAwaitWrapper>& py_aw) {
        TORCH_CHECK(py_aw, "Await can't be None");
        return py_aw->wait();
      });
  m.def("fork", [](const py::args& args, const py::kwargs& kwargs) {
    AT_ASSERT(!args.empty());

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
    TORCH_CHECK(fut, "Future can't be None");
    return fut->wait();
  });

  m.def(
      "_collect_all",
      [](const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>& futures)
          -> std::shared_ptr<jit::PythonFutureWrapper> {
        auto typePtr = futures.empty() || futures[0] == nullptr
            ? AnyType::get()
            : futures[0]->fut->elementType();
        c10::List<c10::intrusive_ptr<c10::ivalue::Future>> asList(
            c10::FutureType::create(typePtr));
        asList.reserve(futures.size());
        for (const auto& f : futures) {
          TORCH_CHECK(f, "Future can't be None");
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
  // initNvFuserPythonBindings(module);

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

} // namespace torch::jit
