#include <onnx/onnx_pb.h>
#include <torch/csrc/onnx/back_compat.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/onnx/onnx.h>
#include <torch/version.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/deduplicate_initializers.h>
#include <torch/csrc/jit/passes/onnx/eliminate_unused_items.h>
#include <torch/csrc/jit/passes/onnx/eval_peephole.h>
#include <torch/csrc/jit/passes/onnx/fixup_onnx_controlflow.h>
#include <torch/csrc/jit/passes/onnx/function_extraction.h>
#include <torch/csrc/jit/passes/onnx/function_substitution.h>
#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>
#include <torch/csrc/jit/passes/onnx/naming.h>
#include <torch/csrc/jit/passes/onnx/onnx_log.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/autograd_function_process.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_conversion.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>
#include <torch/csrc/jit/passes/onnx/peephole.h>
#include <torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/preprocess_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch::onnx {

using namespace torch::jit;

void initONNXBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // ONNX specific passes
  m.def("_jit_pass_onnx_remove_print", RemovePrintOps)
      .def("_jit_pass_onnx_preprocess_caffe2", PreprocessCaffe2Ops)
      .def("_jit_pass_onnx", ToONNX)
      .def(
          "_jit_pass_onnx_assign_output_shape",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 const std::vector<at::Tensor>& tensors,
                 const python::IODescriptor& desc,
                 bool onnx_shape_inference,
                 bool is_script,
                 int opset_version) {
                ONNXAssignOutputShape(
                    graph,
                    tensors,
                    desc,
                    onnx_shape_inference,
                    is_script,
                    opset_version);
              }))
      .def(
          "_jit_pass_onnx_function_substitution",
          wrap_pybind_function(ONNXFunctionCallSubstitution))
      .def(
          "_jit_pass_onnx_autograd_function_process",
          wrap_pybind_function(ONNXAutogradFunctionProcess))
      .def(
          "_jit_pass_onnx_peephole",
          ::torch::wrap_pybind_function([](std::shared_ptr<Graph>& graph,
                                           int opset_version,
                                           bool fixed_batch_size) {
            return PeepholeOptimizeONNX(graph, opset_version, fixed_batch_size);
          }))
      .def(
          "_jit_pass_onnx_preprocess",
          ::torch::wrap_pybind_function(PreprocessForONNX))
      .def(
          "_jit_pass_onnx_eval_peephole",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue>& paramsDict) {
                EvalPeepholeONNX(graph, paramsDict);
                return paramsDict;
              }),
          pybind11::return_value_policy::move)
      .def(
          "_jit_pass_onnx_cast_all_constant_to_floating",
          ::torch::wrap_pybind_function(CastAllConstantToFloating))
      .def(
          "_jit_pass_onnx_constant_fold",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue>& paramsDict,
                 int opset_version) {
                ConstantFoldONNX(
                    graph,
                    paramsDict,
                    opset_version); // overload resolution
                return paramsDict;
              }),
          pybind11::return_value_policy::move)
      .def(
          "_jit_pass_onnx_eliminate_unused_items",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue>& paramsDict) {
                EliminateUnusedItemsONNX(
                    graph->block(),
                    paramsDict); // overload resolution
                return paramsDict;
              }),
          pybind11::return_value_policy::move)
      .def(
          "_jit_pass_onnx_scalar_type_analysis",
          ::torch::wrap_pybind_function([](std::shared_ptr<Graph>& graph,
                                           bool lowprecision_cast,
                                           int opset_version) {
            return ScalarTypeAnalysisForONNX(
                graph, lowprecision_cast, opset_version);
          }),
          py::arg("graph"),
          py::arg("lowprecision_cast") = true,
          py::arg("opset_version"))
      .def(
          "_jit_pass_onnx_remove_inplace_ops_for_onnx",
          ::torch::wrap_pybind_function(RemoveInplaceOpsForONNX))
      .def(
          "_jit_pass_onnx_node_shape_type_inference",
          ::torch::wrap_pybind_function(
              [](Node* n,
                 std::map<std::string, IValue>& params_dict,
                 int opset_version) {
                ONNXShapeTypeInference(n, params_dict, opset_version);
              }))
      .def(
          "_jit_pass_onnx_graph_shape_type_inference",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue>& params_dict,
                 int opset_version) {
                ONNXShapeTypeInference(graph, params_dict, opset_version);
              }),
          py::arg("graph"),
          py::arg("params_dict"),
          py::arg("opset_version"))
      .def(
          "_jit_pass_onnx_set_dynamic_input_shape",
          ::torch::wrap_pybind_function(ONNXSetDynamicInputShape))
      .def("_jit_pass_onnx_lint", torch::wrap_pybind_function(ONNXLintGraph))
      .def(
          "_jit_pass_onnx_function_extraction",
          ::torch::wrap_pybind_function(
              torch::jit::onnx::ONNXFunctionExtraction))
      .def("_jit_pass_onnx_block", torch::wrap_pybind_function(BlockToONNX))
      .def(
          "_jit_pass_onnx_unpack_quantized_weights",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue>& paramsDict) {
                UnpackQuantizedWeights(graph, paramsDict);
                return paramsDict;
              }),
          pybind11::return_value_policy::move)
      .def(
          "_jit_pass_onnx_quantization_insert_permutes",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue>& paramsDict) {
                insertPermutes(graph, paramsDict);
                return paramsDict;
              }),
          pybind11::return_value_policy::move)
      .def(
          "_jit_onnx_list_model_parameters",
          ::torch::wrap_pybind_function(
              [](Module& module) { return list_module_parameters(module); }))
      .def(
          "_jit_pass_prepare_division_for_onnx",
          ::torch::wrap_pybind_function(PrepareDivisionForONNX))
      .def(
          "_jit_onnx_convert_pattern_from_subblock",
          ::torch::wrap_pybind_function(ConvertPatternFromSubblock))
      .def(
          "_jit_pass_fixup_onnx_controlflow_node",
          ::torch::wrap_pybind_function(FixupONNXControlflowNode))
      .def(
          "_jit_pass_onnx_deduplicate_initializers",
          ::torch::wrap_pybind_function(
              [](std::shared_ptr<Graph>& graph,
                 std::map<std::string, IValue> params_dict,
                 bool is_train) {
                DeduplicateInitializers(graph, params_dict, is_train);
                return params_dict;
              }),
          pybind11::return_value_policy::move)
      .def(
          "_jit_pass_onnx_clear_scope_records",
          &torch::jit::onnx::ONNXClearScopeRecords)
      .def(
          "_jit_pass_onnx_track_scope_attributes",
          &torch::jit::onnx::ONNXTrackScopeAttributes)
      .def(
          "_jit_is_onnx_log_enabled",
          ::torch::jit::onnx::is_log_enabled,
          "Returns whether ONNX logging is enabled or disabled.")
      .def(
          "_jit_set_onnx_log_enabled",
          ::torch::jit::onnx::set_log_enabled,
          "Enables or disables ONNX logging.")
      .def(
          "_jit_set_onnx_log_output_stream",
          [](const std::string& stream_name = "stdout") -> void {
            std::shared_ptr<std::ostream> out;
            if (stream_name == "stdout") {
              out = std::shared_ptr<std::ostream>(
                  &std::cout, [](std::ostream*) {});
            } else if (stream_name == "stderr") {
              out = std::shared_ptr<std::ostream>(
                  &std::cerr, [](std::ostream*) {});
            } else {
              std::cerr << "ERROR: only `stdout` and `stderr`"
                        << "are supported as `stream_name`" << std::endl;
            }
            ::torch::jit::onnx::set_log_output_stream(out);
          },
          "Set specific file stream for ONNX logging.")
      .def(
          "_jit_onnx_log",
          [](const py::args& args) -> void {
            if (::torch::jit::onnx::is_log_enabled()) {
              auto& out = ::torch::jit::onnx::_get_log_output_stream();
              for (auto arg : args) {
                out << ::c10::str(arg);
              }
              out << std::endl;
            }
          },
          "Write `args` to the previously specified ONNX log stream.")
      .def(
          "_jit_pass_onnx_assign_scoped_names_for_node_and_value",
          ::torch::wrap_pybind_function(
              ::torch::jit::onnx::AssignScopedNamesForNodeAndValue),
          "Assign informative scoped names for nodes and values.")
      .def(
          "_jit_onnx_create_full_scope_name",
          ::torch::wrap_pybind_function(
              ::torch::jit::onnx::ONNXScopeName::createFullScopeName),
          "Create a full scope name from class name and variable name.");

  m.def(
      "_check_onnx_proto",
      ::torch::wrap_pybind_function([](const std::string& proto_string) {
        check_onnx_proto(proto_string);
      }),
      py::arg("proto_string"));

  auto onnx = m.def_submodule("_onnx");
  py::enum_<::ONNX_NAMESPACE::TensorProto_DataType>(onnx, "TensorProtoDataType")
      .value("UNDEFINED", ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED)
      .value("FLOAT", ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      .value("UINT8", ::ONNX_NAMESPACE::TensorProto_DataType_UINT8)
      .value("INT8", ::ONNX_NAMESPACE::TensorProto_DataType_INT8)
      .value("UINT16", ::ONNX_NAMESPACE::TensorProto_DataType_UINT16)
      .value("INT16", ::ONNX_NAMESPACE::TensorProto_DataType_INT16)
      .value("INT32", ::ONNX_NAMESPACE::TensorProto_DataType_INT32)
      .value("INT64", ::ONNX_NAMESPACE::TensorProto_DataType_INT64)
      .value("STRING", ::ONNX_NAMESPACE::TensorProto_DataType_STRING)
      .value("BOOL", ::ONNX_NAMESPACE::TensorProto_DataType_BOOL)
      .value("FLOAT16", ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)
      .value("DOUBLE", ::ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)
      .value("UINT32", ::ONNX_NAMESPACE::TensorProto_DataType_UINT32)
      .value("UINT64", ::ONNX_NAMESPACE::TensorProto_DataType_UINT64)
      .value("COMPLEX64", ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64)
      .value("COMPLEX128", ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128)
      .value("BFLOAT16", ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)
      .value("FLOAT8E4M3FN", ::torch::onnx::TensorProto_DataType_FLOAT8E4M3FN)
      .value(
          "FLOAT8E4M3FNUZ", ::torch::onnx::TensorProto_DataType_FLOAT8E4M3FNUZ)
      .value("FLOAT8E5M2", ::torch::onnx::TensorProto_DataType_FLOAT8E5M2)
      .value(
          "FLOAT8E5M2FNUZ", ::torch::onnx::TensorProto_DataType_FLOAT8E5M2FNUZ);

  py::enum_<OperatorExportTypes>(onnx, "OperatorExportTypes")
      .value("ONNX", OperatorExportTypes::ONNX)
      .value("ONNX_ATEN", OperatorExportTypes::ONNX_ATEN)
      .value("ONNX_ATEN_FALLBACK", OperatorExportTypes::ONNX_ATEN_FALLBACK)
      .value("ONNX_FALLTHROUGH", OperatorExportTypes::ONNX_FALLTHROUGH);

  py::enum_<TrainingMode>(onnx, "TrainingMode")
      .value("EVAL", TrainingMode::EVAL)
      .value("PRESERVE", TrainingMode::PRESERVE)
      .value("TRAINING", TrainingMode::TRAINING);

  onnx.attr("PRODUCER_VERSION") = py::str(TORCH_VERSION);
}
} // namespace torch::onnx
