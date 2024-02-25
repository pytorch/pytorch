#pragma once

#include <torch/csrc/onnx/onnx.h>

#include <ostream>

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace torch {
namespace jit {

// This map is used to keep track of parameters that should be exported
// externally. When `defer_weight_export` is true, the returned map contains
// kv pairs that map {external reference name} -> {at::Tensor to be exported}.
// It is the responsibility of the caller to export these appropriately.
//
// For example, when exporting to a zip archive, the caller may write out files
// for each entry in the export map, with the filename being the key and the
// file contents being the raw tensor data.
using RawDataExportMap = std::unordered_map<std::string, at::Tensor>;

using SymbolDimMap = std::map<c10::ShapeSymbol, std::string>;

using NodeNameMap = std::unordered_map<const Node*, std::string>;

// Used for modularized export settling function and node attributes.
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

TORCH_API std::tuple<
    std::shared_ptr<::ONNX_NAMESPACE::ModelProto>,
    RawDataExportMap,
    SymbolDimMap,
    bool,
    NodeNameMap>
export_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    bool defer_weight_export = false,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool strip_doc_string = true,
    bool keep_initializers_as_inputs = true,
    const std::map<std::string, int>& custom_opsets = {},
    bool add_node_names = true,
    bool use_external_data_format = false,
    const std::string& onnx_file_path = std::string(),
    const NodeAttrNameMap& node_attr_to_name = {});

TORCH_API std::string serialize_model_proto_to_string(
    const std::shared_ptr<::ONNX_NAMESPACE::ModelProto>& model_proto);

TORCH_API void check_onnx_proto(const std::string& proto_string);

// For testing purposes
TORCH_API std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool google_printer = false,
    bool keep_initializers_as_inputs = true,
    const std::map<std::string, int>& custom_opsets = {},
    bool add_node_names = true);

} // namespace jit
} // namespace torch