#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/onnx/onnx.h>

#include <ostream>

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

TORCH_API std::tuple<std::string, RawDataExportMap> export_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::vector<at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export = false,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX);

// For testing purposes
TORCH_API std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::vector<at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool google_printer = false);

TORCH_API void ExportModule(
    const script::Module& module,
    std::ostream& out,
    const script::ExtraFilesMap& metadata = script::ExtraFilesMap());

TORCH_API void ExportModule(
    const script::Module& module,
    const std::string& filename,
    const script::ExtraFilesMap& metadata = script::ExtraFilesMap());

} // namespace jit
} // namespace torch
