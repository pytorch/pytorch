#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

using RawDataExportMap = std::unordered_map<std::string, std::string>;

std::tuple<std::string, RawDataExportMap> ExportGraph(
    const std::shared_ptr<Graph>& graph,
    const std::vector<at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export = false);

}}
