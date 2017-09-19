#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// Exports a graph to ONNX
std::string ExportGraph(std::shared_ptr<Graph>& graph,
                        const std::unordered_map<void*, Node*>& buffer_map,
                        const std::vector<std::string> & input_names,
                        const std::vector<std::string> & output_names,
                        bool verbose,
                        const std::vector<at::Tensor> & initializers);

}}
