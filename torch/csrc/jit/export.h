#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

std::string ExportGraph(const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version);

}}
