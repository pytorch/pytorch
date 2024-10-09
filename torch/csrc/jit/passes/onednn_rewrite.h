#pragma once

#include <ATen/Config.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#if AT_ONEDNN_ENABLED()

#include <ideep/tensor.hpp>

#endif // AT_ONEDNN_ENABLED()

namespace torch::jit {

#if AT_ONEDNN_ENABLED()

namespace onednn {

const static std::map<std::string, std::vector<torch::jit::MatchFilter>>
    fusion_rewrite_map = {
        {"none", {}},
        {"relu", {}},
};

} // namespace onednn

#endif // AT_ONEDNN_ENABLED()

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
