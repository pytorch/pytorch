#pragma once

#include <ATen/Config.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>

#endif // AT_MKLDNN_ENABLED()

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()

namespace mkldnn {

struct PostOp {
  std::vector<std::string> scalar_input;
  std::string algorithm_input = "";
  std::vector<torch::jit::MatchFilter> filters = {};
};

} // namespace mkldnn

#endif // AT_MKLDNN_ENABLED()

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
