#pragma once

#include <ATen/Config.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>

#endif // AT_MKLDNN_ENABLED()

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()

namespace mkldnn {

struct PostOp {
  ideep::attr_t op_attr;
  std::vector<torch::jit::MatchFilter> filters = {};
};

const static std::map<std::string, PostOp> fusion_attr_map = {
    {"none", {ideep::attr_t()}},
    {"relu", {ideep::attr_t::fuse_relu()}},
};

} // namespace mkldnn

#endif // AT_MKLDNN_ENABLED()

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
