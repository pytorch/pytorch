#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#endif

namespace torch {
namespace jit {

class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(std::shared_ptr<torch::jit::Graph> g)
      : graph_(std::move(g)) {}

  explicit StaticRuntime(const torch::jit::Module& m);

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inps) const;

#ifdef FBCODE_CAFFE2
  using ConstantMap = folly::F14FastMap<Value*, IValue>;
#else
  using ConstantMap = std::unordered_map<Value*, IValue>;
#endif

 private:
  torch::jit::Module module_;
  std::shared_ptr<torch::jit::Graph> graph_;

  // Static runtime states
  // Constant table (including weights)
  ConstantMap constant_table_;
  // The nodes we need to run
  std::vector<std::pair<Node*, Operation>> nodes_;

  void getInputIValues(
      Node* node,
      const ConstantMap& ws,
      std::vector<IValue>& stack) const;

  void runNodes(ConstantMap& ws_) const;
};

} // namespace jit
} // namespace torch
