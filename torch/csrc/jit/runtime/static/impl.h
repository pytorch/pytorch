#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/fuser/arg_spec.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#endif

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g);
TORCH_API std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    const torch::jit::Module& m);

class ProcessedNode;
class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(std::shared_ptr<torch::jit::Graph> g);
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inps) const;

#ifdef FBCODE_CAFFE2
  using ConstantMap = folly::F14FastMap<Value*, IValue>;
#else
  using ConstantMap = std::unordered_map<Value*, IValue>;
#endif

 private:
  std::shared_ptr<torch::jit::Graph> graph_;

  // Static runtime states
  // Value table (including weights)
  mutable ConstantMap workspace_;

  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
};

class ProcessedNode {
 public:
  ProcessedNode(Node* n);
  void run(StaticRuntime::ConstantMap& workspace) const;
  Node* get_node() const {
    return node_;
  }

 private:
  Node* node_;
  c10::optional<Operation> op_;
  c10::optional<std::function<void(StaticRuntime::ConstantMap&)>> fn_;
  mutable std::unordered_map<uint64_t, Operation> compiled_;
};

} // namespace jit
} // namespace torch
