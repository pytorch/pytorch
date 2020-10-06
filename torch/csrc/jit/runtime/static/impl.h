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

TORCH_API std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g);
TORCH_API std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    const torch::jit::Module& m);

class ProcessedNode;
class TORCH_API StaticRuntime {
 public:
  // g is the optimized graph produced by PrepareForStaticRuntime
  explicit StaticRuntime(std::shared_ptr<torch::jit::Graph> g);

  // m is unoptimized
  explicit StaticRuntime(const torch::jit::Module& m);

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inps) const;

  c10::IValue run(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs) const;

#ifdef FBCODE_CAFFE2
  using ConstantMap = folly::F14FastMap<Value*, IValue>;
#else
  using ConstantMap = std::unordered_map<Value*, IValue>;
#endif

 private:
  explicit StaticRuntime(
      std::shared_ptr<torch::jit::Graph> g, // optimized graph
      c10::optional<torch::jit::Module> m);

  std::shared_ptr<torch::jit::Graph> graph_;

  std::unique_ptr<c10::FunctionSchema> schema_{nullptr};

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
};

} // namespace jit
} // namespace torch
