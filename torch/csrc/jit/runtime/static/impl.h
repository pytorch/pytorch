#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>

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

  void benchmark(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs) const;

  float benchmark_model(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs) const;

  struct IndividualMetrics {
    float setup_time;
    float total_time;
    std::vector<float> time_per_node;
    std::unordered_map<std::string, float> time_per_node_type;
    std::unordered_map<std::string, float> percent_per_node_type;
    std::unordered_map<std::string, int> instances_per_node_type;
  };

  IndividualMetrics benchmark_individual_ops(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs) const;

 private:
  explicit StaticRuntime(
      std::shared_ptr<torch::jit::Graph> g, // optimized graph
      c10::optional<torch::jit::Module> m);

  std::shared_ptr<torch::jit::Graph> graph_;

  std::unique_ptr<c10::FunctionSchema> schema_{nullptr};

  // Static runtime states
  // IValue table (including inputs, outputs, intermediates, and weights)
  mutable std::vector<IValue> reg_;
  std::vector<size_t> input_regs_; // inputs to the graph
  std::vector<size_t> output_regs_; // outputs of the graph

  // Input is readwrite
  IValue& Input(size_t i) const {
    DCHECK(i < input_regs_.size());
    return reg_[input_regs_[i]];
  }

  // Output is readonly. The writing process happens inside ProcessedNodes
  const IValue& Output(size_t i) const {
    DCHECK(i < output_regs_.size());
    return reg_[output_regs_[i]];
  }

  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
};

class ProcessedNode {
 public:
  ProcessedNode(
      Node* n,
      std::vector<size_t>&& input_regs,
      std::vector<size_t>&& output_regs);
  void run(std::vector<IValue>& reg) const;

  Node* get_node() const {
    return node_;
  }

  // Input is readonly
  const IValue& Input(size_t i, std::vector<IValue>& reg) const {
    DCHECK(i < input_regs_.size());
    return reg[input_regs_[i]];
  }

  // Output is readwrite
  IValue& Output(size_t i, std::vector<IValue>& reg) const {
    DCHECK(i < output_regs_.size());
    return reg[output_regs_[i]];
  }

 private:
  Node* node_;
  c10::optional<Operation> op_;
  c10::optional<std::function<void(const ProcessedNode*, std::vector<IValue>&)>>
      fn_;

  std::vector<size_t> input_regs_;
  std::vector<size_t> output_regs_;
};

} // namespace jit
} // namespace torch
