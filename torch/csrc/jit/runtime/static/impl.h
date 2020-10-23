#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>

namespace torch {
namespace jit {

struct TORCH_API StaticRuntimeOptions {
  bool cleanup_activations{true};
  bool enable_out_variant{true};
};

/// Static runime supports two execution modes.
///
/// Mode 1: single-threaded with no parallelism except for intra-op parallelism
/// For this mode, you can do either:
/// @code
///   // m is the TorchScript module
///   auto runtime = StaticRuntime(m, opts);
///   auto output = runtime.run(args, kwargs);
/// @endcode
/// or
/// @code
///   auto mod = PrepareForStaticRuntime(m);
///   auto runtime = StaticRuntime(mod, opts);
///   auto output = runtime.run(args, kwargs);
/// @endcode
/// Mode 2: similar to data parallelism, run the same model for different inputs
/// on different threads at the same time. In this case, run
/// PrepareForStaticRuntime to prepare the graph for Static Runtime. You
/// should have one InferenceModule instance per model, and one Static Runtime
/// instance per running thread. To avoiding creating StaticRuntime on the fly,
/// use a synchronized stack (i.e. boost::lockfree::stack) to cache all the
/// Static Runtime instances in your code.
/// @code
///   // initialization
///   auto mod = PrepareForStaticRuntime(m);
///   // 128 is good for most cases. Pick a number that works for you
///   boost::lockfree::stack<std::shared_ptr<StaticRuntime>,
///     boost::lockfree::fixed_sized<true>> pool(128);
///
///   // inference
///   std::shared_ptr<StaticRuntime> runtime = nullptr;
///   pool.pop(runtime);
///   if (!runtime) {
///     runtime = std::make_shared<StaticRuntime>(mod, opts);
///   }
///   auto output = runtime->run(args, kwargs);
///   pool.push(runtime);
/// @endcode
///

// Group readonly data structures into InferenceModule
struct TORCH_API InferenceModule {
 public:
  explicit InferenceModule(const torch::jit::Module& m);
  explicit InferenceModule(std::shared_ptr<torch::jit::Graph> g);
  torch::jit::Module module;
  std::shared_ptr<torch::jit::Graph> graph;
  std::unique_ptr<c10::FunctionSchema> schema;

  std::unordered_map<Value*, size_t> value_to_reg;
  std::vector<size_t> input_regs; // inputs to the graph
  std::vector<size_t> output_regs; // outputs of the graph
  std::vector<size_t> internals;

 private:
  void init();
};

inline TORCH_API std::shared_ptr<InferenceModule> PrepareForStaticRuntime(
    const torch::jit::Module& m) {
  return std::make_shared<InferenceModule>(m);
}

inline TORCH_API std::shared_ptr<InferenceModule> PrepareForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g) {
  return std::make_shared<InferenceModule>(g);
}

class ProcessedNode;
class TORCH_API StaticRuntime {
 public:
  // InferenceModule m is created by PrepareForStaticRuntime
  explicit StaticRuntime(
      std::shared_ptr<InferenceModule> m,
      const StaticRuntimeOptions& opts = StaticRuntimeOptions());

  // m is unoptimized
  explicit StaticRuntime(
      const torch::jit::Module& m,
      const StaticRuntimeOptions& opts = StaticRuntimeOptions());

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inps) const;

  // This interface only works module_ that has a non-empty TorchScript module
  // member; otherwise use the above interface
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
  // Static runtime states
  std::shared_ptr<InferenceModule> module_;
  StaticRuntimeOptions opts_;
  // IValue table (including inputs, outputs, intermediates, and weights)
  mutable std::vector<IValue> reg_;
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;

  // Input is readwrite
  IValue& Input(size_t i) const {
    DCHECK(i < module_->input_regs.size());
    return reg_[module_->input_regs[i]];
  }

  // Output is readonly. The writing process happens inside ProcessedNodes
  const IValue& Output(size_t i) const {
    DCHECK(i < module_->output_regs.size());
    return reg_[module_->output_regs[i]];
  }
};

class ProcessedNode {
 public:
  ProcessedNode(
      Node* n,
      std::vector<size_t>&& input_regs,
      std::vector<size_t>&& output_regs,
      bool enable_out_variant);
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
