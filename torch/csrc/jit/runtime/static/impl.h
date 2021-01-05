#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>

namespace torch {
namespace jit {

struct TORCH_API InferenceModuleOptions {
  bool optimize_memory{true}; // TODO remove when logic moves to runtime
};

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
  explicit InferenceModule(const torch::jit::Module& m, InferenceModuleOptions);
  explicit InferenceModule(
      std::shared_ptr<torch::jit::Graph> g,
      InferenceModuleOptions);
  torch::jit::Module module;
  std::shared_ptr<torch::jit::Graph> graph;
  std::unique_ptr<c10::FunctionSchema> schema;

  std::unordered_map<Value*, size_t> value_to_reg;
  std::vector<Value*> values; // useful for debugging
  std::vector<size_t> input_regs; // inputs to the graph
  std::vector<size_t> output_regs; // outputs of the graph
  std::vector<size_t> internals;
  size_t reused_regs = 0;
  InferenceModuleOptions opts;

 private:
  void init();
};

TORCH_API void PrepareGraphForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g);

inline TORCH_API std::shared_ptr<InferenceModule> PrepareForStaticRuntime(
    const torch::jit::Module& m,
    InferenceModuleOptions opts = InferenceModuleOptions()) {
  return std::make_shared<InferenceModule>(m, opts);
}

inline TORCH_API std::shared_ptr<InferenceModule> PrepareForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g,
    InferenceModuleOptions opts = InferenceModuleOptions()) {
  return std::make_shared<InferenceModule>(g, opts);
}

class MemoryPlanner;
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

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inps);

  // This interface only works module_ that has a non-empty TorchScript module
  // member; otherwise use the above interface
  c10::IValue run(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  void benchmark(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs);

  float benchmark_model(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs);

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
      const int main_runs);

  const InferenceModule* get_inference_module() {
    return module_.get();
  }

  const std::vector<ProcessedNode>& get_nodes() {
    return nodes_;
  }

  const std::vector<IValue>& get_registers() {
    return reg_;
  }

  size_t num_outputs() const;

 private:
  // Static runtime states
  std::shared_ptr<InferenceModule> module_;
  StaticRuntimeOptions opts_;
  // IValue table (including inputs, outputs, intermediates, and weights)
  std::vector<IValue> reg_;
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;

  // Memory planning is only enabled if opts_.cleanup_activations is true.
  // Otherwise, the memory used by activations is cached inside the static
  // runtime.
  std::unique_ptr<MemoryPlanner> planner_;
  // The memory planner does not take care of the IValues that are stored in the
  // registers but don't participate in memory planning. They need to be cleaned
  // up if cleanup_activations is enabled to avoid memory leak.
  void deallocate_registers(const std::vector<size_t>& internals);

  // Input is readwrite
  IValue& Input(size_t i) {
    DCHECK(i < module_->input_regs.size());
    return reg_[module_->input_regs[i]];
  }

  // Output is readonly. The writing process happens inside ProcessedNodes
  const IValue& Output(size_t i) const {
    DCHECK(i < module_->output_regs.size());
    return reg_[module_->output_regs[i]];
  }
};

/// There are three types of ops in a processed graph in Static Runtime:
///   1. op with _out variant
///   2. view producing op
///   3. tensor producing op (could be replaced with 1 by adding the _out
///      variant to Static Runtime)
/// The memory planner only manages tensors that are outputs of type 1 ops,
/// because type 2 ops don't incur memory allocation and for type 3, the output
/// tensors are allocated inside the operator and can't be directly managed by
/// memory planner.
///
/// Memory planner tries to minimize the number of memory allocations by
/// tracking the unique StorageImpls of the output tensors of ops with _out
/// variants. It tries to do this in several steps:
///   1. record the max memory usage for each StorageImpl at the end of each
///      iteration
///   2. in the next iteration, allocate the buffer for the max total usage and
///      compute the offset of each allocation with regard to the single memory
///      buffer. In the first iteration, we rely on the default allocator for
///      memory allocation.
///   3. free the buffer at the end of each iteration
/// Steps 1 and 3 are handled by `deallocate()`, and step 2 by `allocate()`.
/// Only models with simple output types are supported, i.e. None, Tensor or
/// List/Tuple of Tensors. Complex output types such as List of Lists are not
/// supported.

class MemoryPlanner {
 public:
  explicit MemoryPlanner(StaticRuntime* runtime);

  void allocate();
  void deallocate();
  size_t total_managed() const {
    return internal_blob_max_sizes_sum_;
  }

 private:
  const std::vector<IValue>& reg_;
  std::unordered_set<size_t> reg_out_variant_;
  std::vector<c10::StorageImpl*> internal_storages_;
  std::vector<size_t> internal_blob_max_sizes_;
  size_t internal_blob_max_sizes_sum_{0};
  at::DataPtr buffer_; // allocated each time we call Run()

  static size_t compute_aligned_tensor_size(size_t nbytes);
  static at::DataPtr allocate_buffer(size_t size);
  std::unordered_set<c10::StorageImpl*> reg_to_storage_impls();
  void verify_internal_storages();
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

  bool has_out_variant() const {
    return static_cast<bool>(fn_);
  }

  const std::vector<size_t>& input_regs() const {
    return input_regs_;
  }

  const std::vector<size_t>& output_regs() const {
    return output_regs_;
  }

  const TypePtr& get_output_type(size_t i) const {
    DCHECK_LT(i, output_regs().size());
    return node_->outputs()[i]->type();
  }

 private:
  Node* node_;
  c10::optional<Operation> op_;
  std::function<void(const ProcessedNode*, std::vector<IValue>&)> fn_;
  std::function<void(const ProcessedNode*, std::vector<IValue>&)> native_fn_;

  std::vector<size_t> input_regs_;
  std::vector<size_t> output_regs_;
};

} // namespace jit
} // namespace torch
