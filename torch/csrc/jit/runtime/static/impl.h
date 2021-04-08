#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>

namespace torch {
namespace jit {

struct TORCH_API StaticModuleOptions {
  bool cleanup_activations{true};
  bool enable_out_variant{true};
  bool optimize_memory{true};
  // to enable MemoryPlanner on output tensors
  bool optimize_output_memory{false};
};

/// The static runime supports two execution modes.
///
/// Mode 1: single-threaded with no parallelism except for intra-op parallelism
/// For this mode, you can do either:
/// @code
///   // m is a TorchScript module
///   auto module = StaticModule(m, opts);
///   auto output = module(args, kwargs);
/// @endcode
///
/// or
///
/// @code
///   // g is the TorchScript graph
///   auto module = StaticModule(g, opts);
///   auto output = module(args, kwargs);
/// @endcode
///
/// Mode 2: similar to data parallelism, run the same model for different inputs
/// on different threads at the same time.
/// You should have one StaticModule per model, and one StaticRuntime instance
/// per running thread. To avoiding creating StaticRuntimes on the fly, use a
/// synchronized stack (i.e. boost::lockfree::stack) to cache all the
/// StaticRuntime instances in your code.
/// @code
///   // initialization
///   auto module = std::make_shared<StaticModule>(m, opts);
///
///   // 128 is good for most cases. Pick a number that works for you
///   boost::lockfree::stack<std::shared_ptr<StaticRuntime>,
///     boost::lockfree::fixed_sized<true>> pool(128);
///
///   // inference
///   std::shared_ptr<StaticRuntime> runtime = nullptr;
///   pool.pop(runtime);
///   if (!runtime) {
///     // holds a reference to the underlying module
///     // but does its own memory management
///     runtime = std::make_shared<StaticRuntime>(*module);
///   }
///   auto output = runtime(args, kwargs);
///   pool.push(runtime);
/// @endcode
///

class MemoryPlanner;
class ProcessedNode;
class StaticRuntime;
class TORCH_API StaticModule {
 public:
  explicit StaticModule(
      std::shared_ptr<torch::jit::Graph> g,
      const StaticModuleOptions& opts = StaticModuleOptions());

  explicit StaticModule(
      const torch::jit::Module& m,
      const StaticModuleOptions& opts = StaticModuleOptions());

  typedef enum {
    CONSTANT_VALUE = -2, // VALUE nodes defined by prim::Constant
    INPUT_VALUE = -1, // VALUE nodes representing graph inputs
  } VALUE_KIND;

 private:
  explicit StaticModule(
      std::pair<
          std::shared_ptr<torch::jit::Graph>,
          c10::optional<c10::FunctionSchema>> graph_and_schema,
      const StaticModuleOptions& opts);

  // for <kind, idx>
  //   if kind == CONSTANT_KIND: map to constants_[idx]
  //   if kind == INPUT_KIND: map to inputs_[idx]
  //   otherwise: map to nodes_[kind].outputs()[idx]
  using DefInfo = std::pair<int, int>;

 public:
  std::vector<at::Tensor> operator()(const std::vector<at::Tensor>& inps);

  // This interface only works if StaticModule was initialized
  // with a TorchScript module, otherwise use the above interface
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  const Graph& graph() const {
    return *graph_;
  }

  const StaticModuleOptions& opts() const;
  size_t num_inputs() const;
  size_t num_outputs() const;

  inline const std::unordered_map<int, std::vector<DefInfo>>& index_map()
      const {
    return node_inputs_ssa_def_map_;
  }

  inline const std::vector<DefInfo>& output_indices() const {
    return output_ssa_defs_;
  }

  inline const std::vector<IValue>& constants() const {
    return constants_;
  }

  inline const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

  inline const c10::optional<c10::FunctionSchema>& schema() const {
    return schema_;
  }

  inline const std::unordered_map<const Value*, std::vector<const Value*>>&
  values_share_same_storage() const {
    return value_to_same_storage_values_;
  }

  inline const std::unordered_set<const Value*>& external_values() const {
    return external_values_;
  }

  StaticRuntime& runtime();

 private:
  // Static runtime states
  StaticModuleOptions opts_;
  std::unique_ptr<StaticRuntime> cached_runtime_;
  // IValue table (defined by prim::Constant nodes)
  std::vector<IValue> constants_;
  // a vector of ssa_defs corresponding to graph->outputs()
  std::vector<DefInfo> output_ssa_defs_;
  // map a node idx (in graph order) to a vector of ssa_defs for node inputs
  std::unordered_map<int, std::vector<DefInfo>> node_inputs_ssa_def_map_;
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
  // map a value to the set of values that may share the same storage with it
  std::unordered_map<const Value*, std::vector<const Value*>>
      value_to_same_storage_values_;
  // values whose live-time exceeds that of running one inference (e.g., input,
  // output, prim::Constants, and their aliases)
  std::unordered_set<const Value*> external_values_;

  // Original input
  std::shared_ptr<torch::jit::Graph> graph_;
  c10::optional<c10::FunctionSchema> schema_;
};

class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(const StaticModule& sm);

  std::vector<at::Tensor> operator()(const std::vector<at::Tensor>& inps);

  // This interface only works if StaticModule was initialized
  // with a TorchScript module, otherwise use the above interface
  c10::IValue operator()(
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
    float setup_time{0.0};
    float memory_alloc_time{0.0};
    float memory_dealloc_time{0.0};
    float output_dealloc_time{0.0};
    float total_time{0.0};
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

  // Input is readwrite
  IValue& Input(size_t i) {
    DCHECK(i < inputs_.size());
    return inputs_[i];
  }

  // Output is readonly. The writing process happens inside ProcessedNodes
  const IValue& Output(size_t i) const {
    DCHECK(i < outputs_.size());
    return *outputs_[i];
  }

  const std::vector<IValue*> outputs() const {
    return outputs_;
  }

  inline const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

  inline std::vector<ProcessedNode>& nodes() {
    return nodes_;
  }

  const Graph& graph() const {
    return static_module_.graph();
  }

  void check_for_memory_leak(bool output_returned = true);

 private:
  // Memory planning is only enabled if sm->opts().cleanup_activations is true.
  // Otherwise, the memory used by activations is cached inside the static
  // runtime.
  std::unique_ptr<MemoryPlanner> planner_;
  std::vector<IValue> inputs_;
  std::vector<IValue*> outputs_;
  const StaticModule& static_module_;
  std::vector<ProcessedNode> nodes_;
};

/// There are three types of ops in a processed graph in Static Runtime:
///   1. op with _out variant
///   2. view producing op
///   3. tensor producing op (could be replaced with type 1 by adding the _out
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
///      buffer, optionally reusing memory.  In the first iteration, we rely on
///      the default allocator for memory allocation.
///   3. free the buffer at the end of each iteration
/// Steps 1 and 3 are handled by `deallocate()`, and step 2 by `allocate()`.
/// Only models with simple output types are supported, i.e. None, Tensor or
/// List/Tuple of Tensors. Complex output types such as List of Lists are not
/// supported.

class MemoryPlanner {
 public:
  explicit MemoryPlanner(
      StaticRuntime* runtime,
      const std::unordered_map<const Value*, std::vector<const Value*>>&,
      const std::unordered_set<const Value*>& external_values,
      bool out_variants);

  void allocate();
  void deallocate();
  size_t total_managed() const {
    return managed_bytes_;
  }
  size_t total_reused_tensors() const {
    return reused_tensors_;
  }

 private:
  std::vector<IValue*> unmanaged_ivalues_;
  // each pair contains the size (in bytes) of data to be allocated
  // and a vector of StorageImpl's that should be backed by that same data
  // Thus, if memonger is disabled, all vectors are of size 1.
  std::vector<std::pair<size_t, std::vector<c10::StorageImpl*>>>
      managed_storage_;
  size_t managed_bytes_{0};
  size_t reused_tensors_{0};
  at::DataPtr buffer_; // allocated each time we call Run()

  static size_t compute_aligned_tensor_size(size_t nbytes);
  static at::DataPtr allocate_buffer(size_t size);
};

class ProcessedNode {
 public:
  ProcessedNode() = default;
  ProcessedNode(
      Node* n,
      std::vector<const IValue*>&& inputs,
      bool enable_out_variant);

  void run();

  Node* node() const {
    return node_;
  }

  inline void set_input(size_t index, const IValue* ival) {
    inputs_[index] = ival;
  }

  // Input is readonly
  const IValue& Input(size_t i) const {
    DCHECK(i < inputs_.size());
    return *inputs_[i];
  }

  // Output is readwrite
  IValue& Output(size_t i) {
    DCHECK(i < outputs_.size());
    return outputs_[i];
  }

  const std::vector<IValue>& outputs() const {
    return outputs_;
  }

  const std::vector<const IValue*>& inputs() const {
    return inputs_;
  }

  bool has_out_variant() const {
    return static_cast<bool>(fn_);
  }

 private:
  Node* node_;
  c10::optional<Operation> op_;
  std::function<void(ProcessedNode*)> fn_;
  std::function<void(ProcessedNode*)> native_fn_;
  std::vector<const IValue*> inputs_; // unowned
  std::vector<IValue> outputs_;
};

} // namespace jit
} // namespace torch
