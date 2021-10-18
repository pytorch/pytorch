#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/variant.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#endif

namespace torch {
namespace jit {

#ifdef FBCODE_CAFFE2
template <typename Key, typename Value>
using FastMap = folly::F14FastMap<Key, Value>;
template <typename Key>
using FastSet = folly::F14FastSet<Key>;
#else
template <typename Key, typename Value>
using FastMap = std::unordered_map<Key, Value>;
template <typename Key>
using FastSet = std::unordered_set<Key>;
#endif

TORCH_API bool canEnableStaticRuntime(
    const std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name = "");

// Group values used by `graph` into three categories:
//
// - output_aliases:
//     values that are either outputs or contain aliases of outputs
// - external_aliases:
//     values that are inputs, constants, outputs, or their aliases.
//     The output aliases that end up here are as a result of aliasDb failing to
//     recognize them as outputs due to collection object (e.g., Tuple) aliasing
//     inputs.
// Values that dont't show up in output_aliases or external_aliases are created
// and consumed within the graph.
class ValueGroup {
 public:
  explicit ValueGroup() = default;
  void init(const std::shared_ptr<torch::jit::Graph>& graph, AliasDb& db);

  bool isExternalAlias(const Value* value) const {
    return external_aliases_.find(value) != external_aliases_.end();
  }

  bool isOutputAlias(const Value* value) const {
    return output_aliases_.find(value) != output_aliases_.end();
  }

  bool isAlwaysAlive(const Value* value) const {
    return isExternalAlias(value) || isOutputAlias(value);
  }

  std::string toString() const {
    return c10::str(
        dumpValueSet(output_aliases_, "ValueGroup::output_aliases_"),
        "\n",
        dumpValueSet(external_aliases_, "ValueGroup::external_aliases_"));
  }

 private:
  FastSet<const Value*> output_aliases_;
  FastSet<const Value*> external_aliases_;
};

struct TORCH_API StaticModuleOptions {
  // to batch allocate (deallocate) tensor storage for all non-escaping
  // temporary tensors
  bool cleanup_activations{true};
  // enabling out variant allows Static Runtime to do memory planning
  bool enable_out_variant{true};
  // to reuse tensor storage for tensors whose live-range do not overlap to
  // reduce memory footprint (enable_out_variant must be true)
  bool optimize_memory{true};
  // to batch allocate tensor storage for output tensors of the
  // graph, where storage is deallocated outside static runtime
  // (enable_out_variant must be true)
  bool manage_output_tensors{false};
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
      bool is_frozen = false,
      const StaticModuleOptions& opts = StaticModuleOptions());

  typedef enum {
    CONSTANT_VALUE = -2, // VALUE nodes defined by prim::Constant
    INPUT_VALUE = -1, // VALUE nodes representing graph inputs
  } VALUE_KIND;

 private:
  explicit StaticModule(
      std::pair<std::shared_ptr<torch::jit::Graph>, c10::optional<Module>>
          graph_and_module,
      const StaticModuleOptions& opts);

  // for <kind, idx>
  //   if kind == CONSTANT_VALUE: map to constants_[idx]
  //   if kind == INPUT_VALUE: map to inputs_[idx]
  //   otherwise: map to nodes_[kind].outputs()[idx]
  using DefInfo = std::pair<int, int>;

 public:
  std::vector<at::Tensor> operator()(const std::vector<at::Tensor>& inps);

  // This interface only works if StaticModule was initialized
  // with a TorchScript module, otherwise use the above interface
  c10::IValue operator()(
      c10::ArrayRef<c10::IValue> args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  const Graph& graph() const {
    return *graph_;
  }

  const Module& module() const {
    return *module_;
  }

  const StaticModuleOptions& opts() const;

  const ValueGroup& valueGroup() const {
    return value_group_;
  }

  size_t num_inputs() const;
  size_t num_outputs() const;

  const FastMap<int, std::vector<DefInfo>>& index_map() const {
    return node_inputs_ssa_def_map_;
  }

  const std::vector<DefInfo>& output_indices() const {
    return output_ssa_defs_;
  }

  const std::vector<IValue>& constants() const {
    return constants_;
  }

  const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

  bool is_optimizable_container_type(const Node* n) const {
    auto it = node_is_optimizable_container_type_.find(n);
    return it != node_is_optimizable_container_type_.end();
  }

  const c10::optional<c10::FunctionSchema>& schema() const {
    return schema_;
  }

  const FastMap<const Value*, std::vector<const Value*>>&
  values_share_same_storage() const {
    return value_to_same_storage_values_;
  }

  const ValueGroup& value_group() const {
    return value_group_;
  }

  bool first_input_is_self() const {
    return first_input_is_self_;
  }

  StaticRuntime& runtime();

 private:
  StaticModuleOptions opts_;
  bool first_input_is_self_{false};
  std::shared_ptr<torch::jit::Graph> graph_;
  c10::optional<torch::jit::Module> module_;
  c10::optional<c10::FunctionSchema> schema_;
  std::unique_ptr<StaticRuntime> cached_runtime_;

  // Bookkeeping for creating new StaticRuntime instances
  // IValue table (defined by prim::Constant nodes)
  std::vector<IValue> constants_;
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
  // a vector of ssa_defs corresponding to graph->outputs()
  std::vector<DefInfo> output_ssa_defs_;
  // map a node idx (in graph order) to a vector of ssa_defs for node inputs
  FastMap<int, std::vector<DefInfo>> node_inputs_ssa_def_map_;

  ValueGroup value_group_;

  // map a value to the set of values that may share the same storage with it
  FastMap<const Value*, std::vector<const Value*>>
      value_to_same_storage_values_;

  FastSet<const Node*> node_is_optimizable_container_type_;
};

class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(const StaticModule& sm);
  StaticRuntime(StaticRuntime&&) = delete;
  StaticRuntime& operator=(StaticRuntime&&) = delete;
  ~StaticRuntime();

  C10_DISABLE_COPY_AND_ASSIGN(StaticRuntime);

  std::vector<at::Tensor> operator()(const std::vector<at::Tensor>& inps);

  // This interface only works if StaticModule was initialized
  // with a TorchScript module, otherwise use the above interface
  c10::IValue operator()(
      c10::ArrayRef<c10::IValue> args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  void display_nodes(
      c10::ArrayRef<c10::IValue> args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  void benchmark(
      c10::ArrayRef<c10::IValue> args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs,
      bool print_per_node_time = false,
      bool generate_ai_pep_output = false);

  float benchmark_model(
      c10::ArrayRef<c10::IValue> args,
      const std::unordered_map<std::string, c10::IValue>& kwargs,
      const int warmup_runs,
      const int main_runs);

  struct IndividualMetrics {
    float setup_time{0.0};
    float memory_alloc_time{0.0};
    float memory_dealloc_time{0.0};
    float output_dealloc_time{0.0};
    float first_iter_time{0.0};
    float total_time{0.0};
    size_t out_nodes_count{0};
    size_t total_nodes_count{0};
    std::vector<float> time_per_node;
    std::unordered_map<std::string, float> time_per_node_type;
    std::unordered_map<std::string, float> percent_per_node_type;
    std::unordered_map<std::string, int> instances_per_node_type;
    std::unordered_set<std::string> out_nodes;
    std::unordered_set<std::string> native_nodes;
  };

  IndividualMetrics benchmark_individual_ops(
      c10::ArrayRef<c10::IValue> args,
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

  const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

  std::vector<ProcessedNode>& nodes() {
    return nodes_;
  }

  const Graph& graph() const {
    return static_module_.graph();
  }

  void check_for_memory_leak(bool output_returned = true);

  bool is_optimizable_container_type(Node* n) const {
    return static_module_.is_optimizable_container_type(n);
  }

  // Deallocate managed output tensors. This should be called only when all the
  // references to the output from Static Runtime are gone.
  void deallocateOutputTensors();

  bool checkOutputTensorMemoryLeaks();

  bool isManagedOutputTensor(const IValue& ivalue);

 private:
  // helper method for copying input args/kwargs into inputs_
  void set_inputs(
      c10::ArrayRef<c10::IValue> args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  // clean up owning refs of input IValues
  void clean_up_input_ivalues() {
    for (IValue& ival : inputs_) {
      ival = IValue();
    }
  }

  // Memory planning is only enabled if sm->opts().cleanup_activations is true.
  // Otherwise, the memory used by activations is cached inside the static
  // runtime.
  const StaticModule& static_module_;
  std::unique_ptr<MemoryPlanner> planner_;
  std::vector<IValue> inputs_;
  std::vector<IValue*> outputs_;
  std::vector<ProcessedNode> nodes_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ProcessedNode() = default;
  ProcessedNode(
      Node* n,
      std::unique_ptr<const IValue*[]> inputs,
      size_t inputsSize,
      bool enable_out_variant);

  ProcessedNode(const ProcessedNode& rhs)
      : node_(rhs.node_),
        fn_(rhs.fn_),
        inputs_(std::make_unique<const IValue*[]>(rhs.inputs_size_)),
        outputs_(std::make_unique<IValue[]>(rhs.outputs_size_)),
        inputs_size_(rhs.inputs_size_),
        outputs_size_(rhs.outputs_size_),
        op_name_(rhs.op_name_) {
    std::copy(
        rhs.inputs_.get(), rhs.inputs_.get() + inputs_size_, inputs_.get());
    std::copy(
        rhs.outputs_.get(), rhs.outputs_.get() + outputs_size_, outputs_.get());
  }

  ProcessedNode& operator=(const ProcessedNode& rhs) {
    if (this == &rhs) {
      return *this;
    }
    node_ = rhs.node_;
    fn_ = rhs.fn_;
    if (!inputs_ || inputs_size_ != rhs.inputs_size_) {
      inputs_ = std::make_unique<const IValue*[]>(rhs.inputs_size_);
      inputs_size_ = rhs.inputs_size_;
    }
    std::copy(
        rhs.inputs_.get(), rhs.inputs_.get() + inputs_size_, inputs_.get());

    if (!outputs_ || outputs_size_ != rhs.outputs_size_) {
      outputs_ = std::make_unique<IValue[]>(rhs.outputs_size_);
      outputs_size_ = rhs.outputs_size_;
    }
    std::copy(
        rhs.outputs_.get(), rhs.outputs_.get() + outputs_size_, outputs_.get());
    op_name_ = rhs.op_name_;

    return *this;
  }

  // These should be noexcept, but some Android build is failing
  // saying the noexcept specification doesn't match the calculated
  // one. Maybe c10::variant is throwing it off?
  ProcessedNode(ProcessedNode&&) = default;
  ProcessedNode& operator=(ProcessedNode&&) = default;

  void run();

  Node* node() const {
    return node_;
  }

  // Input is readonly
  const IValue& Input(size_t i) const {
    DCHECK(i < inputs_size_);
    return *inputs_[i];
  }

  // Output is readwrite
  IValue& Output(size_t i) {
    DCHECK(i < outputs_size_);
    return outputs_[i];
  }

  void set_input(size_t index, const IValue* ival) {
    inputs_[index] = ival;
  }

  C10_NODISCARD c10::ArrayRef<const IValue> outputs() const {
    return c10::ArrayRef<const IValue>(outputs_.get(), outputs_size_);
  }

  C10_NODISCARD c10::ArrayRef<const IValue*> inputs() const {
    return c10::ArrayRef<const IValue*>(inputs_.get(), inputs_size_);
  }

  std::vector<IValue> clone_inputs() const;

  bool has_out_variant() const {
    return function_kind_ == FunctionKind::kOutVariant;
  }

  bool has_native() const {
    return function_kind_ == FunctionKind::kNativeFunction;
  }

  bool verify_no_memory_overlap() const;

  const char* get_op_name() const {
    return op_name_;
  }

 private:
  bool verify_outputs_dont_overlap_each_other() const;

  bool verify_inputs_dont_overlap_outputs() const;

  Node* node_;
  enum class FunctionKind {
    kOutVariant,
    kNativeFunction,
    kInterpreterFallback,
  };
  FunctionKind function_kind_;
  std::function<void(ProcessedNode*)> fn_;
  std::unique_ptr<const IValue*[]> inputs_; // unowned
  std::unique_ptr<IValue[]> outputs_;
  size_t inputs_size_;
  size_t outputs_size_;
  const char* op_name_;
};

} // namespace jit
} // namespace torch
