#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/variant.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>

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
class ProcessedFunction;
class ProcessedNode;
class StaticRuntime;
class ValueGroup;
class TORCH_API StaticModule {
 public:
  explicit StaticModule(
      std::shared_ptr<torch::jit::Graph> g,
      const StaticModuleOptions& opts = StaticModuleOptions());

  explicit StaticModule(
      const torch::jit::Module& m,
      bool is_frozen = false,
      const StaticModuleOptions& opts = StaticModuleOptions());

  StaticModule(const torch::jit::StaticModule&) = delete;
  StaticModule(torch::jit::StaticModule&&) noexcept;
  StaticModule& operator=(StaticModule&&) = delete;
  ~StaticModule();

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
  using KeywordArgs = std::unordered_map<std::string, c10::IValue>;
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs());
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs());

  const Graph& graph() const {
    return *graph_;
  }

  const Module& module() const {
    DCHECK(module_.has_value());
    return *module_;
  }

  const StaticModuleOptions& opts() const;

  const ValueGroup& valueGroup() const {
    return *value_group_;
  }

  size_t num_inputs() const;
  size_t num_outputs() const;

  C10_NODISCARD const std::vector<uint16_t>& output_indices() const {
    return output_indices_;
  }

  const std::vector<IValue>& constants() const {
    return constants_;
  }

 private:
  friend class StaticRuntime;

  // Our nodes don't have their inputs & outputs initialized; don't
  // let anybody but StaticRuntime and tests get them.
  const std::vector<ProcessedNode>& nodes() const;

 public:
  size_t num_nodes() const;

  C10_NODISCARD Node* findNodeWithKindForTesting(const std::string& kind) const;

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
    return *value_group_;
  }

  const FastSet<const Value*>& managed_tensor_values() const {
    return managed_tensor_values_;
  }

  const FastSet<const Value*>& managed_output_tensor_values() const {
    return managed_output_tensor_values_;
  }

  const FastSet<const Value*>& leaked_values() const {
    return leaked_values_;
  }

  bool first_input_is_self() const {
    return module_.has_value();
  }

  StaticRuntime& runtime();

 private:
  // Initialize various attributes that the memory planner will need.
  // To be called at the tail of the ctor.
  void prepareForMemoryPlanner();

  StaticModuleOptions opts_;
  std::shared_ptr<torch::jit::Graph> graph_;
  c10::optional<torch::jit::Module> module_;
  c10::optional<c10::FunctionSchema> schema_;
  std::unique_ptr<StaticRuntime> cached_runtime_;

  // Bookkeeping for creating new StaticRuntime instances
  // IValue table (defined by prim::Constant nodes)
  std::vector<IValue> constants_;
  // The functions to be called by corresponding ProcessedNode.
  std::vector<ProcessedFunction> functions_;
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
  // Indices of graph outputs in the single values array.
  std::vector<uint16_t> output_indices_;

  std::unique_ptr<ValueGroup> value_group_;

  // map a value to the set of values that may share the same storage with it
  FastMap<const Value*, std::vector<const Value*>>
      value_to_same_storage_values_;

  FastSet<const Node*> node_is_optimizable_container_type_;

  FastSet<const Value*> managed_tensor_values_{};
  FastSet<const Value*> managed_output_tensor_values_{};
  FastSet<const Value*> leaked_values_{};
};

class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(const StaticModule& sm);
  StaticRuntime(StaticRuntime&&) = delete;
  StaticRuntime& operator=(StaticRuntime&&) = delete;
  ~StaticRuntime();

  C10_DISABLE_COPY_AND_ASSIGN(StaticRuntime);

  using KeywordArgs = std::unordered_map<std::string, c10::IValue>;
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs());
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs());

  void benchmark(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const int warmup_runs,
      const int main_runs,
      bool print_per_node_time = false,
      bool generate_ai_pep_output = false);

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
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const int warmup_runs,
      const int main_runs);

  // Input is readwrite
  IValue& Input(uint32_t i) {
    DCHECK_LT(i, static_module_.num_inputs());
    DCHECK_LT(i, values_.size());
    return values_[i];
  }

  // Output is readonly. The writing process happens inside ProcessedNodes
  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < outputs_.size());
    return *outputs_[i];
  }

  const std::vector<IValue*> outputs() const {
    return outputs_;
  }

  const std::vector<ProcessedNode>& nodes() const;

  std::vector<ProcessedNode>& nodes();

  const Graph& graph() const {
    return static_module_.graph();
  }

  const MemoryPlanner* get_memory_planner() const {
    return planner_.get();
  }

  void check_for_memory_leak(bool output_returned = true);

  bool is_optimizable_container_type(Node* n) const {
    return static_module_.is_optimizable_container_type(n);
  }

  // WARNING: Deallocate managed output tensors.  A client receiving Static
  // Runtime-managed Tensors needs to be very careful to call
  // `StaticRuntime::deallocateOutputTensors` after all references of output
  // Tensors are gone.
  void deallocateOutputTensors();

  bool checkOutputTensorMemoryLeaks();

  bool isManagedOutputTensor(const IValue& ivalue) const;
  bool isManagedOutputTensorValue(const Value* value) const;

  void disableManageOutputTensors();

 private:
  template <typename IValueList>
  c10::IValue run_impl(IValueList&& args, const KeywordArgs& kwargs);

  template <typename IValueList>
  c10::IValue run_impl_record_functions(
      IValueList&& args,
      const KeywordArgs& kwargs);

  // helper method for copying input args/kwargs into inputs_
  void set_inputs(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs);
  void set_inputs(std::vector<c10::IValue>&& args, const KeywordArgs& kwargs);

  void verify_and_correct_memory_overlap(ProcessedNode& n);

  // clean up owning refs of input IValues
  void clean_up_input_ivalues() {
    for (const auto idx : c10::irange(static_module_.num_inputs())) {
      values_[idx] = IValue();
    }
  }

  IValue move_outputs_to_tuple(uint32_t num_outputs);

  void create_memory_planner();

  float benchmark_model(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const int warmup_runs,
      const int main_runs);

  void display_nodes(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs);

  // Memory planning is only enabled if sm->opts().cleanup_activations is true.
  // Otherwise, the memory used by activations is cached inside the static
  // runtime.
  const StaticModule& static_module_;
  bool manage_output_tensors_enabled_ = false;
  std::unique_ptr<MemoryPlanner> planner_;
  // first static_module_.num_inputs() slots are inputs, next
  // static_module_.constants().size() slots are a copy of
  // static_module_.constants(), rest are regular values in the
  // graph. ProcessedNodes reference their inputs and outputs with
  // offsets into this array, which saves memory.
  std::vector<IValue> values_;
  std::vector<IValue*> outputs_;
  std::vector<ProcessedNode> nodes_;
};

} // namespace jit
} // namespace torch
