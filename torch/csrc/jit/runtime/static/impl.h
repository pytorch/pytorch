#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <c10/core/CPUAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/variant.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/graph_node_list.h>
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

TORCH_API std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name = "");

TORCH_API inline bool doesNotHeapAllocateWhenStoredInIValue(const Type& type) {
  switch (type.kind()) {
    // NOTE: NumberType may allocate because it includes complex.
    case TypeKind::NoneType:
    case TypeKind::IntType:
    case TypeKind::FloatType:
    case TypeKind::BoolType:
    case TypeKind::DeviceObjType:
    case TypeKind::StreamObjType:
      return true;
    default:
      return false;
  }
}

TORCH_API inline bool borrowsOutputs(c10::Symbol kind) {
  static const std::array<c10::Symbol, 3> symbols_with_borrowed_outputs = {
      c10::Symbol::fromQualString("static_runtime::select_tensor"),
      c10::Symbol::fromQualString("static_runtime::dict_unpack"),
      c10::Symbol::fromQualString("static_runtime::VarTupleUnpack"),
  };
  return std::find(
             symbols_with_borrowed_outputs.begin(),
             symbols_with_borrowed_outputs.end(),
             kind) != symbols_with_borrowed_outputs.end();
}

// Group values used by `graph` into three categories:
//
// - output_aliases:
//     values that are either outputs or contain aliases of outputs
// - external_aliases:
//     values that are inputs, constants, or their aliases.
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

class TORCH_API ManagedTensorRanges {
 public:
  ManagedTensorRanges() = default;
  ManagedTensorRanges(
      const std::shared_ptr<Graph>& graph,
      const FastSet<const Value*>& managed_tensor_values);

  // If true, then this node is the last use of at least one
  // managed tensor. availableTensorValuesAfterNode(node) will return a vector
  // of the managed tensors that are available for re-use
  // in the nodes following this one.
  bool nodeFreesManagedTensors(Node* node) const;
  const std::vector<const Value*>& availableTensorValuesAfterNode(
      Node* node) const;

  // For testing. True if v1 and v2 are both mutable types and have lifetimes
  // that overlap.
  bool lifetimesOverlap(const Value* v1, const Value* v2) const;

 private:
  struct Lifetime {
    Lifetime(size_t start_, size_t end_) : start(start_), end(end_) {}
    size_t start;
    size_t end;
  };

  // Returns nullptr if we are not tracking the lifetime of value
  Lifetime* getLifetime(const Value* value);
  const Lifetime* getLifetime(const Value* value) const;
  // Collect all values in the input that have tracked lifetimes.
  // A value's lifetime may not be tracked if it is a graph input
  // or immutable type (containers with at least one mutable
  // type are mutable)
  std::vector<const Value*> collectValuesWithTrackedLifetimes(
      at::ArrayRef<const Value*> values);

  // Maps Node* to the set of managed tensors that are now available
  // for re-use after this node.
  FastMap<Node*, std::vector<const Value*>> node_to_newly_free_tensors_{};
  // Maps each Value* to its lifetime (start node index, end node index)
  FastMap<const Value*, Lifetime> value_lifetimes_{};
};

struct TORCH_API StaticModuleOptions {
  // enabling out variant allows Static Runtime to do memory planning
  bool enable_out_variant{true};
  // to reuse tensor storage for tensors whose live-range do not overlap to
  // reduce memory footprint (enable_out_variant must be true)
  bool optimize_memory{true};
  // to batch allocate tensor storage for output tensors of the
  // graph, where storage is deallocated outside static runtime
  // (enable_out_variant must be true)
  bool manage_output_tensors{false};
  // Gates the ReplaceWithMaybeCopy pass, which replaces ops that
  // sometimes alias their outputs with subgraphs that include an out
  // variant.
  bool use_maybe_copy_variants{true};
  // enable TensorExpr fusion of ops at model loading time
  bool enable_tensorexpr_fusion{false};
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
class TORCH_API StaticModule {
 public:
  explicit StaticModule(
      std::shared_ptr<torch::jit::Graph> g,
      const StaticModuleOptions& opts = StaticModuleOptions(),
      std::vector<IValue> sample_inputs = {});

  explicit StaticModule(
      const torch::jit::Module& m,
      bool is_frozen = false,
      const StaticModuleOptions& opts = StaticModuleOptions(),
      std::vector<IValue> sample_inputs = {});

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
    return value_group_;
  }

  size_t num_inputs() const;
  size_t num_outputs() const;

  size_t num_constants() const {
    return constants_.size();
  }

  size_t num_intermediate_values() const {
    return num_intermediate_values_;
  }

  size_t total_num_values() const {
    return num_inputs() + num_constants() + num_intermediate_values();
  }

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
  const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

 public:
  auto num_nodes() const {
    return nodes_.size();
  }

  C10_NODISCARD Node* findNodeWithKindForTesting(const std::string& kind) const;

  graph_node_list node_ptrs() const {
    return graph_->nodes();
  }

  bool is_optimizable_container_type(const Node* n) const {
    auto it = node_is_optimizable_container_type_.find(n);
    return it != node_is_optimizable_container_type_.end();
  }

  const c10::optional<c10::FunctionSchema>& schema() const {
    return schema_;
  }

  const ValueGroup& value_group() const {
    return value_group_;
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

  const ManagedTensorRanges& managed_tensor_ranges() const {
    return managed_tensor_ranges_;
  }

  bool first_input_is_self() const {
    return module_.has_value();
  }

  size_t inputs_offset() const {
    return 0;
  }

  size_t constants_offset() const {
    return inputs_offset() + num_inputs();
  }

  size_t intermediate_values_offset() const {
    return constants_offset() + num_constants();
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
  std::vector<ProcessedFunction> functions_{};
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
  // Indices of graph outputs in the single values array.
  std::vector<uint16_t> output_indices_;

  ValueGroup value_group_;

  FastSet<const Node*> node_is_optimizable_container_type_;

  FastSet<const Value*> managed_tensor_values_{};
  FastSet<const Value*> managed_output_tensor_values_{};
  FastSet<const Value*> leaked_values_{};
  ManagedTensorRanges managed_tensor_ranges_{};

  size_t num_intermediate_values_ = 0;

  // Includes self if module_ != nullopt.
  // Note that we might have num_inputs_ == 0 even if the schema has a `self`
  // argument. In this case, `self` isn't used in the graph, but the schema
  // includes it anyways to be consistent with the JIT interpreter.
  size_t num_inputs_;
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

  const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

  std::vector<ProcessedNode>& nodes() {
    return nodes_;
  }

  graph_node_list node_ptrs() const {
    return static_module_.node_ptrs();
  }

  const Graph& graph() const {
    return static_module_.graph();
  }

  const MemoryPlanner* get_memory_planner() const {
    return planner_.get();
  }

  bool check_for_memory_leak(bool output_returned = true);

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

  // This is the fallback path taken if we can't construct the memory planner
  // on the first iteration.
  // IMPORTANT: Nothing here should be able to throw!!!
  // This function can be called from the (implicitly) `noexcept` destructor
  // of Deallocator, meaning that std::terminate will be called
  // if any exception escapes. Even if resetMemory and ~Deallocator were
  // `noexcept(false)`, it's possible that when ~Deallocator is called, the
  // stack is already unwinding, so there's still danger of calling
  // std::terminate.
  void resetMemory() noexcept;

 private:
  // A helper object that invokes memory planner deallocation code
  // when destructed.
  class Deallocator {
   public:
    explicit Deallocator(StaticRuntime& runtime) : runtime_(runtime) {}

    Deallocator(Deallocator&&) = default;
    Deallocator(const Deallocator&) = default;
    Deallocator& operator=(const Deallocator&) = delete;
    Deallocator& operator=(Deallocator&&) = delete;
    ~Deallocator();

    void setFinished() {
      finished_ = true;
    }

   private:
    void cleanupImpl();

    bool finished_ = false;
    StaticRuntime& runtime_;
  };

  template <typename IValueList>
  c10::IValue run_impl(IValueList&& args, const KeywordArgs& kwargs);

  template <typename IValueList>
  c10::IValue run_impl_record_functions(
      IValueList&& args,
      const KeywordArgs& kwargs);

  // helper method for copying input args/kwargs into inputs_
  template <typename IValueList>
  void set_inputs(
      IValueList&& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  // Set Input(idx) to args[idx]. Invoked by set_inputs. Copies or moves
  // depending on overload.
  void set_arg(const size_t idx, std::vector<IValue>&& args);
  void set_arg(const size_t idx, const std::vector<IValue>& args);

  // Set Input(idx) to arg. Always copies. Used for kwargs.
  void set_arg(const size_t idx, const IValue& arg);

  bool fast_check_and_correct_overlap_with(
      ProcessedNode& n,
      c10::IValue& tensor_ival);
  void verify_and_correct_memory_overlap(ProcessedNode& n);

  // clean up owning refs of input IValues
  void clean_up_input_ivalues() noexcept {
    for (const auto idx : c10::irange(static_module_.num_inputs())) {
      values_[idx] = IValue();
    }
  }

  void clean_up_intermediate_ivalues() noexcept;

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

  const StaticModule& static_module_;
  // Cache this so we don't have to call static_module_.first_input_is_self()
  const bool first_input_is_self_;
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

class TORCH_API ProcessedFunction {
 public:
  ProcessedFunction(
      Node* node,
      bool enable_out_variant,
      bool check_memory_overlap);

  enum class Kind : uint8_t {
    kOutVariant,
    kNativeFunction,
    kInterpreterFallback,
  };

  void run(ProcessedNode* pnode) const {
    return f_(pnode);
  }

  Kind kind() const {
    return kind_;
  }

  bool checkMemoryOverlap() const {
    return check_memory_overlap_;
  }

 private:
  std::function<void(ProcessedNode*)> f_;
  Kind kind_{ProcessedFunction::Kind::kOutVariant};
  bool check_memory_overlap_{false};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ProcessedNode() = default;
  // ProcessedNodes are created within StaticModule and then
  // associated with a shared values array using set_values() when
  // they are copied into a StaticRuntime.
  ProcessedNode(
      Node* n,
      ProcessedFunction* fn,
      ProcessedNodeInputs inputs,
      uint16_t outputs_offset);

  ProcessedNode(const ProcessedNode&) = default;
  ProcessedNode& operator=(const ProcessedNode&) = default;

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
  C10_NODISCARD const IValue& Input(uint32_t i) const {
    return values_[inputs_[i]];
  }

  // Output is readwrite
  IValue& Output(uint32_t i) {
    DCHECK(i < num_outputs_);
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < num_outputs_);
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD c10::ArrayRef<const IValue> outputs() const {
    return c10::ArrayRef<const IValue>(values_ + outputs_offset_, num_outputs_);
  }

  C10_NODISCARD auto num_outputs() const {
    return num_outputs_;
  }

  C10_NODISCARD uint16_t num_inputs() const {
    return inputs_.size();
  }

  std::vector<IValue> inputs_ivalue_vec() const;

  bool has_out_variant() const {
    return fn_->kind() == ProcessedFunction::Kind::kOutVariant;
  }

  bool has_native() const {
    return fn_->kind() == ProcessedFunction::Kind::kNativeFunction;
  }

#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* get_op_name() const {
    return op_name_;
  }
#endif

  bool check_outputs_for_memory_overlap() const {
    return fn_->checkMemoryOverlap();
  }

  void set_outputs_memory_overlap_detected() {
    overlap_detected_ = true;
  }

  bool outputs_memory_overlap_detected() {
    return overlap_detected_;
  }

  bool check_and_correct_overlap_with(
      const at::Tensor& input,
      c10::IValue& output);
  void verify_and_correct_memory_overlap();

  void set_values(IValue* values) {
    DCHECK(values_ == nullptr);
    values_ = values;
  }

  C10_NODISCARD uint16_t output_ivalue_index(uint16_t i) const {
    return outputs_offset_ + i;
  }
  // used in debug mode
  bool verify_no_memory_overlap(bool force_check = false) const;

 private:
  C10_NODISCARD bool verify_outputs_dont_overlap_each_other() const;

  C10_NODISCARD bool verify_inputs_dont_overlap_outputs(bool force_check) const;

  Node* node_;
  const ProcessedFunction* fn_;
  bool overlap_detected_{false};
  ProcessedNodeInputs inputs_;
  uint16_t outputs_offset_;
  uint16_t num_outputs_;
  IValue* values_ = nullptr; // unowned
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* op_name_;
#endif
};

} // namespace jit
} // namespace torch
