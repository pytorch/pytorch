#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/symbol.h>
#include <c10/core/CPUAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/FbcodeMaps.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/graph_node_list.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>
#include <torch/custom_class.h>
#include <limits>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#endif

namespace torch::jit {

TORCH_API bool canEnableStaticRuntime(
    const std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API std::string dumpValueSet(
    const c10::FastSet<const Value*>& value_set,
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

TORCH_API inline c10::Symbol getStaticRuntimeMetadataSymbol() {
  return Symbol::attr("static_runtime::metadata");
}

TORCH_API inline bool borrowsOutputs(c10::Symbol kind) {
  static const std::array<c10::Symbol, 4> symbols_with_borrowed_outputs = {
      c10::Symbol::fromQualString("static_runtime::select_tensor"),
      c10::Symbol::fromQualString("static_runtime::dict_unpack"),
      c10::Symbol::fromQualString("static_runtime::VarTupleUnpack"),
      c10::Symbol::fromQualString("prim::IfThenElse"),
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
  void init(const Block& block, const AliasDb& db);

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
  c10::FastSet<const Value*> output_aliases_;
  c10::FastSet<const Value*> external_aliases_;
};

class TORCH_API ManagedTensorRanges {
 public:
  ManagedTensorRanges() = default;
  ManagedTensorRanges(
      Block& block,
      const AliasDb& alias_db,
      const c10::FastSet<const Value*>& managed_tensor_values);

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
  c10::FastMap<Node*, std::vector<const Value*>> node_to_newly_free_tensors_{};
  // Maps each Value* to its lifetime (start node index, end node index)
  c10::FastMap<const Value*, Lifetime> value_lifetimes_{};
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
  // Gates the ReplaceWithCopy pass, which replaces ops that
  // sometimes alias their outputs with out variants that
  // always copy (so the output may participate in memory planning).
  // Since replacing with copies is done after TensorExpr fusion, the
  // resulting graph does not conform to the assumptions made in the fuser.
  // So, even if this flag is turned on, the ReplaceWithCopy pass will not
  // be executed if TensorExpr fusion is enabled.
  bool use_copy_variants{true};
  // Gates the ReplaceWithMaybeCopy pass, which replaces ops that
  // sometimes alias their outputs with subgraphs that include an out
  // variant.
  // For the same reason as `use_copy_variants`, the ReplaceWithMaybeCopy pass
  // will not be executed if TensorExpr fusion is enabled, even if this flag
  // is turned on.
  bool use_maybe_copy_variants{true};
  // enable TensorExpr fusion of ops at model loading time
  bool enable_tensorexpr_fusion{false};
};

/*
  Responsible for plugging StaticRuntime metadata onto the
  IR nodes. StaticRuntimeMetdata extends CustomClassHolder
  which can be casted to IValue and attached to IR node.
  This is needed to pass parent graph metadata to forked
  graph in presence of prim::fork operator
*/
class TORCH_API StaticRuntimeMetadata : public torch::CustomClassHolder {
 public:
  explicit StaticRuntimeMetadata(const StaticModuleOptions& opts)
      : opts_(opts) {}

  const StaticModuleOptions& get_opts() {
    return opts_;
  }

 private:
  StaticModuleOptions opts_;
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
class StaticNodeInfo;
class ProcessedFunction;
class ProcessedNode;
class StaticRuntime;

using SROperator = std::function<void(ProcessedNode*)>;

#ifdef FBCODE_CAFFE2
struct TORCH_API SROperatorObserver {
  using OperatorCallback = void (*)(const Node*);
  OperatorCallback startCb = nullptr;
  OperatorCallback endCb = nullptr;

  static void setCurrentThreadObserver(SROperatorObserver* observer);
  static SROperatorObserver* getCurrentThreadObserver();
  static void onStart(const Node* name);
  static void onEnd(const Node* name);
};
#endif

// A `BlockInfo` instance stores all of the shared state that each
// `BlockRunner` will need to access. Most of this information is
// read-only and shared between threads.
// - Each `BlockInfo` corresponds to one block in the graph.
// - Each `BlockInfo` may be used by multiple block runners (when there are many
//   threads).
// - All of the `BlockInfo`s are stored in a vector in the `StaticModule` and
//   are initialized during `StaticModule` construction.
// - Most of the information stored is used to initialize the block's memory
//   planner.
class BlockInfo {
 public:
  BlockInfo(uint32_t input_idx, Block& block);

  void set_nodes(
      std::vector<StaticNodeInfo> nodes,
      const c10::FastMap<Node*, bool>& node_has_out_variant);

  const std::vector<StaticNodeInfo>& nodes() const {
    return nodes_;
  }

  size_t num_nodes() const;

  size_t num_inputs() const {
    return block_.inputs().size();
  }

  size_t num_outputs() const {
    return block_.outputs().size();
  }

  graph_node_list node_ptrs() const {
    return block_.nodes();
  }

  void set_output_indices(std::vector<uint16_t> indices) {
    output_indices_ = std::move(indices);
  }

  const std::vector<uint16_t>& block_output_indices() const {
    return output_indices_;
  }

  auto block_inputs_idx() const {
    return input_idx_;
  }

  bool node_is_optimizable_container_type(const Node* node) const {
    return node_is_optimizable_container_type_.find(node) !=
        node_is_optimizable_container_type_.end();
  }

  bool value_is_managed_tensor(const Value* value) const {
    return managed_tensor_values_.find(value) != managed_tensor_values_.end();
  }

  bool value_is_leaked_container(const Value* value) const {
    return leaked_values_.find(value) != leaked_values_.end();
  }

  const ValueGroup& value_group() const {
    return value_group_;
  }

  const ManagedTensorRanges& managed_tensor_ranges() const {
    return managed_tensor_ranges_;
  }

  void init_value_group(const AliasDb& alias_db) {
    value_group_.init(block_, alias_db);
  }

  void prepare_for_memory_planner(
      const AliasDb& alias_db,
      const StaticModuleOptions& opt);

  const auto& managed_output_tensor_values() const {
    return managed_output_tensor_values_;
  }

  const auto& managed_tensor_values() const {
    return managed_tensor_values_;
  }

  const auto& leaked_values() const {
    return leaked_values_;
  }

 private:
  std::vector<StaticNodeInfo> nodes_;

  ValueGroup value_group_;

  c10::FastSet<const Node*> node_is_optimizable_container_type_;
  c10::FastSet<const Value*> managed_tensor_values_;
  c10::FastSet<const Value*> managed_output_tensor_values_;
  c10::FastSet<const Value*> leaked_values_;

  ManagedTensorRanges managed_tensor_ranges_{};

  // The index of this block's inputs in the shared values_ array.
  const uint16_t input_idx_;
  // The indices of this block's outputs in the shared values_ array.
  std::vector<uint16_t> output_indices_;
  Block& block_;
};

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

 private:
  explicit StaticModule(
      std::pair<std::shared_ptr<torch::jit::Graph>, c10::optional<Module>>
          graph_and_module,
      const StaticModuleOptions& opts);

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

  const BlockInfo& block_info(Block* block) const {
    return block_infos_.at(block);
  }

  Block* root_block() const {
    return graph_->block();
  }

 private:
  friend class StaticRuntime;
  friend class BlockRunner;

 public:
  auto num_nodes() const {
    return std::accumulate(
        block_infos_.begin(),
        block_infos_.end(),
        0,
        [](size_t sum, const auto& block_and_info) {
          auto& block_info = block_and_info.second;
          return sum + block_info.num_nodes();
        });
  }

  C10_NODISCARD Node* findNodeWithKindForTesting(const std::string& kind) const;

  const c10::optional<c10::FunctionSchema>& schema() const {
    return schema_;
  }

  bool first_input_is_self() const {
    return module_.has_value();
  }

  StaticRuntime& runtime();

  // See [Shared values array]
  size_t value_buffer_size() const {
    return value_buffer_size_;
  }

 private:
  // Recursively prepares the BlockInfo array.
  // - Populates `value_to_index` with the indices of each intermediate value
  // - Returns the number of Value* processed, including sub-blocks.
  size_t prepareBlockInfo(
      Block* block,
      const size_t start_idx,
      c10::FastMap<const Value*, uint32_t>& value_to_index);

  void prepareFunctionsAndConstants(
      Block* block,
      const AliasDb& alias_db,
      c10::FastMap<const Value*, uint32_t>& value_to_index);

  // Recursively traverse the graph and attach SR metadata
  // to the prim::fork nodes as additional attributes
  void attachNodeMetadata(Block* block);

  // Recurses on sub-blocks and populates the array of ProcessedNodes
  // Returns (number of nodes processed, number of blocks processed)
  size_t prepareStaticNodeInfos(
      Block* block,
      const c10::FastMap<const Value*, uint32_t>& value_to_index,
      const AliasDb& alias_db,
      size_t node_idx = 0);

  // Initialize various attributes that the memory planner will need.
  // To be called at the tail of the ctor.
  void prepareForMemoryPlanner();

  StaticModuleOptions opts_;
  // metadata that is stored in IR nodes as attribute
  at::intrusive_ptr<jit::StaticRuntimeMetadata> sr_metadata_;
  std::shared_ptr<torch::jit::Graph> graph_;
  c10::optional<torch::jit::Module> module_;
  c10::optional<c10::FunctionSchema> schema_;
  std::unique_ptr<StaticRuntime> cached_runtime_;

  // Bookkeeping for creating new StaticRuntime instances
  // IValue table (defined by prim::Constant nodes)
  std::vector<IValue> constants_;
  // The functions to be called by corresponding ProcessedNode.
  std::vector<ProcessedFunction> functions_{};
  // A list of pre-processed nodes from which ProcessedNode are created per
  // StaticRuntime instance.
  std::vector<StaticNodeInfo> nodes_;
  // Indices of graph outputs in the single values array.
  std::vector<uint16_t> output_indices_;

  size_t num_intermediate_values_ = 0;

  // Includes self if module_ != nullopt.
  // Note that we might have num_inputs_ == 0 even if the schema has a `self`
  // argument. In this case, `self` isn't used in the graph, but the schema
  // includes it anyways to be consistent with the JIT interpreter.
  size_t num_inputs_;
  // See `BlockInfo` definition. The blocks are stored in depth-first order.
  c10::FastMap<Block*, BlockInfo> block_infos_;
  size_t value_buffer_size_ = 0;
};

// `BlockRunner` contains the core runtime logic. Each block runner
// corresponds to one block in the graph and has its own memory planner.
// `StaticRuntime` will initialize all `BlockRunner`s
// upon construction. Each block runner only directly executes nodes from its
// block. Special ops with sub-blocks like `prim::If` may have
// `BlockRunner`s stored in their `ProcessedNode`s; these
// sub-blocks get executed in the op's implementation.
// `StaticRuntime` stores a vector of IValues that all
// `BlockRunner`s share. This vector is used to store all
// constants, inputs, and intermediate tensors.
class TORCH_API BlockRunner {
 public:
  BlockRunner(
      const StaticModule& sm,
      IValue* values,
      Block* block,
      torch::jit::TaskLauncher* launcher,
      bool is_root_block = false);
  BlockRunner(BlockRunner&&) noexcept;
  BlockRunner& operator=(BlockRunner&&) = delete;
  ~BlockRunner();

  C10_DISABLE_COPY_AND_ASSIGN(BlockRunner);

  using KeywordArgs = std::unordered_map<std::string, c10::IValue>;
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs());
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs());

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs);

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs);

  void benchmark(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const uint32_t warmup_runs,
      const uint32_t main_runs,
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
      const uint32_t warmup_runs,
      const uint32_t main_runs);

  // Input is readwrite
  IValue& Input(uint32_t i) {
    TORCH_DCHECK_LT(i, block_info_.num_inputs());
    return values_[i + block_info_.block_inputs_idx()];
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
    return block_info_.node_ptrs();
  }

  const Graph& graph() const {
    return static_module_.graph();
  }

  const MemoryPlanner* get_memory_planner() const {
    return planner_.get();
  }

  bool check_for_memory_leak(
      bool output_returned = true,
      bool recurse_on_sub_blocks = false);

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
    explicit Deallocator(BlockRunner& block_runner)
        : block_runner_(block_runner) {}

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
    BlockRunner& block_runner_;
  };

  template <typename IValueList>
  c10::IValue run_impl(IValueList&& args, const KeywordArgs& kwargs);

  template <typename IValueList>
  c10::IValue run_impl_record_functions(
      IValueList&& args,
      const KeywordArgs& kwargs);

  template <typename IValueList>
  c10::intrusive_ptr<c10::ivalue::Future> run_impl_async(
      IValueList&& args,
      const KeywordArgs& kwargs);

  template <typename IValueList>
  c10::intrusive_ptr<c10::ivalue::Future> run_impl_record_functions_async(
      IValueList&& args,
      const KeywordArgs& kwargs);

  // helper method for copying input args/kwargs into inputs_
  template <typename IValueList>
  void set_inputs(IValueList&& args, const KeywordArgs& kwargs);

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
    for (const auto idx : c10::irange(block_info_.num_inputs())) {
      values_[idx + inputs_begin_] = IValue();
    }
  }

  void clean_up_intermediate_ivalues() noexcept;

  IValue move_outputs_to_tuple(uint32_t num_outputs);

  void create_memory_planner();

  float benchmark_model(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const uint32_t warmup_runs,
      const uint32_t main_runs);

  void display_nodes(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs);

  const StaticModule& static_module_;
  const BlockInfo& block_info_;

  const bool is_root_block_;
  // Cache this so we don't have to call static_module_.first_input_is_self()
  const bool first_input_is_self_;
  // Index of the start of this blocks inputs in the shared values_ array.
  const uint16_t inputs_begin_;

  bool manage_output_tensors_enabled_ = false;
  std::unique_ptr<MemoryPlanner> planner_;
  // [Shared values array]
  // ProcessedNodes reference their inputs and outputs with
  // offsets into this array, which saves memory.
  // All BlockRunners share the same array. The layout is as
  // follows:
  // [constants][block_0][block_1]...[block_N]
  // Note that constants from all blocks are pooled together at the start.
  // The block ordering is depth-first.
  // Each block is further divided into inputs and intermediates:
  // [block_i] = [inputs_i][intermediates_i]
  // Each BlockRunner knows where its inputs start. Each ProcessedNode
  // knows how to find the indices of its outputs/inputs in this array.
  IValue* values_;

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

  size_t num_outputs() const {
    return num_outputs_;
  }

 private:
  SROperator f_;
  Kind kind_{ProcessedFunction::Kind::kOutVariant};
  bool check_memory_overlap_{false};
  size_t num_outputs_{0};
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API StaticNodeInfo {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  StaticNodeInfo(
      Node* n,
      ProcessedFunction* fn,
      ProcessedNodeInputs inputs,
      uint16_t outputs_offset);

  Node* node() const {
    return node_;
  }

  size_t num_outputs() const {
    DCHECK(fn_ != nullptr);
    return fn_->num_outputs();
  }

  bool has_out_variant() const {
    return fn_->kind() == ProcessedFunction::Kind::kOutVariant;
  }

 private:
  friend class ProcessedNode;

  Node* node_;
  const ProcessedFunction* fn_;
  ProcessedNodeInputs inputs_;
  uint16_t outputs_offset_;
};

inline size_t BlockInfo::num_nodes() const {
  return nodes_.size();
}

/*
  ProcessedNodeMetadata class wraps the possible metadata
  for ProcessedNode. Depending upon the nature of op, processedNode
  can have one of the below possibilities of metadata:
  - prim::If/prim::Loop ops contains block_runners_ as their metadata
  - prim::fork op contains TaskLauncher (std::function) responsible for
    execution of forked subgraph
*/
class TORCH_API ProcessedNodeMetadata {
 public:
  ProcessedNodeMetadata(
      std::vector<BlockRunner> runners,
      torch::jit::TaskLauncher* launcher)
      : block_runners_(std::move(runners)), launcher_(launcher) {}

  ProcessedNodeMetadata() : launcher_(nullptr) {}

  // deleted copy ctor/assignment as standard containers (vector) always
  // have copy constructors, but their instantiation is not well-formed
  // if the contained type (BlockRunner) is not copyable
  ProcessedNodeMetadata(const ProcessedNodeMetadata&) = delete;
  ProcessedNodeMetadata& operator=(const ProcessedNodeMetadata&) = delete;

  std::vector<BlockRunner>& block_runners() {
    return block_runners_;
  }

  void set_block_runners(std::vector<BlockRunner> runners) {
    block_runners_ = std::move(runners);
  }

  void set_launcher(torch::jit::TaskLauncher* launcher) {
    launcher_ = launcher;
  }

  torch::jit::TaskLauncher* launcher() {
    return launcher_;
  }

 private:
  std::vector<BlockRunner> block_runners_;
  torch::jit::TaskLauncher* launcher_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ProcessedNode() = default;

  ProcessedNode(const StaticNodeInfo& other, IValue* values)
      : node_(other.node_),
        fn_(other.fn_),
        inputs_(other.inputs_),
        outputs_offset_(other.outputs_offset_),
        values_(values),
        metadata_(nullptr) {}

  // These should be noexcept, but some Android build is failing
  // saying the noexcept specification doesn't match the calculated
  // one. Maybe std::variant is throwing it off?
  ProcessedNode(ProcessedNode&&) = default;

  ProcessedNode(const ProcessedNode&) = delete;
  ProcessedNode& operator=(const ProcessedNode& other) = delete;
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
    DCHECK(i < num_outputs());
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < num_outputs());
    return values_[outputs_offset_ + i];
  }

  size_t num_outputs() const {
    DCHECK(fn_ != nullptr);
    return fn_->num_outputs();
  }

  C10_NODISCARD c10::ArrayRef<const IValue> outputs() const {
    return c10::ArrayRef<const IValue>(
        values_ + outputs_offset_, num_outputs());
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
    return node_->kind().toQualString();
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
    DCHECK(i < num_outputs());
    return outputs_offset_ + i;
  }
  // used in debug mode
  bool verify_no_memory_overlap(bool force_check = false) const;

  // returns pointer to ProcessedNodeMetadata or nullptr if no object is owned
  ProcessedNodeMetadata* metadata() {
    return metadata_.get();
  }

  // attach block_runner to metadata of ProcessedNode
  void set_metadata(std::vector<BlockRunner> block_runners) {
    if (metadata_ == nullptr) {
      metadata_ = std::make_unique<ProcessedNodeMetadata>();
    }
    metadata_->set_block_runners(std::move(block_runners));
  }

  // attach TaskLauncher to metadata of ProcessedNode
  void set_metadata(torch::jit::TaskLauncher* launcher) {
    if (metadata_ == nullptr) {
      metadata_ = std::make_unique<ProcessedNodeMetadata>();
    }
    metadata_->set_launcher(launcher);
  }

 private:
  C10_NODISCARD bool verify_outputs_dont_overlap_each_other() const;

  C10_NODISCARD bool verify_inputs_dont_overlap_outputs(bool force_check) const;

  Node* node_;
  const ProcessedFunction* fn_;
  ProcessedNodeInputs inputs_;
  uint16_t outputs_offset_;
  bool overlap_detected_{false};
  IValue* values_ = nullptr; // unowned
  // Metadata for ProcessedNode.
  // 1. prim::If/Loop nodes contains sub-blocks as metadata
  // 2. prim::fork nodes contains custom executor for async execution
  std::unique_ptr<ProcessedNodeMetadata> metadata_;
};

// `StaticRuntime` is the owner of the array of IValues (used for constants,
// inputs, and intermediate tensors) that all `BlockRunner`s share.
// Upon construction, it initializes all block runners. `operator()` simply
// forwards the inputs to the top-level block runner. Each `StaticRuntime`
// instance corresponds to one `StaticModule`. Multiple `StaticRuntime`
// instances can be created; this is useful for multi-threaded execution, since
// `operator()` is not thread-safe.
class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(const StaticModule& sm);

  using KeywordArgs = std::unordered_map<std::string, c10::IValue>;
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs());
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs());

  // runAsync performs inline execution of graph on
  // caller thread and async execution on taskLauncher
  // If no custom taskLauncher is specified, execution is done
  // on inter-op thread pool.
  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      const std::vector<c10::IValue>& args,
      const KeywordArgs& kwargs = KeywordArgs(),
      torch::jit::TaskLauncher taskLauncher = at::launch);

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      std::vector<c10::IValue>&& args,
      const KeywordArgs& kwargs = KeywordArgs(),
      torch::jit::TaskLauncher taskLauncher = at::launch);

  bool check_for_memory_leak(bool output_returned = true);
  bool checkOutputTensorMemoryLeaks();

  void deallocateOutputTensors();
  bool isManagedOutputTensor(const IValue& ivalue) const;
  void disableManageOutputTensors();

  // Gets the top-level memory planner. Used for testing.
  const MemoryPlanner* get_memory_planner() const;

  void benchmark(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const uint32_t warmup_runs,
      const uint32_t main_runs,
      bool print_per_node_time = false,
      bool generate_ai_pep_output = false) {
    block_->benchmark(
        args_list,
        kwargs_list,
        warmup_runs,
        main_runs,
        print_per_node_time,
        generate_ai_pep_output);
  }

  using IndividualMetrics = BlockRunner::IndividualMetrics;

  IndividualMetrics benchmark_individual_ops(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<KeywordArgs>& kwargs_list,
      const int warmup_runs,
      const int main_runs) {
    return block_->benchmark_individual_ops(
        args_list, kwargs_list, warmup_runs, main_runs);
  }

 private:
  // An array of IValues with unchanging size/data ptr.
  class IValueArray {
   public:
    IValueArray() = default;
    explicit IValueArray(size_t size) : array_(allocate(size)), size_(size) {}

    IValue* data() const {
      return array_.get();
    }

    size_t size() const {
      return size_;
    }

   private:
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    static std::unique_ptr<IValue[]> allocate(size_t size) {
      if (size) {
        return std::make_unique<IValue[]>(size);
      }
      return nullptr;
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    std::unique_ptr<IValue[]> array_ = nullptr;
    size_t size_ = 0;
  };

  std::unique_ptr<BlockRunner> block_;
  // for execution of async operations present in graph
  torch::jit::TaskLauncher async_task_launcher_;
  IValueArray values_;
};

} // namespace torch::jit
