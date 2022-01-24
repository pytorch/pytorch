#include <torch/csrc/jit/runtime/static/impl.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/symbol.h>
#include <ATen/record_function.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/macros/Macros.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/irange.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/eliminate_no_ops.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/clone_native.h>
#endif

#include <iterator>
#include <sstream>
#include <stdexcept>

#ifdef FBCODE_CAFFE2
#include <common/logging/logging.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#endif

// used in test only
C10_DEFINE_bool(
    static_runtime_disable_debug_memory_overlap_check,
    false,
    "If true, disable the memory overlap check in debug mode in ProcessedNode::run()");

namespace torch {
namespace jit {

// A manually curated set of ops that are disallowed in static runtime.
// These are rarely-used ops. Disallowing them typically eliminates
// corner cases in graph optimizations, allowing for more aggressive
// optimizations and better performance.
bool isUnsupportedOp(const NodeKind& kind) {
  return kind == aten::__is__ || kind == aten::__isnot__;
}

// graph must be frozen or canEnableStaticRuntime would return false if there's
// any prim::CallMethod op left in the graph
bool canEnableStaticRuntime(const std::shared_ptr<torch::jit::Graph>& graph) {
  // check for sub-blocks
  bool can_support = true;
  bool has_blocks = false;
  for (auto* node : graph->block()->nodes()) {
    if (node->blocks().size() > 0) {
      has_blocks = true;
      VLOG(1) << "Found nested sub-blocks in graph at node: "
              << PrintNode(node);
    }
    const auto kind = node->kind();
    if (kind == prim::Constant) {
      continue;
    }
    // check if can get op from Node
    const Operator* op = node->maybeOperator();
    if (isUnsupportedOp(kind) || (!op && !nativeOpIsRegistered(kind))) {
      can_support = false;
      LOG(WARNING) << "Found unsupported op: " << kind.toQualString();
    }
  }
  if (has_blocks) {
    LOG(WARNING)
        << "Found nested sub-block in graph. Static Runtime doesn't support nested sub-blocks.";
    can_support = false;
  }
  return can_support;
}

std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name) {
  std::ostringstream oss;
  oss << set_name << ": {";
  for (const auto* val : value_set) {
    oss << "%" << val->debugName() << ", ";
  }
  oss << "}";
  return oss.str();
}

namespace {

void OptimizeGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  GRAPH_DUMP("Before optimizations: ", graph);
  if (opts.enable_tensorexpr_fusion) {
    if (sample_inputs.empty()) {
      VLOG(1) << "Cannot perform TensorExpr fusion - sample_inputs is empty";
    } else {
      VLOG(1) << "Performing TensorExpr fusion";
      performTensorExprFusion(graph, std::move(sample_inputs));
    }
  }
  Inline(*graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);
  FuseInferenceOpsForSparseNN(graph);
  UseVariadicCat(graph);
  UseVariadicStack(graph);
  EliminateTrivialEquallySplit(graph);

  if (opts.enable_out_variant) {
    UseVariadicOp(
        graph,
        fromQualString("fb::sigrid_transforms_torch_bind"),
        fromQualString("fb::variadic_sigrid_transforms_torch_bind"));
    FuseSignLog1P(graph);

    // TODO: we can avoid this guard by moving operations
    // to exposed folders.
#ifdef FBCODE_CAFFE2
    ReplaceWithCopy(graph);
    if (opts.use_maybe_copy_variants) {
      ReplaceWithMaybeCopy(graph);
    }
    FuseListUnpack(graph);
    EnableStaticRuntimeLayerNorm(graph);
#endif
  }

  ConstantPropagation(graph);
  RemoveImmutableInputDictLookups(graph);
  UseVariadicTupleUnpack(graph);
  UseVariadicGroupedAccessor(graph);
  EliminateNoOps(
      graph, /* custom_ops */ {fromQualString("fb::scale_gradient")});
  GRAPH_DUMP("Final graph after optimizations: ", graph);
}

bool IsSelfInGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  return !graph->inputs().empty() && graph->inputs().at(0)->type()->is_module();
}

// remove unused input 0 from graph
bool removeSelfFromGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  if (graph->inputs().at(0)->type()->is_module()) {
    if (graph->inputs().at(0)->hasUses()) {
      return false;
    }
    graph->eraseInput(0);
  }
  return true;
}

std::vector<Value*> valueVecFromFastSet(const FastSet<const Value*>& s) {
  std::vector<Value*> result;
  result.reserve(s.size());
  for (auto* v : s) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    result.emplace_back(const_cast<Value*>(v));
  }
  return result;
}

bool mayContainAlias(AliasDb& db, const Value* a, const Value* b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), const_cast<Value*>(b));
}

bool mayContainAlias(
    AliasDb& db,
    const Value* a,
    const FastSet<const Value*>& b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), valueVecFromFastSet(b));
}

bool mayContainAlias(
    AliasDb& db,
    const FastSet<const Value*>& a,
    const FastSet<const Value*>& b) {
  return db.mayContainAlias(valueVecFromFastSet(a), valueVecFromFastSet(b));
}

void PrepareGraphForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  TORCH_CHECK(canEnableStaticRuntime(graph));
  OptimizeGraph(graph, opts, std::move(sample_inputs));
}

std::pair<std::shared_ptr<Graph>, c10::optional<Module>> PrepareForStaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  LOG(INFO) << "StaticModuleOptions: enable_out_variant "
            << opts.enable_out_variant << ", optimize_memory "
            << opts.optimize_memory << ", manage_output_tensors "
            << opts.manage_output_tensors << ", use_maybe_copy_variants "
            << opts.use_maybe_copy_variants << ", enable_tensorexpr_fusion "
            << opts.enable_tensorexpr_fusion;

  Module module = m.copy();
  if (!is_frozen) {
    module.eval();
    module = freeze_module(module);
  }

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  if (!sample_inputs.empty() && IsSelfInGraphInput(graph)) {
    sample_inputs.insert(sample_inputs.begin(), m._ivalue());
  }
  PrepareGraphForStaticModule(graph, opts, std::move(sample_inputs));

  return std::make_pair(graph, module);
}

std::pair<std::shared_ptr<Graph>, c10::optional<Module>> PrepareForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  PrepareGraphForStaticModule(graph, opts, std::move(sample_inputs));
  return std::make_pair(graph, c10::nullopt);
}

} // namespace

void ValueGroup::init(
    const std::shared_ptr<torch::jit::Graph>& graph,
    AliasDb& db) {
  external_aliases_.clear();
  output_aliases_.clear();
  // Build `external_aliases` as we look through nodes forwardly from
  // the graph's inputs and add aliases of the inputs being created by the
  // nodes.
  external_aliases_.insert(graph->inputs().begin(), graph->inputs().end());
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      for (const auto* output : node->outputs()) {
        external_aliases_.insert(output);
      }
    }
  }
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      // Constants are already in `external_aliases`.
      continue;
    }
    for (const auto* v : node->outputs()) {
      if (mayContainAlias(db, v, external_aliases_)) {
        external_aliases_.insert(v);
      }
    }
  }

  // Build `output_aliases` as we look through nodes reversely so that we can
  // start from the output values, and follow the flows backwardly from there.
  output_aliases_.insert(graph->outputs().begin(), graph->outputs().end());
  for (const auto* node : graph->nodes().reverse()) {
    if (node->kind() == prim::Constant) {
      // Constants cannot create any aliases.
      continue;
    }
    for (const auto* v : node->outputs()) {
      // Add values that can aliase input/constant values. Note some output
      // aliases may end up in this category via collection objects (e.g.,
      // Tuple).
      if (mayContainAlias(db, v, external_aliases_)) {
        external_aliases_.insert(v);
        continue;
      }
      if (mayContainAlias(db, v, output_aliases_)) {
        output_aliases_.insert(v);
      }
    }
  }
}

namespace {

bool isTensorList(const Value* value) {
  auto* type = value->type()->castRaw<ListType>();
  if (!type) {
    return false;
  }
  return type->getElementType()->kind() == c10::TypeKind::TensorType;
}

bool containTensorsOnly(at::ArrayRef<Value*> values) {
  // return true only if all outputs are tensors
  return std::all_of(values.begin(), values.end(), [](const Value* value) {
    return value->type()->kind() == c10::TypeKind::TensorType ||
        isTensorList(value);
  });
}

bool mayContainAlias(const Value* v1, const Value* v2, const AliasDb& db) {
  // AliasDb is not const-correct here, so we have to const_cast
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(v1), const_cast<Value*>(v2));
}

bool isPureFunction(const Node* node) {
  auto* schema = node->maybeSchema();
  return schema &&
      schema->aliasAnalysis() == c10::AliasAnalysisKind::PURE_FUNCTION;
}

} // namespace

ManagedTensorRanges::ManagedTensorRanges(
    const std::shared_ptr<Graph>& graph,
    const FastSet<const Value*>& managed_tensor_values) {
  AliasDb alias_db(graph);
  const std::vector<Node*> nodes(graph->nodes().begin(), graph->nodes().end());
  const FastSet<const Value*> graph_inputs(
      graph->inputs().begin(), graph->inputs().end());

  auto isUntrackedValue = [&alias_db, &graph_inputs](const Value* value) {
    return !alias_db.isMutableType(value) ||
        graph_inputs.find(value) != graph_inputs.end();
  };

  const auto num_nodes = nodes.size();
  for (const auto i : c10::irange(num_nodes)) {
    auto* node = nodes[i];
    for (auto* input : node->inputs()) {
      auto* lifetime = getLifetime(input);
      if (!lifetime) {
        DCHECK(isUntrackedValue(input));
        continue;
      }
      DCHECK(lifetime->end <= i);
      lifetime->end = i;
    }
    for (auto* output : node->outputs()) {
      if (!alias_db.isMutableType(output)) {
        continue;
      }
      value_lifetimes_.emplace(output, Lifetime(i, i));
    }
  }
  for (auto* graph_output : graph->outputs()) {
    auto* lifetime = getLifetime(graph_output);
    if (!lifetime) {
      DCHECK(isUntrackedValue(graph_output));
      continue;
    }
    lifetime->end = num_nodes;
  }

  // Handle aliases. Aliases may extend a Value*'s lifetime. If a node
  // has an input and output that may alias each other, set the input's
  // lifetime end to max(input.lifetime_end, output.lifetime_end). Iterate
  // backwards to handle chains of aliases.
  for (const auto* node : graph->nodes().reverse()) {
    if (isPureFunction(node)) {
      // If the node is a pure function, it doesn't create any aliases,
      // so we can safely skip it.
      continue;
    }

    auto inputs = collectValuesWithTrackedLifetimes(node->inputs());
    auto outputs = collectValuesWithTrackedLifetimes(node->outputs());
    for (auto* input : inputs) {
      auto* input_lifetime = getLifetime(input);
      DCHECK(input_lifetime != nullptr);
      for (auto* output : outputs) {
        if (mayContainAlias(input, output, alias_db)) {
          auto* output_lifetime = getLifetime(output);
          DCHECK(output_lifetime != nullptr);
          input_lifetime->end =
              std::max(output_lifetime->end, input_lifetime->end);
        }
      }
    }
  }
  for (auto* managed_tensor : managed_tensor_values) {
    auto* lifetime = getLifetime(managed_tensor);
    DCHECK(lifetime && lifetime->end <= num_nodes);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* freeing_node;
    if (lifetime->end == num_nodes) {
      freeing_node = graph->return_node();
    } else {
      freeing_node = nodes[lifetime->end];
    }
    node_to_newly_free_tensors_[freeing_node].emplace_back(managed_tensor);
  }
}

bool ManagedTensorRanges::nodeFreesManagedTensors(Node* node) const {
  auto it = node_to_newly_free_tensors_.find(node);
  return it != node_to_newly_free_tensors_.end() && !it->second.empty();
}

const std::vector<const Value*>& ManagedTensorRanges::
    availableTensorValuesAfterNode(Node* node) const {
  return node_to_newly_free_tensors_.at(node);
}

bool ManagedTensorRanges::lifetimesOverlap(const Value* v1, const Value* v2)
    const {
  const auto* v1_lifetime = getLifetime(v1);
  const auto* v2_lifetime = getLifetime(v2);
  if (!v1_lifetime || !v2_lifetime) {
    return false;
  }

  if (v1_lifetime->start < v2_lifetime->start) {
    return v1_lifetime->end >= v2_lifetime->start;
  }
  return v2_lifetime->end >= v1_lifetime->start;
}

const ManagedTensorRanges::Lifetime* ManagedTensorRanges::getLifetime(
    const Value* value) const {
  auto it = value_lifetimes_.find(value);
  if (it != value_lifetimes_.end()) {
    return &it->second;
  }
  return nullptr;
}

ManagedTensorRanges::Lifetime* ManagedTensorRanges::getLifetime(
    const Value* value) {
  // const_cast is safe here, this is just a way to avoid code duplication
  // between the const/non-const versions of getLifetime.

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const auto* const_this = const_cast<const ManagedTensorRanges*>(this);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<ManagedTensorRanges::Lifetime*>(
      const_this->getLifetime(value));
}

std::vector<const Value*> ManagedTensorRanges::
    collectValuesWithTrackedLifetimes(at::ArrayRef<const Value*> values) {
  std::vector<const Value*> mutable_values;
  mutable_values.reserve(values.size());
  std::copy_if(
      values.begin(),
      values.end(),
      std::back_inserter(mutable_values),
      [this](const Value* value) { return getLifetime(value) != nullptr; });
  return mutable_values;
}

StaticModule::StaticModule(
    std::shared_ptr<torch::jit::Graph> g,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs)
    : StaticModule(
          PrepareForStaticModule(g->copy(), opts, std::move(sample_inputs)),
          opts) {}

StaticModule::StaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs)
    : StaticModule(
          PrepareForStaticModule(m, is_frozen, opts, std::move(sample_inputs)),
          opts) {}

StaticModule::StaticModule(
    std::pair<std::shared_ptr<torch::jit::Graph>, c10::optional<Module>>
        graph_and_module,
    const StaticModuleOptions& opts)
    : opts_(opts),
      graph_(std::move(graph_and_module.first)),
      module_(std::move(graph_and_module.second)),
      num_inputs_(graph_->inputs().size()) {
  // check opt flags
  if (opts.manage_output_tensors) {
    TORCH_CHECK(
        opts_.enable_out_variant,
        "When manage_output_tensors is true, enable_out_variant must be set to true");
  }
  if (opts_.optimize_memory) {
    TORCH_CHECK(
        opts_.enable_out_variant,
        "When optimize_memory is true, enable_out_variant must be set to true");
  }

  // handle schema
  if (module_.has_value()) {
    Method method = module_->get_method("forward");
    schema_ = method.function().getSchema();
    const auto num_schema_args = schema_->arguments().size();
    DCHECK(num_schema_args > 0);
    if (removeSelfFromGraphInput(graph_)) {
      module_ = c10::nullopt;
      num_inputs_ = num_schema_args - 1;
    }
  }

  // map Value* to its SSA definition IR
  FastMap<Value*, DefInfo> value_to_ssa_def;

  // N inputs map to the first N entries in storage
  for (const auto i : c10::irange(graph_->inputs().size())) {
    Value* input = graph_->inputs()[i];
    value_to_ssa_def[input] = std::make_pair(INPUT_VALUE, i);
  }

  {
    size_t nodes_size = 0, constants_size = 0;
    for (Node* node : graph_->nodes()) {
      ++(node->kind() == prim::Constant ? constants_size : nodes_size);
    }

    constants_.reserve(constants_size);
    functions_.reserve(nodes_size);
    nodes_.reserve(nodes_size);
  }

  // Create ProcessedFunction instances first to freeze their addresses to pass
  // to ProcessedNode.
  AliasDb alias_db(graph_, /*isFrozen=*/false);
  GRAPH_DEBUG("AliasDb: ", alias_db.toString());

  // Construct constant and function nodes
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      auto* v = node->output();
      TORCH_CHECK(v->type()->kind() != FunctionType::Kind);
      // construct SSA definition for constant nodes
      value_to_ssa_def[v] = std::make_pair(CONSTANT_VALUE, constants_.size());
      constants_.emplace_back(toIValue(v).value());
      continue;
    }

    // see [Check and correct bad schema alias info at runtime]
    bool check_outputs_for_overlap =
        !alias_db.mayContainAlias(node->inputs(), node->outputs()) &&
        containTensorsOnly(node->outputs());
    // new ProcessedFunction
    functions_.emplace_back(
        node, opts.enable_out_variant, check_outputs_for_overlap);
  }

  // construct SSA definition for non-constant nodes
  int node_idx = 0;
  FastMap<Node*, bool> node_has_out_variant;

  const auto inputs_index_offset = inputs_offset();
  const auto constants_index_offset = constants_offset();
  const auto values_index_offset = intermediate_values_offset();

  // Map node_idx to index offset in values_. Can't reserve space
  // because we don't know how many non-constant nodes there are yet.
  std::vector<uint32_t> node_output_idx_map;
  uint32_t node_outputs_seen_so_far = 0;
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }
    // Assign memory for the outputs
    const auto outputs_offset_for_node =
        node_outputs_seen_so_far + values_index_offset;
    TORCH_CHECK(
        outputs_offset_for_node < (1 << 16),
        "outputs offset in values table",
        outputs_offset_for_node,
        " would overflow 2-byte index storage");
    node_output_idx_map.push_back(outputs_offset_for_node);
    node_outputs_seen_so_far += node->outputs().size();
  }

  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }
    ProcessedNodeInputs input_indices(node->inputs().size());
    std::vector<DefInfo> input_ssa_defs;
    for (const auto input_idx : c10::irange(node->inputs().size())) {
      Value* const input = node->inputs()[input_idx];
      int inner_node_idx = 0;
      int out_idx = 0;
      std::tie(inner_node_idx, out_idx) = value_to_ssa_def.at(input);
      unsigned int input_ivalue_idx = 0;
      if (inner_node_idx == StaticModule::INPUT_VALUE) {
        input_ivalue_idx = out_idx + inputs_index_offset;
      } else if (inner_node_idx == StaticModule::CONSTANT_VALUE) {
        input_ivalue_idx = out_idx + constants_index_offset;
      } else {
        DCHECK_GE(inner_node_idx, 0);
        const auto global_value_idx =
            node_output_idx_map[inner_node_idx] + out_idx;
        if (inner_node_idx < node_output_idx_map.size() - 1) {
          DCHECK_LT(global_value_idx, node_output_idx_map[inner_node_idx + 1]);
        } else {
          DCHECK_LT(
              global_value_idx,
              constants_index_offset + node_outputs_seen_so_far);
        }
        input_ivalue_idx = global_value_idx;
      }
      TORCH_CHECK(
          input_ivalue_idx < (1 << 16),
          "input index in values table ",
          input_ivalue_idx,
          " would overflow 2-byte index storage");
      input_indices[input_idx] = input_ivalue_idx;
    }

    ProcessedFunction* fn = &functions_[node_idx];
    // create a new ProcessedNode
    // see [Check and correct bad schema alias info at runtime]
    bool check_outputs_for_overlap =
        !alias_db.mayContainAlias(node->inputs(), node->outputs()) &&
        containTensorsOnly(node->outputs());
    nodes_.emplace_back(
        node, fn, std::move(input_indices), node_output_idx_map[node_idx]);

    node_has_out_variant.emplace(node, nodes_.back().has_out_variant());
    for (const auto i : c10::irange(node->outputs().size())) {
      value_to_ssa_def[node->outputs()[i]] = std::make_pair(node_idx, i);
    }
    node_idx++;
  }

  num_intermediate_values_ = std::accumulate(
      nodes_.begin(),
      nodes_.end(),
      0,
      [](uint32_t sum, const ProcessedNode& pnode) {
        return sum + pnode.num_outputs();
      });

  for (auto& pnode : nodes_) {
    if (pnode.num_outputs() == 1 &&
        isOptimizableContainerType(pnode.node(), node_has_out_variant)) {
      node_is_optimizable_container_type_.emplace(pnode.node());
    }
  }
  output_indices_.reserve(graph_->outputs().size());
  for (auto output : graph_->outputs()) {
    int node_idx = 0;
    int out_idx = 0;
    std::tie(node_idx, out_idx) = value_to_ssa_def[output];
    uint32_t output_index = 0;
    if (node_idx == StaticModule::INPUT_VALUE) {
      output_index = out_idx + inputs_index_offset;
    } else if (node_idx == StaticModule::CONSTANT_VALUE) {
      output_index = constants_index_offset + out_idx;
    } else {
      output_index = nodes_[node_idx].output_ivalue_index(out_idx);
    }
    TORCH_CHECK(
        output_index < (1 << 16),
        "output index ",
        output_index,
        " would overflow 2-byte index storage");
    output_indices_.emplace_back(output_index);
  }

  // Prepare for memory planning
  value_group_.init(graph_, alias_db);
  GRAPH_DEBUG(value_group_.toString());

  prepareForMemoryPlanner();
}

void StaticModule::prepareForMemoryPlanner() {
  if (!opts_.enable_out_variant) {
    return;
  }

  // Never manage graph outputs so that we can do std::move(output_ivalue).
  // This does not affect performance if the graph returns a collection object.
  FastSet<const Value*> graph_output_values(
      graph_->outputs().begin(), graph_->outputs().end());

  // collect register indices of outputs of ops with out variant
  for (ProcessedNode& pnode : nodes_) {
    if (!pnode.has_out_variant()) {
      continue;
    }
    auto outputs = pnode.node()->outputs();
    for (const auto i : c10::irange(outputs.size())) {
      const Value* out_v = outputs[i];
      // Types are stored in the underlying TorchScript IR
      bool is_tensor_type = out_v->type()->castRaw<TensorType>();
      if (opts_.manage_output_tensors && is_tensor_type &&
          graph_output_values.find(out_v) == graph_output_values.end() &&
          value_group_.isOutputAlias(out_v)) {
        managed_output_tensor_values_.insert(out_v);
        continue;
      }
      if (value_group_.isAlwaysAlive(out_v)) {
        continue;
      }
      if (is_tensor_type) {
        managed_tensor_values_.insert(out_v);
      } else if (is_optimizable_container_type(pnode.node())) {
        // We "leak" certain container types because their allocations
        // take a long time
        leaked_values_.insert(out_v);
      }
    }
  }

  for (const Value* output : graph_->outputs()) {
    managed_tensor_values_.erase(output);
  }
  GRAPH_DEBUG("managed_tensor_values: ", dumpValueSet(managed_tensor_values_));
  GRAPH_DEBUG(
      "managed_output_tensor_values_: ",
      dumpValueSet(managed_output_tensor_values_));

  managed_tensor_ranges_ = ManagedTensorRanges(graph_, managed_tensor_values_);
}

const StaticModuleOptions& StaticModule::opts() const {
  return opts_;
}

size_t StaticModule::num_outputs() const {
  return graph_->outputs().size();
}

size_t StaticModule::num_inputs() const {
  return num_inputs_;
}

StaticRuntime& StaticModule::runtime() {
  if (!cached_runtime_) {
    cached_runtime_ = std::make_unique<StaticRuntime>(*this);
  }
  return *cached_runtime_;
}

Node* StaticModule::findNodeWithKindForTesting(const std::string& kind) const {
  for (auto& pnode : nodes()) {
    if (pnode.node()->kind().toQualString() == kind) {
      return pnode.node();
    }
  }
  return nullptr;
}

c10::IValue StaticModule::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
  return runtime()(args, kwargs);
}

c10::IValue StaticModule::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
  return runtime()(std::move(args), kwargs);
}

StaticRuntime::StaticRuntime(const StaticModule& sm)
    : static_module_(sm),
      first_input_is_self_(static_module_.first_input_is_self()),
      manage_output_tensors_enabled_(sm.opts().manage_output_tensors),
      nodes_(sm.nodes()) {
  values_.resize(sm.total_num_values());
  const auto constants_index_offset = sm.constants_offset();
  const auto constants_begin_it = values_.begin() + constants_index_offset;
  const auto constants_end_it = constants_begin_it + sm.constants().size();
  std::copy(sm.constants().begin(), sm.constants().end(), constants_begin_it);

  for (const auto idx : c10::irange(sm.nodes().size())) {
    auto& n = nodes_[idx];
    n.set_values(values_.data());
  }

  // TODO: can we convert outputs_ to store indices?
  for (auto index : sm.output_indices()) {
    outputs_.emplace_back(&values_[index]);
  }
}

StaticRuntime::~StaticRuntime() = default;

void StaticRuntime::set_arg(const size_t idx, std::vector<IValue>&& args) {
  DCHECK(idx < args.size());
  Input(idx + first_input_is_self_) = std::move(args[idx]);
}

void StaticRuntime::set_arg(const size_t idx, const std::vector<IValue>& args) {
  DCHECK(idx < args.size());
  Input(idx + first_input_is_self_) = args[idx];
}

void StaticRuntime::set_arg(const size_t idx, const IValue& arg) {
  Input(idx + first_input_is_self_) = arg;
}

namespace {
void check_type(const Argument& schema_arg, const IValue& arg) {
  // Fast path for most common case
  if (arg.isTensor() &&
      schema_arg.type()->kind() == c10::TypeKind::TensorType) {
    return;
  }
  TORCH_CHECK(arg.type()->isSubtypeOf(schema_arg.type()));
}
} // namespace

template <typename IValueList>
void StaticRuntime::set_inputs(
    IValueList&& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  const auto total_num_inputs =
      args.size() + kwargs.size() + first_input_is_self_;
  TORCH_CHECK(total_num_inputs == static_module_.num_inputs());

  const auto& schema = static_module_.schema();
  if (first_input_is_self_) {
    Input(0) = static_module_.module()._ivalue();
  }

  if (C10_UNLIKELY(!schema)) {
    TORCH_CHECK(
        kwargs.empty(),
        "Schema is not available, but StaticRuntime got kwargs. "
        "Consider creating the Static Runtime instance "
        "with StaticModule(const torch::jit::Module& m) instead.");
    for (size_t i = 0; i < args.size(); ++i) {
      set_arg(i, std::forward<IValueList>(args));
    }
    return;
  }

  const auto& schema_args = schema->arguments();
  size_t consumed_kwargs = 0;
  DCHECK(schema_args.size() > 0);

  for (size_t i = 0; i < schema_args.size() - 1; ++i) {
    // Start at 1 since the schema always contains `self`.
    const auto& schema_arg = schema_args[i + 1];

    if (i < args.size()) {
      check_type(schema_arg, args[i]);
      set_arg(i, std::forward<IValueList>(args));
      continue;
    }

    auto it = kwargs.find(schema_arg.name());
    if (it != kwargs.end()) {
      check_type(schema_arg, it->second);
      set_arg(i, it->second);
      ++consumed_kwargs;
      continue;
    }

    auto maybe_default_val = schema_arg.default_value();
    if (maybe_default_val) {
      set_arg(i, *maybe_default_val);
      continue;
    }

    TORCH_CHECK(
        false, "Static runtime is missing required kwarg ", schema_arg.name());
  }
  TORCH_CHECK(
      consumed_kwargs == kwargs.size() &&
      args.size() + consumed_kwargs == schema_args.size() - 1);
}

void StaticRuntime::create_memory_planner() {
  if (!planner_) {
    planner_ = std::make_unique<MemoryPlanner>(
        this,
        static_module_.value_group(),
        static_module_.managed_tensor_values(),
        static_module_.managed_output_tensor_values(),
        static_module_.leaked_values(),
        static_module_.managed_tensor_ranges(),
        static_module_.opts().enable_out_variant,
        manage_output_tensors_enabled_,
        static_module_.opts().optimize_memory);
  }
}

namespace {

void destroyNodeOutputs(ProcessedNode& p_node) {
  const auto borrows_outputs = borrowsOutputs(p_node.node()->kind());
  for (const auto i : c10::irange(p_node.num_outputs())) {
    auto& output = p_node.Output(i);
    if (doesNotHeapAllocateWhenStoredInIValue(*output.type())) {
      continue;
    }

    if (borrows_outputs) {
      // NB: No need to incref here. This codepath is only hit if the run didn't
      // finish, so we shouldn't be returning anything to the client.
      c10::MaybeOwnedTraits<IValue>::destroyBorrow(output);
    } else {
      output = IValue();
    }
  }
}

} // namespace

void StaticRuntime::clean_up_intermediate_ivalues() noexcept {
  for (auto& p_node : nodes_) {
    destroyNodeOutputs(p_node);
  }
}

void StaticRuntime::resetMemory() noexcept {
  planner_.reset();
  clean_up_input_ivalues();
  clean_up_intermediate_ivalues();
}

c10::IValue StaticRuntime::move_outputs_to_tuple(uint32_t num_outputs) {
  switch (num_outputs) {
    case 1:
      return c10::ivalue::Tuple::create(IValue(std::move(*outputs_[0])));
    case 2:
      return c10::ivalue::Tuple::create(
          IValue(std::move(*outputs_[0])), IValue(std::move(*outputs_[1])));
    case 3:
      return c10::ivalue::Tuple::create(
          IValue(std::move(*outputs_[0])),
          IValue(std::move(*outputs_[1])),
          IValue(std::move(*outputs_[2])));
    default: {
      std::vector<c10::IValue> outputs;
      outputs.reserve(num_outputs);
      for (const auto i : c10::irange(num_outputs)) {
        // use move here. Otherwise, clean up outputs_[i] explicitly
        outputs.emplace_back(std::move(*outputs_[i]));
      }
      return c10::ivalue::Tuple::create(std::move(outputs));
    }
  }
}

/// [Check and correct bad schema alias info at runtime]
/// Static runtime relies on the operator schema's alias info to be correct for
/// memory planning. Because it's hard to enforce the alias info to be correct,
/// we need to do runtime detection for accidental aliases that do not comply
/// with the schema. Only aliases of managed tensors are problematic. To avoid
/// runtime crashes, we can add runtime detection and force the op to comply
/// with its schema by cloning the alias. Because all managed tensors' data_ptrs
/// are part of the internal buffer that the MemoryPlanner allocates, we can
/// check aliases by checking the memory overlap with this internal buffer. But
/// a tensor's storage can be resized during inferenceso we need another way to
/// handle the resized case.
///
/// There are two ways for incorrect schema to break memory planning. Let's look
/// at two examples:
///
/// Example 1:
/// @code
///   def forward(x):
///     a = x + x
///     b = bad_op(a)  # b ends up aliasing a incorrectly
///     return (b)
/// @endcode
/// bad_op: its schema says it returns a new Tensor, but it actually returns an
/// alias. In this case, the memory planner would recognize `a` as a managed
/// tensor and clean up its memory before returning `b`. But `b` is actually an
/// alias of `a`, when `a`'s data_ptr get reset, `b`'s data_ptr gets reset too.
///
/// Example 2:
/// @code
///   def forward(x):
///     a = x + x
///     a2 = bad_op(a) # a2 ends up alias a incorrectly
///     b = a + a
///     c = b * b # c shares storage with a
///     d = c + 2 # d shares storage with b
///     e = a2 * a2
///     return (d, e)
/// @endcode
/// With the memory reuse algorithm, `c` could end up sharing storage with `a`,
/// but because of bad_op, `a2` now aliases `a`. `c` overwrites `a` and
/// therefore `a2`, leading to the wrong results. We solve this problem with two
/// steps. Note this doesn't happen with the current memory reuse algorithm
/// because of the way it's implemented. Things could change with a different
/// implementation.
///
/// Step 1, annotate the ProcessedNodes with a flag `check_memory_overlap_` set
/// to true if its outputs do not alias its inputs as indicated by the AliasDb
/// and all of its outputs are Tensors. Then at runtime, we check that the
/// nodes' output tensors do not overlap with the internal buffer that the
/// MemoryPlanner allocates. For latency concerns, we only run this check for
/// fallback ops. The schemas of native ops and out variants are vetted and
/// enforced with static runtime unit tests. For the first iteration, we do a
/// full memory overlap check with
/// ProcessedNode::verify_and_correct_memory_overlap() because the internal
/// buffer doesn't exist yet.
///
/// Step 2, if a managed tensor gets resized during inference, it gets a new
/// data_ptr which is not from the buffer. We can tackle this corner case by
/// delaying the deallocation of the managed tensors to after the outputs are no
/// longer used (essentially merging the internal/output buffers into one).
/// Before the merging is implemented, we add another flag `overlap_detected_`
/// to flag any node with overlap detected in Step 1 and do a full memory
/// overlap check if the fast check (by checking memory overlap with internal
/// buffer) fails. There is still a corner case that fails with the added flag.
/// If a resize is triggered at the same time as the op creating an alias at the
/// same time, the current checks would fail to detect the alias.
void StaticRuntime::verify_and_correct_memory_overlap(ProcessedNode& n) {
  // The slow check can be removed once the internal/output buffers are merged
  if (C10_UNLIKELY(n.check_outputs_for_memory_overlap())) {
    if (C10_UNLIKELY(!planner_)) {
      // slow check, for first iter only
      n.verify_and_correct_memory_overlap();
    } else {
      bool overlap_detected_with_fast_check = false;
      for (size_t i = 0; i < n.outputs().size(); i++) {
        auto& output = n.Output(i);
        if (output.isTensor()) {
          overlap_detected_with_fast_check |=
              fast_check_and_correct_overlap_with(n, output);
        } else if (output.isTensorList()) {
          auto tensor_list = output.toListRef();
          for (auto& ival : tensor_list) {
            overlap_detected_with_fast_check |=
                fast_check_and_correct_overlap_with(
                    n,
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<c10::IValue&>(ival));
          }
        }
      }
      if (n.outputs_memory_overlap_detected() &&
          !overlap_detected_with_fast_check) {
        // slow check. Only run when the fast check fails.
        n.verify_and_correct_memory_overlap();
      }
    }
  }
}

bool StaticRuntime::fast_check_and_correct_overlap_with(
    ProcessedNode& n,
    c10::IValue& tensor_ival) {
  auto& tensor = tensor_ival.toTensor();
  if (planner_->overlapWithInternalBuffer(tensor.data_ptr())) {
    DLOG(INFO) << "Detected alias for node: " << PrintNode(n.node());
    tensor_ival = at::native::clone(tensor, c10::nullopt);
    n.set_outputs_memory_overlap_detected();
    return true;
  }
  return false;
}

StaticRuntime::Deallocator::~Deallocator() {
  // Assume cleanup cannot throw.
  cleanupImpl();
#ifndef NDEBUG
  runtime_.check_for_memory_leak(/*output_returned*/ false);
#endif
}

void StaticRuntime::Deallocator::cleanupImpl() {
  // MemoryPlanner is created after the first invocation of `run()`. This
  // is done intentionally because MemoryPlanner uses `Tensor` sizes of
  // the previous `run()` for memory planning of subsequent runs
  if (C10_LIKELY(finished_)) {
    runtime_.create_memory_planner();
  }

  if (C10_LIKELY(runtime_.planner_)) {
    runtime_.planner_->deallocate();
  } else {
    // This is the first run, and it didn't finish, so we can't use a
    // `MemoryPlanner` to deallocate stuff. Just reset everything mannually.
    runtime_.resetMemory();
  }
  // clean up owning refs of input tensors
  runtime_.clean_up_input_ivalues();
  if (C10_UNLIKELY(!finished_)) {
    runtime_.deallocateOutputTensors();
  }
}

template <typename IValueList>
c10::IValue StaticRuntime::run_impl(
    IValueList&& args,
    const KeywordArgs& kwargs) {
  // We assume inference workloads, so we do not need
  // autograd. Enabling this is a significant win on dispatcher
  // overhead because it saves a round of dispatch for at least some
  // functions, such as resize_ and resize_as_.
  c10::InferenceMode mode;

  {
    auto on_exit = Deallocator(*this);

    if (planner_) {
      DCHECK(!manage_output_tensors_enabled_ || checkOutputTensorMemoryLeaks());
      planner_->allocate();
    }

    set_inputs(std::forward<IValueList>(args), kwargs);

    for (auto& n : nodes_) {
      // LOG(INFO) << "Running node: " << PrintNode(n.node());
      n.run();
      // Check for incorrect schema alias info.
      verify_and_correct_memory_overlap(n);
    }
    on_exit.setFinished();
  }

  // no need to keep references of outputs in static runtime anymore
  if (static_module_.num_outputs() > 1) {
    return move_outputs_to_tuple(static_module_.num_outputs());
  }

  DCHECK(check_for_memory_leak(/*output_returned*/ false));

  // use move here. Otherwise, clean up outputs_[0] explicitly
  return std::move(*outputs_[0]);
}

template <typename IValueList>
c10::IValue StaticRuntime::run_impl_record_functions(
    IValueList&& args,
    const KeywordArgs& kwargs) {
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    at::RecordFunction guard(
        at::RecordScope::TORCHSCRIPT_FUNCTION, pre_sampled);
    if (guard.isActive()) {
      if (guard.needsInputs()) {
        guard.before("forward", &args);
      } else {
        guard.before("forward");
      }
    }
    return run_impl(std::forward<IValueList>(args), kwargs);
  }
  return run_impl(std::forward<IValueList>(args), kwargs);
}

c10::IValue StaticRuntime::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  return run_impl(args, kwargs);
#else
  return run_impl_record_functions(args, kwargs);
#endif
}

c10::IValue StaticRuntime::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  return run_impl(std::move(args), kwargs);
#else
  return run_impl_record_functions(std::move(args), kwargs);
#endif
}

namespace {

std::string generate_latency_json(const std::string& label, double millis) {
#ifdef FBCODE_CAFFE2
  folly::dynamic json = folly::dynamic::object();
  json["type"] = label;
  json["metric"] = "latency";
  json["unit"] = "ms";
  json["value"] = millis;
  return "PyTorchObserver " + folly::toJson(json);
#else
  return "";
#endif
}

} // namespace

void StaticRuntime::benchmark(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const int warmup_runs,
    const int main_runs,
    bool print_per_node_time,
    bool generate_ai_pep_output) {
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());
  std::cout << "Input size: " << args_list.size() << std::endl;
  if (args_list.size() == 0) {
    return;
  }
  float time_per_iter =
      benchmark_model(args_list, kwargs_list, warmup_runs, main_runs);
  std::cout << "Static runtime ms per iter: " << time_per_iter
            << ". Iters per second: " << 1000.0 / time_per_iter << std::endl;

  IndividualMetrics results =
      benchmark_individual_ops(args_list, kwargs_list, warmup_runs, main_runs);

  if (print_per_node_time) {
    for (const auto i : c10::irange(nodes_.size())) {
      const Node* node = nodes_[i].node();
      std::cout << "Node #" << i << ": " << results.time_per_node[i]
                << " ms/iter, ";
      node->print(std::cout, 0, nullptr, false);
    }
  }

  std::vector<std::pair<std::string, double>> time_per_node_type_vec{
      results.time_per_node_type.begin(), results.time_per_node_type.end()};
  std::sort(
      time_per_node_type_vec.begin(),
      time_per_node_type_vec.end(),
      [](auto& left, auto& right) { return left.second > right.second; });

  std::cout << "Time per node type:" << std::endl;
  for (const auto& p : time_per_node_type_vec) {
    const std::string& kind = p.first;
    const double ms = p.second;
    std::cout << std::setw(15) << ms << " ms. " << std::setw(10)
              << results.percent_per_node_type[kind] << "%. " << kind << " ("
              << results.instances_per_node_type[kind] << " nodes";
    if (results.out_nodes.count(kind)) {
      std::cout << ", out variant)" << std::endl;
    } else if (results.native_nodes.count(kind)) {
      std::cout << ", native)" << std::endl;
    } else {
      std::cout << ")" << std::endl;
    }

    if (generate_ai_pep_output) {
      LOG(INFO) << generate_latency_json(kind, ms);
    }
  }
  if (generate_ai_pep_output) {
    LOG(INFO) << generate_latency_json(
        "static_runtime_first_iter", results.first_iter_time);
  }
  std::cout << std::setw(15) << results.total_time << " ms. in Total"
            << std::endl;
  std::cout << "StaticRuntime setup time: " << results.setup_time << " ms"
            << std::endl;
  std::cout << "Memory allocation time: " << results.memory_alloc_time
            << " ms\n";
  std::cout << "Memory deallocation time: " << results.memory_dealloc_time
            << " ms" << std::endl;
  std::cout << "Outputs deallocation time: " << results.output_dealloc_time
            << " ms" << std::endl;
  std::cout << "First iter time: " << results.first_iter_time << " ms"
            << std::endl;
  std::cout << "Number of operators: " << nodes_.size() << std::endl;

  if (planner_) {
    std::cout << "Total number of managed tensors: "
              << planner_->total_num_managed_tensors() << std::endl;
    std::cout << "Total number of managed output tensors: "
              << planner_->total_num_managed_output_tensors() << std::endl;
    std::cout << "Total number of unmanaged values: "
              << planner_->total_num_unmanaged() << std::endl;
    std::cout << "Number of unmanaged values requiring cleanup: "
              << planner_->num_unmanaged_non_scalars() << std::endl;
    std::cout << "Number of unmanaged values not requiring cleanup: "
              << planner_->num_unmanaged_scalars() << std::endl;
    std::cout << "Total memory managed: " << planner_->total_managed()
              << " bytes" << std::endl;
    if (static_module_.opts().optimize_memory) {
      std::cout << "Total number of reused tensors: "
                << planner_->total_reused_tensors() << std::endl;
    }
    std::cout << "Total number of 'out' variant nodes/total number of nodes: "
              << results.out_nodes_count << "/" << results.total_nodes_count
              << " ("
              << 100.0 * (results.out_nodes_count) /
            static_cast<float>(results.total_nodes_count)
              << "%)" << std::endl;
  }
  check_for_memory_leak();

#ifndef NDEBUG
  KeywordArgs empty_kwargs;
  display_nodes(
      args_list[0], kwargs_list.size() > 0 ? kwargs_list[0] : empty_kwargs);
#endif
}

float StaticRuntime::benchmark_model(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 0 && main_runs >= 1);
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());

  const bool is_kwargs_empty = kwargs_list.size() == 0;
  const KeywordArgs empty_kwargs;
  for (const auto i : c10::irange(warmup_runs)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      if (manage_output_tensors_enabled_) {
        deallocateOutputTensors();
      }
    }
  }
  caffe2::Timer timer;
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      if (manage_output_tensors_enabled_) {
        deallocateOutputTensors();
      }
    }
  }
  float millis = timer.MilliSeconds();
  return millis / (static_cast<float>(main_runs) * args_list.size());
}

bool display_ivalue(const IValue& iv) {
  if (iv.isTensor()) {
    std::cout << "Tensor " << iv.toTensor().toString() << " {";
    for (const auto i : c10::irange(iv.toTensor().sizes().size())) {
      std::cout << iv.toTensor().sizes()[i];
      if (iv.toTensor().sizes().size() > i + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}\n";
    return true;
  } else if (iv.isTensorList()) {
    std::cout << "TensorList {" << iv.toTensorList().size() << "}\n";
    return true;
  } else if (iv.isGenericDict()) {
    std::cout << "Dict {" << iv.toGenericDict().size() << "}\n";
    return true;
  } else if (iv.isTuple()) {
    std::cout << "Tuple {" << iv.toTupleRef().elements().size() << "}\n";
    return true;
  } else if (iv.isInt()) {
    std::cout << "int {" << iv.toInt() << "}\n";
    return true;
  } else if (iv.isBool()) {
    std::cout << "bool {" << iv.toBool() << "}\n";
    return true;
  } else if (iv.isDouble()) {
    std::cout << "double {" << iv.toDouble() << "}\n";
    return true;
  }
  return false;
}

void display_pnode_info(const ProcessedNode& pnode) {
  pnode.node()->print(std::cout, 0, nullptr, false);
  for (const auto i : c10::irange(pnode.num_inputs())) {
    std::cout << "\ti" << i << ": ";
    if (!display_ivalue(pnode.Input(i))) {
      std::cout << *(pnode.node()->inputs()[i]->type()) << '\n';
    }
  }
  const auto outputs = pnode.outputs();
  for (const auto i : c10::irange(outputs.size())) {
    std::cout << "\to" << i << ": ";
    if (!display_ivalue(outputs[i])) {
      std::cout << *(pnode.node()->outputs()[i]->type()) << '\n';
    }
  }
}

void StaticRuntime::display_nodes(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
  c10::InferenceMode mode;

  auto on_exit = Deallocator(*this);

  if (planner_) {
    planner_->allocate();
  }
  set_inputs(args, kwargs);

  for (auto& node : nodes_) {
    node.run();
    display_pnode_info(node);
  }
  on_exit.setFinished();
}

StaticRuntime::IndividualMetrics StaticRuntime::benchmark_individual_ops(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());
  TORCH_CHECK(warmup_runs >= 1 && main_runs >= 1);
  if (args_list.size() == 0) {
    return {};
  }

  const bool is_kwargs_empty = kwargs_list.size() == 0;
  const KeywordArgs empty_kwargs;
  bool manage_output_tensors = static_module_.opts().manage_output_tensors;
  // See comment on above use of InferenceMode for
  // explanation.
  c10::InferenceMode mode;

  IndividualMetrics results;
  results.time_per_node.resize(nodes_.size(), 0);

  // setup time
  caffe2::Timer timer;

  set_inputs(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);

  results.setup_time = timer.MilliSeconds();

  // The first iteration profiles each node's output Tensors' sizes and
  // initializes the memory planner with the profile information. Folllowing
  // iterations just use the already established memory planning.
  timer.Start();
  operator()(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);
  if (manage_output_tensors) {
    deallocateOutputTensors();
  }
  results.first_iter_time = timer.MilliSeconds();

  // warmup runs
  for (const auto i : c10::irange(warmup_runs - 1)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
    }
  }

  // main runs
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning

    for (const auto j : c10::irange(args_list.size())) {
      set_inputs(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);

      timer.Start();
      if (planner_) {
        planner_->allocate();
      }
      float millis = timer.MilliSeconds();
      results.memory_alloc_time += millis;

      for (const auto k : c10::irange(nodes_.size())) {
        timer.Start();
        nodes_[k].run();
        millis = timer.MilliSeconds();
        results.time_per_node[k] += millis;
        verify_and_correct_memory_overlap(nodes_[k]);
      }
      timer.Start();
      create_memory_planner();
      planner_->deallocate();
      // clean up owning refs of input tensors
      clean_up_input_ivalues();
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
      millis = timer.MilliSeconds();
      results.memory_dealloc_time += millis;

      timer.Start();
      // no need to keep references of outputs in static runtime anymore
      c10::IValue output;
      if (static_module_.num_outputs() > 1) {
        output = move_outputs_to_tuple(static_module_.num_outputs());
      }

      DCHECK(check_for_memory_leak(/*output_returned*/ false));

      // use move here. Otherwise, clean up outputs_[0] explicitly
      output = std::move(*outputs_[0]);
      // release outputs explicitly to measure the time it takes
      output = IValue();
      millis = timer.MilliSeconds();
      results.output_dealloc_time += millis;
    }
  }

  // post processing
  const float num_total_iters =
      (static_cast<float>(main_runs) * args_list.size());
  for (const auto i : c10::irange(nodes_.size())) {
    const Node* node = nodes_[i].node();
    std::string kind = std::string(node->kind().toQualString());
    results.time_per_node[i] /= num_total_iters;
    results.time_per_node_type[kind] += results.time_per_node[i];
    results.instances_per_node_type[kind]++;
    if (nodes_[i].has_out_variant()) {
      results.out_nodes.insert(kind);
      results.out_nodes_count++;
    } else if (nodes_[i].has_native()) {
      results.native_nodes.insert(kind);
    }
    results.total_time += results.time_per_node[i];
  }
  results.total_nodes_count = nodes_.size();
  results.memory_alloc_time /= num_total_iters;
  results.memory_dealloc_time /= num_total_iters;
  results.output_dealloc_time /= num_total_iters;
  for (const auto& p : results.time_per_node_type) {
    const std::string& kind = p.first;
    results.percent_per_node_type[kind] = p.second / results.total_time * 100;
  }
  return results;
}

bool StaticRuntime::check_for_memory_leak(bool output_returned) {
  // check for inputs
  for (const auto i : c10::irange(static_module_.num_inputs())) {
    TORCH_CHECK(values_[i].isNone(), "Input ", i, " was not cleaned up");
  }
  FastSet<const IValue*> output_ivalues(outputs_.begin(), outputs_.end());
  for (const auto n : c10::irange(nodes_.size())) {
    auto& pnode = nodes_[n];
    for (const auto i : c10::irange(pnode.num_outputs())) {
      const IValue* ival = &pnode.Output(i);
      const Value* val = pnode.node()->output(i);
      // subtlety: isManagedOutputTensorValue may give a false
      // negative here if an output is an alias of this value, so
      // check the actual tensor!
      if (planner_ &&
          (isManagedOutputTensor(*ival) || isManagedOutputTensorValue(val))) {
        // `ival` contains a managed output tensor that the runtime doesn't
        // reclaim at the end of an iteration, but the client does so
        // by explicitly calling `StaticRuntime::deallocateOutputTensors`.
        continue;
      }
      const std::string error_msg = "Output " + c10::to_string(i) + ", %" +
          val->debugName() + " of node " + c10::to_string(n) +
          " which has kind " + pnode.node()->kind().toQualString() +
          " was not cleaned up";
      if (output_ivalues.count(ival) == 0) {
        // check for intermediates
        if (!ival->isNone()) {
          TORCH_CHECK(
              ival->isTensor() ||
                  static_module_.is_optimizable_container_type(pnode.node()) ||
                  doesNotHeapAllocateWhenStoredInIValue(*val->type()),
              error_msg);
          if (ival->isTensor()) {
            const auto& t = ival->toTensor();
            if (t.defined()) {
              auto* storage_impl = t.storage().unsafeGetStorageImpl();
              TORCH_CHECK(
                  storage_impl->data() == nullptr ||
                      (planner_ &&
                       planner_->isManagedStorageImpl(storage_impl)),
                  error_msg);
            }
          }
        }
      } else {
        // check for outputs
        if (output_returned) {
          TORCH_CHECK(ival->isNone(), error_msg);
        }
      }
    }
  }
  VLOG(1) << "Finished checking for memory leak";
  return true;
}

void StaticRuntime::deallocateOutputTensors() {
  if (!static_module_.opts().manage_output_tensors) {
    TORCH_CHECK(
        !planner_ || planner_->numOutputBufferBytes() == 0,
        "manage_output_tensors is disabled, but output tensor buffer is not empty.");
    return;
  }
  if (planner_) {
    planner_->deallocateOutputTensors();
    DCHECK(checkOutputTensorMemoryLeaks());
  }
}

bool StaticRuntime::checkOutputTensorMemoryLeaks() {
  if (!static_module_.opts().manage_output_tensors || !planner_) {
    return true;
  }
  for (const auto n : c10::irange(nodes_.size())) {
    auto& pnode = nodes_[n];
    for (const auto i : c10::irange(pnode.num_outputs())) {
      const IValue* ival = &pnode.Output(i);
      const Value* val = pnode.node()->output(i);
      if (!isManagedOutputTensorValue(val)) {
        continue;
      }
      const auto& t = ival->toTensor();
      if (t.defined()) {
        auto* storage_impl = t.storage().unsafeGetStorageImpl();
        const std::string error_msg = "Output " + c10::to_string(i) + ", %" +
            val->debugName() + " of node " + c10::to_string(n) +
            " was not cleaned up";
        TORCH_CHECK(storage_impl->data() == nullptr, error_msg);
      }
    }
  }
  VLOG(1) << "Finished checking for memory leak from output tensors";
  return true;
}

bool StaticRuntime::isManagedOutputTensor(const IValue& ivalue) const {
  return planner_ && planner_->isManagedOutputTensor(ivalue);
}

bool StaticRuntime::isManagedOutputTensorValue(const Value* value) const {
  // It's possible that manage_output_tensors_ was disabled after initializing
  // managed_output_tensor_values, so we have to check that flag here.
  if (!planner_ || !manage_output_tensors_enabled_) {
    return false;
  }
  const auto& managed_outputs = static_module_.managed_output_tensor_values();
  return managed_outputs.find(value) != managed_outputs.end();
}

void StaticRuntime::disableManageOutputTensors() {
  if (!manage_output_tensors_enabled_) {
    return;
  }
  manage_output_tensors_enabled_ = false;
  if (!planner_) {
    return;
  }
  // Reset all IValues and destruct planner_ so that it can be reconstructed in
  // the next run.
  for (auto& n : nodes_) {
    for (const auto i : c10::irange(n.outputs().size())) {
      n.Output(i) = IValue();
    }
  }
  planner_.reset();
}

ProcessedFunction::ProcessedFunction(
    Node* node,
    bool enable_out_variant,
    bool check_memory_overlap)
    : check_memory_overlap_(check_memory_overlap) {
  if (enable_out_variant) {
    f_ = getOutOfPlaceOperation(node);
    if (f_) {
      kind_ = ProcessedFunction::Kind::kOutVariant;
      // do not check memory overlap for out variants
      check_memory_overlap_ = false;
      VLOG(1) << "Switch to out variant for node: " << PrintNode(node);
      return;
    }
  }
  {
    f_ = getNativeOperation(node);
    if (f_) {
      kind_ = ProcessedFunction::Kind::kNativeFunction;
#ifdef NDEBUG
      // skip this check in opt mode because these ops are better vetted
      check_memory_overlap_ = false;
#endif
      VLOG(1) << "Switch to native impl for node: " << PrintNode(node);
      return;
    }
  }
  {
    const Operator& op = node->getOperator();
    f_ = [node_op = op.getOperation(node),
          has_var_args = hasVarArgs(node)](ProcessedNode* pnode) mutable {
      std::vector<IValue> stack;
      const size_t size = pnode->num_inputs();
      stack.reserve(size + has_var_args);
      for (const auto i : c10::irange(size)) {
        stack.emplace_back(pnode->Input(i));
      }
      // Need to store the number of inputs in stack for variadic ops.
      if (has_var_args) {
        stack.emplace_back(static_cast<int>(size));
      }
      node_op(stack);
      DCHECK_EQ(stack.size(), pnode->num_outputs());
      for (const auto i : c10::irange(pnode->num_outputs())) {
        pnode->Output(i) = std::move(stack[i]);
      }
    };
    kind_ = ProcessedFunction::Kind::kInterpreterFallback;
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

ProcessedNode::ProcessedNode(
    Node* node,
    ProcessedFunction* fn,
    ProcessedNodeInputs inputs,
    uint16_t outputs_offset)
    : node_(node),
      fn_(fn),
      inputs_(std::move(inputs)),
      outputs_offset_(outputs_offset)
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
      ,
      op_name_(node->kind().toQualString())
#endif
{
  TORCH_CHECK(
      node->outputs().size() < (1 << (sizeof(num_outputs_) * 8)),
      node->outputs().size(),
      " outputs to ProcessedNode ",
      node->kind().toQualString(),
      " is too many to use 2-byte indexing");
  num_outputs_ = node->outputs().size();
}

std::vector<IValue> ProcessedNode::inputs_ivalue_vec() const {
  std::vector<IValue> result;
  result.reserve(inputs_.size());
  for (const auto idx : c10::irange(num_inputs())) {
    result.emplace_back(Input(idx));
  }
  return result;
}

void ProcessedNode::run() {
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    at::RecordFunction guard(at::RecordScope::FUNCTION, pre_sampled);
    if (guard.isActive()) {
      if (guard.needsInputs()) {
        guard.before(get_op_name(), inputs_ivalue_vec());
      } else {
        guard.before(get_op_name());
      }
    }
    fn_->run(this);
  } else {
    fn_->run(this);
  }
#else
  fn_->run(this);
#endif
#ifndef NDEBUG
  if (FLAGS_static_runtime_disable_debug_memory_overlap_check) {
    // run check but do not enforce
    verify_no_memory_overlap();
  } else {
    DCHECK(verify_no_memory_overlap());
  }
#endif
}

static bool checkNoMemoryOverlap(const at::Tensor& a, const at::Tensor& b) {
  at::MemOverlapStatus status = at::get_overlap_status(a, b);
  if (status == at::MemOverlapStatus::FULL ||
      status == at::MemOverlapStatus::PARTIAL) {
    return false;
  }
  if (status == at::MemOverlapStatus::TOO_HARD) {
    VLOG(1) << "Detected TOO_HARD memory overlap status";
  }
  return true;
}

bool ProcessedNode::verify_no_memory_overlap(bool force_check) const {
  const static std::array<c10::Symbol, 4> special_case_ops = {
      fromQualString("prim::TypeCheck"),
      fromQualString("static_runtime::select_tensor"),
      fromQualString("static_runtime::VarTupleUnpack"),
      fromQualString("static_runtime::dict_unpack")};
  if (!force_check &&
      std::find(
          begin(special_case_ops), end(special_case_ops), node()->kind()) !=
          end(special_case_ops)) {
    return true;
  }

  return verify_outputs_dont_overlap_each_other() &&
      verify_inputs_dont_overlap_outputs(force_check);
}

bool ProcessedNode::verify_outputs_dont_overlap_each_other() const {
  for (const auto i : c10::irange(num_outputs_)) {
    if (!Output(i).isTensor()) {
      continue;
    }
    const auto& out0_t = Output(i).toTensor();
    for (const auto j : c10::irange(i + 1, num_outputs_)) {
      if (!Output(j).isTensor()) {
        continue;
      }
      const auto& out1_t = Output(j).toTensor();
      if (!checkNoMemoryOverlap(out0_t, out1_t)) {
        LOG(INFO) << "Node output " << i << " overlaps with output " << j
                  << ", " << PrintNode(node_);
        return false;
      }
    }
  }
  return true;
}

bool ProcessedNode::verify_inputs_dont_overlap_outputs(bool force_check) const {
  auto schema = node()->maybeSchema();
  // skip memory overlap check for mutable or view ops with only one output
  bool skip_check = !schema ||
      ((schema->is_mutable() || !fn_->checkMemoryOverlap()) &&
       num_outputs_ == 1);
  if (!force_check && skip_check) {
    if (!schema) {
      VLOG(2) << "Detected that op schema is null";
      return true;
    }
    VLOG(2) << "schema->is_mutable: " << schema->is_mutable()
            << ", fn_->checkMemoryOverlap: " << fn_->checkMemoryOverlap()
            << ", num_outputs_: " << num_outputs_;
    return true;
  }

  for (const auto i : c10::irange(inputs_.size())) {
    const IValue* in = &Input(i);
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    for (const auto j : c10::irange(num_outputs_)) {
      const IValue& out = Output(j);
      if (!out.isTensor()) {
        continue;
      }
      const auto& out_t = out.toTensor();
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        LOG(INFO) << "Node input " << i << " overlaps with output " << j << ", "
                  << PrintNode(node_);
        LOG(INFO) << *schema;
        return false;
      }
    }
  }
  return true;
}

bool ProcessedNode::check_and_correct_overlap_with(
    const at::Tensor& input,
    c10::IValue& output_ival) {
  auto& tensor = output_ival.toTensor();
  if (!checkNoMemoryOverlap(input, tensor)) {
    DLOG(INFO) << "Detected alias for node: " << PrintNode(node());
    output_ival = at::native::clone(tensor, c10::nullopt);
    set_outputs_memory_overlap_detected();
    return true;
  }
  return false;
}

void ProcessedNode::verify_and_correct_memory_overlap() {
  for (const auto i : c10::irange(inputs_.size())) {
    const IValue& in = Input(i);
    if (!in.isTensor()) {
      continue;
    }
    const auto& in_t = in.toTensor();
    for (const auto j : c10::irange(num_outputs_)) {
      auto& output = Output(j);
      if (output.isTensor()) {
        check_and_correct_overlap_with(in_t, output);
      } else if (output.isTensorList()) {
        auto tensors = output.toListRef();
        for (const auto& ival : tensors) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          check_and_correct_overlap_with(in_t, const_cast<c10::IValue&>(ival));
        }
#ifdef FBCODE_CAFFE2
        if (outputs_memory_overlap_detected()) {
          LOG_EVERY_MS(WARNING, 60000)
              << "Detected alias for node: " << PrintNode(node());
        }
#endif
      }
    }
  }
}

} // namespace jit
} // namespace torch
