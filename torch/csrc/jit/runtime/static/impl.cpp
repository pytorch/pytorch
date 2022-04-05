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
#include <torch/csrc/jit/passes/add_if_then_else.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/eliminate_no_ops.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/clone_native.h>
#endif

#include <iterator>
#include <limits>
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

namespace {

bool allArgsAreTensors(Node* node) {
  const auto& inputs = node->inputs();
  return std::all_of(inputs.begin(), inputs.end(), [](Value* value) {
    return value->type()->kind() == TypeKind::TensorType;
  });
}

} // namespace

// A manually curated set of ops that are disallowed in static runtime.
// These are rarely-used ops. Disallowing them typically eliminates
// corner cases in graph optimizations, allowing for more aggressive
// optimizations and better performance.
bool isUnsupportedOp(Node* node) {
  auto kind = node->kind();
  if (kind != aten::__is__ && kind != aten::__isnot__) {
    return false;
  }

  // We can't support aten::__is__ (and __isnot__) with tensor arguments.
  // Consider the following graph:
  // def forward(x):
  //     y = x.detach()
  //     return x is y
  // We have a graph optimization that removes the `detach` node since it is
  // a no-op during inference. But this affects the result - we get true
  // instead of false! There are many other graph passes affected by this
  // issue.
  return allArgsAreTensors(node);
}

// graph must be frozen or canEnableStaticRuntime would return false
// if there's any prim::CallMethod op left in the graph
bool canEnableStaticRuntime(const std::shared_ptr<torch::jit::Graph>& graph) {
  // check for sub-blocks
  bool can_support = true;
  bool has_blocks = false;
  for (auto* node : graph->block()->nodes()) {
    const auto kind = node->kind();
    if (kind == prim::Constant) {
      continue;
    }
    // check if can get op from Node
    const Operator* op = node->maybeOperator();
    if (isUnsupportedOp(node) || (!op && !nativeOpIsRegistered(kind))) {
      can_support = false;
      LOG(WARNING) << "Found unsupported op: " << kind.toQualString();
    }
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

void optimizeGraph(
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
  fuseInferenceOpsForSparseNN(graph);
  UseVariadicCat(graph);
  UseVariadicStack(graph);
  eliminateTrivialEquallySplit(graph);
  eliminateExtraPermuteOps(graph);

  if (opts.enable_out_variant) {
    UseVariadicOp(
        graph,
        fromQualString("fb::sigrid_transforms_torch_bind"),
        fromQualString("fb::variadic_sigrid_transforms_torch_bind"));
    fuseSignLog1P(graph);

    // TODO: we can avoid this guard by moving operations
    // to exposed folders.
#ifdef FBCODE_CAFFE2
    if (opts.use_copy_variants && !opts.enable_tensorexpr_fusion) {
      replaceWithCopy(graph);
    }
    if (opts.use_maybe_copy_variants && !opts.enable_tensorexpr_fusion) {
      replaceWithMaybeCopy(graph);
    }
    fuseListUnpack(graph);
#endif
  }

  ConstantPropagation(graph);
  removeImmutableInputDictLookups(graph);
  useVariadicTupleUnpack(graph);
  useVariadicGroupedAccessor(graph);
  EliminateNoOps(
      graph, /* custom_ops */ {fromQualString("fb::scale_gradient")});
  AddIfThenElseOp(graph);
  useSplitAndSqueeze(graph);
  graph->dump();
  GRAPH_DUMP("Final graph after optimizations: ", graph);
}

bool isSelfInGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
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

bool mayContainAlias(const AliasDb& db, const Value* v1, const Value* v2) {
  // AliasDb is not const-correct here, so we have to const_cast
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(v1), const_cast<Value*>(v2));
}

bool mayContainAlias(
    const AliasDb& db,
    const Value* a,
    const FastSet<const Value*>& b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), valueVecFromFastSet(b));
}

void prepareGraphForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  TORCH_CHECK(canEnableStaticRuntime(graph));
  optimizeGraph(graph, opts, std::move(sample_inputs));

  // Static runtime moves its outputs out of the runtime
  // by default. In some rare cases, this is not actually safe to
  // do - for example, if the value is a constant, static runtime
  // needs to hold onto a copy. Rather than adding special logic
  // to handle this rare case, we use this pass to detect it and
  // create an owned reference that can be safely moved out of the
  // runtime.
  createOwnedRefsForSpecialIValues(*graph);

  // We assume that each sub-block has at least one output. If we
  // detect any that have 0, force the sub-block to return None.
  forceNonEmptyOutputs(*graph);
}

std::pair<std::shared_ptr<Graph>, c10::optional<Module>> prepareForStaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  LOG(INFO) << "StaticModuleOptions: enable_out_variant "
            << opts.enable_out_variant << ", optimize_memory "
            << opts.optimize_memory << ", manage_output_tensors "
            << opts.manage_output_tensors << ", use_copy_variants "
            << opts.use_copy_variants << ", use_maybe_copy_variants "
            << opts.use_maybe_copy_variants << ", enable_tensorexpr_fusion "
            << opts.enable_tensorexpr_fusion;

  Module module = m.copy();
  if (!is_frozen) {
    module.eval();
    module = freeze_module(module);
  }

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  if (!sample_inputs.empty() && isSelfInGraphInput(graph)) {
    sample_inputs.insert(sample_inputs.begin(), m._ivalue());
  }
  prepareGraphForStaticModule(graph, opts, std::move(sample_inputs));

  return std::make_pair(graph, module);
}

std::pair<std::shared_ptr<Graph>, c10::optional<Module>> prepareForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs) {
  prepareGraphForStaticModule(graph, opts, std::move(sample_inputs));
  return std::make_pair(graph, c10::nullopt);
}

} // namespace

void ValueGroup::init(const Block& block, const AliasDb& db) {
  external_aliases_.clear();
  output_aliases_.clear();
  // Build `external_aliases` as we look through nodes forwardly from
  // the graph's inputs and add aliases of the inputs being created by the
  // nodes.
  external_aliases_.insert(block.inputs().begin(), block.inputs().end());
  for (const auto* node : block.nodes()) {
    if (node->kind() == prim::Constant) {
      for (const auto* output : node->outputs()) {
        external_aliases_.insert(output);
      }
    }
  }
  for (const auto* node : block.nodes()) {
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
  output_aliases_.insert(block.outputs().begin(), block.outputs().end());
  for (const auto* node : block.nodes().reverse()) {
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

bool isPureFunction(const Node* node) {
  auto* schema = node->maybeSchema();
  return schema &&
      schema->aliasAnalysis() == c10::AliasAnalysisKind::PURE_FUNCTION;
}

} // namespace

ManagedTensorRanges::ManagedTensorRanges(
    Block& block,
    const AliasDb& alias_db,
    const FastSet<const Value*>& managed_tensor_values) {
  const std::vector<Node*> nodes(block.nodes().begin(), block.nodes().end());
  const FastSet<const Value*> graph_inputs(
      block.inputs().begin(), block.inputs().end());

  const auto num_nodes = nodes.size();
  for (const auto i : c10::irange(num_nodes)) {
    auto* node = nodes[i];
    for (auto* input : node->inputs()) {
      auto* lifetime = getLifetime(input);
      if (!lifetime) {
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
  for (auto* graph_output : block.outputs()) {
    auto* lifetime = getLifetime(graph_output);
    if (!lifetime) {
      continue;
    }
    lifetime->end = num_nodes;
  }

  // Handle aliases. Aliases may extend a Value*'s lifetime. If a node
  // has an input and output that may alias each other, set the input's
  // lifetime end to max(input.lifetime_end, output.lifetime_end). Iterate
  // backwards to handle chains of aliases.
  for (const auto* node : block.nodes().reverse()) {
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
        if (mayContainAlias(alias_db, input, output)) {
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
      freeing_node = block.return_node();
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
          prepareForStaticModule(g->copy(), opts, std::move(sample_inputs)),
          opts) {}

StaticModule::StaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts,
    std::vector<IValue> sample_inputs)
    : StaticModule(
          prepareForStaticModule(m, is_frozen, opts, std::move(sample_inputs)),
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

  {
    size_t nodes_size = 0, constants_size = 0;
    for (Node* node : graph_->nodes()) {
      ++(node->kind() == prim::Constant ? constants_size : nodes_size);
    }

    constants_.reserve(constants_size);
    functions_.reserve(nodes_size);
  }

  // Create ProcessedFunction instances first to freeze their addresses to pass
  // to ProcessedNode.
  AliasDb alias_db(graph_, /*isFrozen=*/false);
  GRAPH_DEBUG("AliasDb: ", alias_db.toString());

  // Maps each Value* in the graph to its index in the values_ array that will
  // eventually be created by StaticRuntime.
  FastMap<const Value*, uint32_t> value_to_index;
  prepareFunctionsAndConstants(graph_->block(), alias_db, value_to_index);

  const auto constants_index_offset = 0;
  const auto values_index_offset = constants_index_offset + constants().size();
  value_buffer_size_ = values_index_offset;

  value_buffer_size_ +=
      prepareBlockInfo(graph_->block(), values_index_offset, value_to_index);

  prepareStaticNodeInfos(graph_->block(), value_to_index, alias_db);

  for (auto& block_and_info : block_infos_) {
    auto& block_info = block_and_info.second;
    block_info.prepareForMemoryPlanner(alias_db, opts);
  }
}

size_t StaticModule::prepareBlockInfo(
    Block* block,
    const size_t start_idx,
    FastMap<const Value*, uint32_t>& value_to_index) {
  block_infos_.emplace(block, BlockInfo(start_idx, *block));

  const auto numInputs = block->inputs().size();
  for (const auto i : c10::irange(numInputs)) {
    value_to_index.emplace(block->inputs()[i], start_idx + i);
  }
  auto cur_idx = start_idx + numInputs;

  for (auto* node : block->nodes()) {
    for (auto* sub_block : node->blocks()) {
      cur_idx += prepareBlockInfo(sub_block, cur_idx, value_to_index);
    }

    if (node->kind() == prim::Constant) {
      continue;
    }

    TORCH_CHECK(
        cur_idx < (1 << 16),
        "outputs offset in values table",
        cur_idx,
        " would overflow 2-byte index storage");

    const auto numOutputs = node->outputs().size();
    for (const auto i : c10::irange(numOutputs)) {
      value_to_index.emplace(node->outputs()[i], cur_idx + i);
    }
    cur_idx += numOutputs;
  }

  std::vector<uint16_t> output_indices;
  output_indices.reserve(block->outputs().size());
  for (auto* output : block->outputs()) {
    const auto output_idx = value_to_index.at(output);
    TORCH_CHECK(
        output_idx < (1 << 16),
        "outputs offset in values table",
        output_idx,
        " would overflow 2-byte index storage");
    output_indices.push_back(output_idx);
  }

  block_infos_.at(block).setOutputIndices(std::move(output_indices));
  return cur_idx - start_idx;
}

void StaticModule::prepareFunctionsAndConstants(
    Block* block,
    const AliasDb& alias_db,
    FastMap<const Value*, uint32_t>& value_to_index) {
  for (auto* node : block->nodes()) {
    for (auto* sub_block : node->blocks()) {
      prepareFunctionsAndConstants(sub_block, alias_db, value_to_index);
    }

    if (node->kind() == prim::Constant) {
      auto* v = node->output();
      TORCH_CHECK(v->type()->kind() != FunctionType::Kind);
      value_to_index.emplace(v, constants_.size());
      constants_.emplace_back(toIValue(v).value());
      continue;
    }

    // see [Check and correct bad schema alias info at runtime]
    bool check_outputs_for_overlap =
        !alias_db.mayContainAlias(node->inputs(), node->outputs()) &&
        containTensorsOnly(node->outputs());
    // new ProcessedFunction
    functions_.emplace_back(
        node, opts_.enable_out_variant, check_outputs_for_overlap);
  }
}

size_t StaticModule::prepareStaticNodeInfos(
    Block* block,
    const FastMap<const Value*, uint32_t>& value_to_index,
    const AliasDb& alias_db,
    size_t node_idx) {
  const auto node_start = node_idx;

  auto& block_info = block_infos_.at(block);
  std::vector<StaticNodeInfo> nodes;
  FastMap<Node*, bool> node_hasOutVariant;

  for (auto* node : block->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }

    for (auto* sub_block : node->blocks()) {
      node_idx +=
          prepareStaticNodeInfos(sub_block, value_to_index, alias_db, node_idx);
    }
    ProcessedNodeInputs input_indices(node->inputs().size());
    for (const auto input_idx : c10::irange(node->inputs().size())) {
      auto* input = node->inputs()[input_idx];
      auto input_ivalue_idx = value_to_index.at(input);
      TORCH_CHECK(
          input_ivalue_idx < (1 << 16),
          "input index in values table ",
          input_ivalue_idx,
          " would overflow 2-byte index storage");
      input_indices[input_idx] = input_ivalue_idx;
    }

    ProcessedFunction* fn = &functions_[node_idx];

    // create a new ProcessedNode
    const auto node_output_idx = node->outputs().empty()
        // The index is unused if there are no outputs, so just create a
        // placeholder value.
        ? std::numeric_limits<uint16_t>::max()
        : value_to_index.at(node->output(0));
    nodes.emplace_back(node, fn, std::move(input_indices), node_output_idx);

    node_hasOutVariant.emplace(node, nodes.back().hasOutVariant());
    ++node_idx;
  }

  block_info.setNodes(std::move(nodes), node_hasOutVariant);
  block_info.initValueGroup(alias_db);

  return node_idx - node_start;
}

void BlockInfo::setNodes(
    std::vector<StaticNodeInfo> nodes,
    const FastMap<Node*, bool>& node_hasOutVariant) {
  nodes_ = std::move(nodes);

  for (auto& node : nodes_) {
    if (node.numOutputs() == 1 &&
        isOptimizableContainerType(node.node(), node_hasOutVariant)) {
      node_is_optimizable_container_type_.emplace(node.node());
    }
  }
}
void BlockInfo::prepareForMemoryPlanner(
    const AliasDb& alias_db,
    const StaticModuleOptions& opts) {
  if (!opts.enable_out_variant) {
    return;
  }

  // Never manage graph outputs so that we can do std::move(output_ivalue).
  // This does not affect performance if the graph returns a collection object.
  FastSet<const Value*> graph_output_values(
      block_.outputs().begin(), block_.outputs().end());

  // collect register indices of outputs of ops with out variant
  for (StaticNodeInfo& pnode : nodes_) {
    if (!pnode.hasOutVariant()) {
      continue;
    }
    auto outputs = pnode.node()->outputs();
    for (const auto i : c10::irange(outputs.size())) {
      const Value* out_v = outputs[i];
      // Types are stored in the underlying TorchScript IR
      bool is_tensor_type = out_v->type()->castRaw<TensorType>();
      if (opts.manage_output_tensors && is_tensor_type &&
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
      } else if (nodeIsOptimizableContainerType(pnode.node())) {
        // We "leak" certain container types because their allocations
        // take a long time
        leaked_values_.insert(out_v);
      }
    }
  }

  for (const Value* output : block_.outputs()) {
    managed_tensor_values_.erase(output);
  }
  GRAPH_DEBUG("managed_tensor_values: ", dumpValueSet(managed_tensor_values_));
  GRAPH_DEBUG(
      "managed_output_tensor_values_: ",
      dumpValueSet(managed_output_tensor_values_));

  managed_tensor_ranges_ =
      ManagedTensorRanges(block_, alias_db, managed_tensor_values_);
}

const StaticModuleOptions& StaticModule::opts() const {
  return opts_;
}

size_t StaticModule::numOutputs() const {
  return graph_->outputs().size();
}

size_t StaticModule::numInputs() const {
  return num_inputs_;
}

StaticRuntime& StaticModule::runtime() {
  if (!cached_runtime_) {
    cached_runtime_ = std::make_unique<StaticRuntime>(*this);
  }
  return *cached_runtime_;
}

StaticRuntime StaticModule::cloneRuntimeFromCached() {
  return runtime().clone();
}

Node* StaticModule::findNodeWithKindForTesting(const std::string& kind) const {
  for (auto& block_and_info : block_infos_) {
    auto& block_info = block_and_info.second;
    for (auto& pnode : block_info.nodes()) {
      if (pnode.node()->kind().toQualString() == kind) {
        return pnode.node();
      }
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

BlockRunner::BlockRunner(
    const StaticModule& sm,
    IValue* values,
    Block* block,
    bool is_root_block)
    : static_module_(sm),
      block_info_(static_module_.blockInfo(block)),
      is_root_block_(is_root_block),
      first_input_is_self_(is_root_block_ && static_module_.firstInputIsSelf()),
      inputs_begin_(block_info_.blockInputsIdx()),
      // TODO(T108633124): Turn on manage output tensors for sub-blocks.
      manage_output_tensors_enabled_(
          is_root_block_ && sm.opts().manage_output_tensors),
      memory_planner_algorithm_(sm.opts().memory_planner_algorithm),
      values_(values) {
  nodes_.reserve(block_info_.nodes().size());
  for (auto& pre_pnode : block_info_.nodes()) {
    nodes_.emplace_back(pre_pnode, values_);
  }

  for (auto index : block_info_.blockOutputIndices()) {
    outputs_.emplace_back(&values_[index]);
  }

  for (auto& pnode : nodes_) {
    auto* node = pnode.node();
    auto blocks = node->blocks();
    const auto num_blocks = blocks.size();
    if (num_blocks == 0) {
      continue;
    }
    DCHECK(node->kind() == prim::If || node->kind() == prim::Loop);
    auto block_runners = std::make_unique<std::vector<BlockRunner>>();
    block_runners->reserve(num_blocks);

    for (auto* b : blocks) {
      block_runners->emplace_back(sm, values_, b);
    }
    pnode.setBlockRunners(std::move(block_runners));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
BlockRunner::BlockRunner(BlockRunner&&) noexcept = default;

BlockRunner::~BlockRunner() = default;

void BlockRunner::maybeCloneMemoryPlanner(
    const BlockRunner& src,
    const FastMap<at::Tensor*, at::Tensor*>& old_tensor_to_new) {
  if (!src.planner_) {
    return;
  }
  planner_ = src.planner_->maybeClone(this, old_tensor_to_new);
}

void BlockRunner::setArg(const size_t idx, std::vector<IValue>&& args) {
  DCHECK(idx < args.size());
  input(idx + first_input_is_self_) = std::move(args[idx]);
}

void BlockRunner::setArg(const size_t idx, const std::vector<IValue>& args) {
  DCHECK(idx < args.size());
  input(idx + first_input_is_self_) = args[idx];
}

void BlockRunner::setArg(const size_t idx, const IValue& arg) {
  input(idx + first_input_is_self_) = arg;
}

namespace {
void checkType(const Argument& schema_arg, const IValue& arg) {
  // Fast path for most common case
  if (arg.isTensor() &&
      schema_arg.type()->kind() == c10::TypeKind::TensorType) {
    return;
  }
  TORCH_CHECK(arg.type()->isSubtypeOf(schema_arg.type()));
}
} // namespace

template <typename IValueList>
void BlockRunner::setInputs(
    IValueList&& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  const auto& schema = static_module_.schema();
  if (first_input_is_self_) {
    input(0) = static_module_.module()._ivalue();
  }

  if (!is_root_block_ || C10_UNLIKELY(!schema)) {
    TORCH_CHECK(
        kwargs.empty(), "Schema is not available, but BlockRunner got kwargs.");

    const auto total_numInputs = args.size() + first_input_is_self_;
    TORCH_CHECK(total_numInputs == block_info_.numInputs());

    for (size_t i = 0; i < args.size(); ++i) {
      setArg(i, std::forward<IValueList>(args));
    }
    return;
  }

  const auto& schema_args = schema->arguments();
  size_t consumed_kwargs = 0;
  DCHECK(schema_args.size() > 0);
  TORCH_CHECK(
      args.size() < schema_args.size(),
      "Static runtime got too many arguments");
  for (size_t i = 0; i < schema_args.size() - 1; ++i) {
    // Start at 1 since the schema always contains `self`.
    const auto& schema_arg = schema_args[i + 1];

    if (i < args.size()) {
      checkType(schema_arg, args[i]);
      setArg(i, std::forward<IValueList>(args));
      continue;
    }

    auto it = kwargs.find(schema_arg.name());
    if (it != kwargs.end()) {
      checkType(schema_arg, it->second);
      setArg(i, it->second);
      ++consumed_kwargs;
      continue;
    }

    auto maybe_default_val = schema_arg.default_value();
    if (maybe_default_val) {
      setArg(i, *maybe_default_val);
      continue;
    }

    TORCH_CHECK(
        false, "Static runtime is missing required kwarg ", schema_arg.name());
  }
  TORCH_CHECK(consumed_kwargs == kwargs.size());
}

namespace {
std::unique_ptr<MemoryPlanner> memoryPlannerFactory(
    MemoryPlannerAlgorithm algorithm,
    BlockRunner* block_runner,
    const BlockInfo& block_info,
    const StaticModuleOptions& opts,
    bool manage_output_tensors_enabled) {
  switch (algorithm) {
    case MemoryPlannerAlgorithm::kStandardResizing:
      return std::make_unique<StandardMemoryPlanner>(
          block_runner,
          block_info,
          opts.enable_out_variant,
          manage_output_tensors_enabled,
          opts.optimize_memory);
    case MemoryPlannerAlgorithm::kPrecomputedOffsets:
      return std::make_unique<PrecomputedOffsetsMemoryPlanner>(
          block_runner,
          block_info,
          opts.enable_out_variant,
          manage_output_tensors_enabled,
          opts.optimize_memory,
          opts.max_allowed_reallocs);
  }
  // Some old compilers can't figure out that control never reaches the end
  // of this function
  TORCH_CHECK(false);
  return nullptr;
}
} // namespace

void BlockRunner::createMemoryPlanner() {
  if (!planner_) {
    const auto& opts = static_module_.opts();
    planner_ = memoryPlannerFactory(
        memory_planner_algorithm_,
        this,
        block_info_,
        opts,
        manage_output_tensors_enabled_);
  }
}

void BlockRunner::maybeAllocate() {
  DCHECK(planner_);
  if (planner_->shouldFallBackToStandardStrategy()) {
    for (auto& n : nodes_) {
      for (const auto i : c10::irange(n.outputs().size())) {
        n.output(i) = IValue();
      }
    }
    planner_ = nullptr;
    memory_planner_algorithm_ = MemoryPlannerAlgorithm::kStandardResizing;
  } else {
    planner_->allocate();
  }
}

namespace {

void destroyNodeOutputs(ProcessedNode& p_node) {
  const auto borrows_outputs = borrowsOutputs(p_node.node()->kind());
  for (const auto i : c10::irange(p_node.numOutputs())) {
    auto& output = p_node.output(i);
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

void BlockRunner::cleanUpIntermediateIValues() noexcept {
  // We have to iterate in reverse order here due to borrowed
  // IValues - we don't want to destroy a value until all of its
  // borrows are cleaned up!
  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    destroyNodeOutputs(*it);
  }
}

void BlockRunner::resetMemory() noexcept {
  planner_.reset();
  // We must clean up intermediate values before inputs in case
  // there are borrowed inputs and static runtime owns the only
  // reference (e.g. the inputs were std::move'd into the runtime)
  cleanUpIntermediateIValues();
  cleanUpInputIValues();
}

c10::IValue BlockRunner::moveOutputsToTuple(uint32_t numOutputs) {
  switch (numOutputs) {
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
      outputs.reserve(numOutputs);
      for (const auto i : c10::irange(numOutputs)) {
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
void BlockRunner::verifyAndCorrectMemoryOverlap(ProcessedNode& n) {
  // The slow check can be removed once the internal/output buffers are merged
  if (C10_UNLIKELY(n.checkOutputsForMemoryOverlap())) {
    if (C10_UNLIKELY(!planner_)) {
      // slow check, for first iter only
      n.verifyAndCorrectMemoryOverlap();
    } else {
      bool overlap_detected_with_fast_check = false;
      for (size_t i = 0; i < n.outputs().size(); i++) {
        auto& output = n.output(i);
        if (output.isTensor()) {
          overlap_detected_with_fast_check |=
              fastCheckAndCorrectOverlapWith(n, output);
        } else if (output.isTensorList()) {
          auto tensor_list = output.toListRef();
          for (auto& ival : tensor_list) {
            overlap_detected_with_fast_check |= fastCheckAndCorrectOverlapWith(
                n,
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                const_cast<c10::IValue&>(ival));
          }
        }
      }
      if (n.outputsMemoryOverlapDetected() &&
          !overlap_detected_with_fast_check) {
        // slow check. Only run when the fast check fails.
        n.verifyAndCorrectMemoryOverlap();
      }
    }
  }
}

bool BlockRunner::fastCheckAndCorrectOverlapWith(
    ProcessedNode& n,
    c10::IValue& tensor_ival) {
  auto& tensor = tensor_ival.toTensor();
  if (planner_->overlapWithInternalBuffer(tensor.data_ptr())) {
    DLOG(INFO) << "Detected alias for node: " << PrintNode(n.node());
    tensor_ival = at::native::clone(tensor, c10::nullopt);
    n.setOutputsMemoryOverlapDetected();
    return true;
  }
  return false;
}

BlockRunner::Deallocator::~Deallocator() {
  // Assume cleanup cannot throw.
  cleanupImpl();
#ifndef NDEBUG
  block_runner_.checkForMemoryLeak(/*output_returned*/ false);
#endif
}

void BlockRunner::Deallocator::cleanupImpl() {
  // MemoryPlanner is created after the first invocation of `run()`. This
  // is done intentionally because MemoryPlanner uses `Tensor` sizes of
  // the previous `run()` for memory planning of subsequent runs
  if (C10_LIKELY(finished_)) {
    block_runner_.createMemoryPlanner();
  }

  if (C10_LIKELY(block_runner_.planner_)) {
    block_runner_.planner_->deallocate();
  } else {
    // This is the first run, and it didn't finish, so we can't use a
    // `MemoryPlanner` to deallocate stuff. Just reset everything mannually.
    block_runner_.resetMemory();
  }
  // clean up owning refs of input tensors
  block_runner_.cleanUpInputIValues();
}

template <typename IValueList>
c10::IValue BlockRunner::runImpl(IValueList&& args, const KeywordArgs& kwargs) {
  // We assume inference workloads, so we do not need
  // autograd. Enabling this is a significant win on dispatcher
  // overhead because it saves a round of dispatch for at least some
  // functions, such as resize_ and resize_as_.
  c10::InferenceMode mode;

  {
    auto on_exit = Deallocator(*this);

    if (planner_) {
      maybeAllocate();
    }

    setInputs(std::forward<IValueList>(args), kwargs);

    for (auto& n : nodes_) {
      // LOG(INFO) << "Running node: " << PrintNode(n.node());
      n.run();
      // Check for incorrect schema alias info.
      verifyAndCorrectMemoryOverlap(n);
    }
    on_exit.setFinished();
  }

  // no need to keep references of outputs in static runtime anymore
  if (block_info_.numOutputs() > 1) {
    return moveOutputsToTuple(block_info_.numOutputs());
  }

  DCHECK(checkForMemoryLeak(/*output_returned*/ false));

  // use move here. Otherwise, clean up outputs_[0] explicitly
  return std::move(*outputs_[0]);
}

template <typename IValueList>
c10::IValue BlockRunner::runImplWithRecordFunctions(
    IValueList&& args,
    const KeywordArgs& kwargs) {
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    at::RecordFunction guard(
        at::RecordScope::STATIC_RUNTIME_MODEL, pre_sampled);
    if (guard.isActive()) {
      if (guard.needsInputs()) {
        guard.before("forward", &args);
      } else {
        guard.before("forward");
      }
    }
    return runImpl(std::forward<IValueList>(args), kwargs);
  }
  return runImpl(std::forward<IValueList>(args), kwargs);
}

c10::IValue BlockRunner::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  return runImpl(args, kwargs);
#else
  return runImplWithRecordFunctions(args, kwargs);
#endif
}

c10::IValue BlockRunner::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
#ifdef PYTORCH_DISABLE_NET_PROFILING
  return runImpl(std::move(args), kwargs);
#else
  return runImplWithRecordFunctions(std::move(args), kwargs);
#endif
}

namespace {

std::string generateLatencyJSON(const std::string& label, double millis) {
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

void BlockRunner::benchmark(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const int warmup_runs,
    const int main_runs,
    bool print_per_node_time,
    bool generate_ai_pep_output) {
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());
  std::cout << "Input size: " << args_list.size() << std::endl;
  float time_per_iter =
      benchmarkModel(args_list, kwargs_list, warmup_runs, main_runs);
  std::cout << "Static runtime ms per iter: " << time_per_iter
            << ". Iters per second: " << 1000.0 / time_per_iter << std::endl;

  IndividualMetrics results =
      benchmarkIndividualOps(args_list, kwargs_list, warmup_runs, main_runs);

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
  if (args_list.size() == 0) {
    std::sort(
        time_per_node_type_vec.begin(),
        time_per_node_type_vec.end(),
        [&results](auto& left, auto& right) {
          return results.instances_per_node_type[left.first] >
              results.instances_per_node_type[right.first];
        });
  } else {
    std::sort(
        time_per_node_type_vec.begin(),
        time_per_node_type_vec.end(),
        [](auto& left, auto& right) { return left.second > right.second; });
  }
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
      LOG(INFO) << generateLatencyJSON(kind, ms);
    }
  }
  if (generate_ai_pep_output) {
    LOG(INFO) << generateLatencyJSON(
        "static_runtime_first_iter", results.first_iter_time);
  }
  std::cout << std::setw(15) << results.total_time << " ms. in Total"
            << std::endl;
  std::cout << "BlockRunner setup time: " << results.setup_time << " ms"
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
              << planner_->totalNumManagedTensors() << std::endl;
    std::cout << "Total number of managed output tensors: "
              << planner_->totalNumManagedOutputTensors() << std::endl;
    std::cout << "Total number of unmanaged values: "
              << planner_->totalNumUnmanaged() << std::endl;
    std::cout << "Number of unmanaged values requiring cleanup: "
              << planner_->numUnmanagedNonScalars() << std::endl;
    std::cout << "Number of unmanaged values not requiring cleanup: "
              << planner_->numUnmanagedScalars() << std::endl;
    std::cout << "Total memory managed: " << planner_->totalManaged()
              << " bytes" << std::endl;
    if (static_module_.opts().optimize_memory) {
      std::cout << "Total number of reused tensors: "
                << planner_->totalReusedTensors() << std::endl;
    }
  }
  std::cout << "Total number of 'out' variant nodes/total number of nodes: "
            << results.out_nodes_count << "/" << results.total_nodes_count
            << " ("
            << 100.0 * (results.out_nodes_count) /
          static_cast<float>(results.total_nodes_count)
            << "%)" << std::endl;

  checkForMemoryLeak();

#ifndef NDEBUG
  KeywordArgs empty_kwargs;
  displayNodes(
      args_list[0], kwargs_list.size() > 0 ? kwargs_list[0] : empty_kwargs);
#endif
}

float BlockRunner::benchmarkModel(
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
    }
  }
  caffe2::Timer timer;
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
    }
  }
  float millis = timer.MilliSeconds();
  return millis / (static_cast<float>(main_runs) * args_list.size());
}

bool displayIValue(const IValue& iv) {
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

void displayProcessedNodeInfo(const ProcessedNode& pnode) {
  pnode.node()->print(std::cout, 0, nullptr, false);
  for (const auto i : c10::irange(pnode.numInputs())) {
    std::cout << "\ti" << i << ": ";
    if (!displayIValue(pnode.input(i))) {
      std::cout << *(pnode.node()->inputs()[i]->type()) << '\n';
    }
  }
  const auto outputs = pnode.outputs();
  for (const auto i : c10::irange(outputs.size())) {
    std::cout << "\to" << i << ": ";
    if (!displayIValue(outputs[i])) {
      std::cout << *(pnode.node()->outputs()[i]->type()) << '\n';
    }
  }
}

void BlockRunner::displayNodes(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
  c10::InferenceMode mode;

  auto on_exit = Deallocator(*this);

  if (planner_) {
    planner_->allocate();
  }
  setInputs(args, kwargs);

  for (auto& node : nodes_) {
    node.run();
    displayProcessedNodeInfo(node);
  }
  on_exit.setFinished();
}

BlockRunner::IndividualMetrics BlockRunner::benchmarkIndividualOps(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<KeywordArgs>& kwargs_list,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());
  TORCH_CHECK(warmup_runs >= 1 && main_runs >= 1);

  IndividualMetrics results;
  results.time_per_node.resize(nodes_.size(), 0);
  if (args_list.size() == 0) {
    // When the given input is empty, compute the op statistics from the given
    // graph without executing it.
    for (const auto i : c10::irange(nodes_.size())) {
      const Node* node = nodes_[i].node();
      std::string kind(node->kind().toQualString());
      // TODO: Collect op statistics from sub-blocks here.
      results.time_per_node[i] = 0;
      results.time_per_node_type[kind] = 0;
      results.instances_per_node_type[kind]++;
      if (nodes_[i].hasOutVariant()) {
        results.out_nodes.insert(kind);
        results.out_nodes_count++;
      } else if (nodes_[i].hasNative()) {
        results.native_nodes.insert(kind);
      }
      results.total_time += results.time_per_node[i];
    }
    results.total_nodes_count = nodes_.size();
    results.memory_alloc_time = 0;
    results.memory_dealloc_time = 0;
    results.output_dealloc_time = 0;
    for (const auto& p : results.time_per_node_type) {
      const std::string& kind = p.first;
      results.percent_per_node_type[kind] = 0;
    }
    return results;
  }

  const bool is_kwargs_empty = kwargs_list.size() == 0;
  const KeywordArgs empty_kwargs;
  bool manage_output_tensors = static_module_.opts().manage_output_tensors;
  // See comment on above use of InferenceMode for
  // explanation.
  c10::InferenceMode mode;

  // setup time
  caffe2::Timer timer;

  setInputs(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);

  results.setup_time = timer.MilliSeconds();

  // The first iteration profiles each node's output Tensors' sizes and
  // initializes the memory planner with the profile information. Folllowing
  // iterations just use the already established memory planning.
  timer.Start();
  operator()(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);
  results.first_iter_time = timer.MilliSeconds();

  // warmup runs
  for (const auto i : c10::irange(warmup_runs - 1)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
    }
  }

  // main runs
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning

    for (const auto j : c10::irange(args_list.size())) {
      setInputs(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);

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
        verifyAndCorrectMemoryOverlap(nodes_[k]);
      }
      timer.Start();
      createMemoryPlanner();
      planner_->deallocate();
      // clean up owning refs of input tensors
      cleanUpInputIValues();
      millis = timer.MilliSeconds();
      results.memory_dealloc_time += millis;

      timer.Start();
      // no need to keep references of outputs in static runtime anymore
      c10::IValue output;
      if (static_module_.numOutputs() > 1) {
        output = moveOutputsToTuple(static_module_.numOutputs());
      }

      DCHECK(checkForMemoryLeak(/*output_returned*/ false));

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
    if (nodes_[i].hasOutVariant()) {
      results.out_nodes.insert(kind);
      results.out_nodes_count++;
    } else if (nodes_[i].hasNative()) {
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

bool BlockRunner::checkForMemoryLeak(
    bool output_returned,
    bool recurse_on_sub_blocks) {
  // check for inputs
  for (const auto i : c10::irange(block_info_.numInputs())) {
    TORCH_CHECK(
        values_[i + block_info_.blockInputsIdx()].isNone(),
        "Input ",
        i,
        " was not cleaned up");
  }
  FastSet<const IValue*> output_ivalues(outputs_.begin(), outputs_.end());
  for (const auto n : c10::irange(nodes_.size())) {
    auto& pnode = nodes_[n];
    for (const auto i : c10::irange(pnode.numOutputs())) {
      const IValue* ival = &pnode.output(i);
      const Value* val = pnode.node()->output(i);
      const std::string error_msg = "Output " + c10::to_string(i) + ", %" +
          val->debugName() + " of node " + c10::to_string(n) +
          " which has kind " + pnode.node()->kind().toQualString() +
          " was not cleaned up";
      if (output_ivalues.count(ival) == 0) {
        // check for intermediates
        if (!ival->isNone()) {
          TORCH_CHECK(
              ival->isTensor() ||
                  block_info_.nodeIsOptimizableContainerType(pnode.node()) ||
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

    auto* block_runners = pnode.blockRunners();
    if (recurse_on_sub_blocks && block_runners) {
      for (auto& block_runner : *block_runners) {
        block_runner.checkForMemoryLeak(output_returned, recurse_on_sub_blocks);
      }
    }
  }
  VLOG(1) << "Finished checking for memory leak";
  return true;
}

ProcessedFunction::ProcessedFunction(
    Node* node,
    bool enable_out_variant,
    bool check_memory_overlap)
    : check_memory_overlap_(check_memory_overlap),
      num_outputs_(node->outputs().size()) {
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
      const size_t size = pnode->numInputs();
      stack.reserve(size + has_var_args);
      for (const auto i : c10::irange(size)) {
        stack.emplace_back(pnode->input(i));
      }
      // Need to store the number of inputs in stack for variadic ops.
      if (has_var_args) {
        stack.emplace_back(static_cast<int>(size));
      }
      node_op(stack);
      DCHECK_EQ(stack.size(), pnode->numOutputs());
      for (const auto i : c10::irange(pnode->numOutputs())) {
        pnode->output(i) = std::move(stack[i]);
      }
    };
    kind_ = ProcessedFunction::Kind::kInterpreterFallback;
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

StaticNodeInfo::StaticNodeInfo(
    Node* node,
    ProcessedFunction* fn,
    ProcessedNodeInputs inputs,
    uint16_t outputs_offset)
    : node_(node),
      fn_(fn),
      inputs_(std::move(inputs)),
      outputs_offset_(outputs_offset) {
  TORCH_CHECK(numOutputs() == node->outputs().size());
}

std::vector<IValue> ProcessedNode::inputIValueVec() const {
  std::vector<IValue> result;
  result.reserve(inputs_.size());
  for (const auto idx : c10::irange(numInputs())) {
    result.emplace_back(input(idx));
  }
  return result;
}

void ProcessedNode::run() {
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    at::RecordFunction guard(at::RecordScope::STATIC_RUNTIME_OP, pre_sampled);
    if (guard.isActive()) {
      if (guard.needsInputs()) {
        guard.before(getOpName(), inputIValueVec());
      } else {
        guard.before(getOpName());
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
    verifyNoMemoryOverlap();
  } else {
    DCHECK(verifyNoMemoryOverlap());
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

bool ProcessedNode::verifyNoMemoryOverlap(bool force_check) const {
  const static std::array<c10::Symbol, 7> special_case_ops = {
      fromQualString("prim::TypeCheck"),
      fromQualString("prim::IfThenElse"),
      fromQualString("static_runtime::select_tensor"),
      fromQualString("static_runtime::VarTupleUnpack"),
      fromQualString("static_runtime::dict_unpack"),
      fromQualString("static_runtime::fused_split_and_squeeze"),
      fromQualString("static_runtime::create_owned_ref")};
  if (!force_check &&
      std::find(
          begin(special_case_ops), end(special_case_ops), node()->kind()) !=
          end(special_case_ops)) {
    return true;
  }

  return verifyOutputsDontOverlapEachOther() &&
      verifyInputsDontOverlapOutputs(force_check);
}

bool ProcessedNode::verifyOutputsDontOverlapEachOther() const {
  for (const auto i : c10::irange(numOutputs())) {
    if (!output(i).isTensor()) {
      continue;
    }
    const auto& out0_t = output(i).toTensor();
    for (const auto j : c10::irange(i + 1, numOutputs())) {
      if (!output(j).isTensor()) {
        continue;
      }
      const auto& out1_t = output(j).toTensor();
      if (!checkNoMemoryOverlap(out0_t, out1_t)) {
        LOG(INFO) << "Node output " << i << " overlaps with output " << j
                  << ", " << PrintNode(node_);
        return false;
      }
    }
  }
  return true;
}

bool ProcessedNode::verifyInputsDontOverlapOutputs(bool force_check) const {
  auto schema = node()->maybeSchema();
  // skip memory overlap check for mutable or view ops with only one output
  bool skip_check = !schema ||
      ((schema->is_mutable() || !fn_->checkMemoryOverlap()) &&
       numOutputs() == 1);
  if (!force_check && skip_check) {
    if (!schema) {
      VLOG(2) << "Detected that op schema is null";
      return true;
    }
    VLOG(2) << "schema->is_mutable: " << schema->is_mutable()
            << ", fn_->checkMemoryOverlap: " << fn_->checkMemoryOverlap()
            << ", numOutputs_: " << numOutputs();
    return true;
  }

  for (const auto i : c10::irange(inputs_.size())) {
    const IValue* in = &input(i);
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    for (const auto j : c10::irange(numOutputs())) {
      const IValue& out = output(j);
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

bool ProcessedNode::checkAndCorrectOverlapWith(
    const at::Tensor& input,
    c10::IValue& output_ival) {
  auto& tensor = output_ival.toTensor();
  if (!checkNoMemoryOverlap(input, tensor)) {
    DLOG(INFO) << "Detected alias for node: " << PrintNode(node());
    output_ival = at::native::clone(tensor, c10::nullopt);
    setOutputsMemoryOverlapDetected();
    return true;
  }
  return false;
}

void ProcessedNode::verifyAndCorrectMemoryOverlap() {
  for (const auto i : c10::irange(inputs_.size())) {
    const IValue& in = input(i);
    if (!in.isTensor()) {
      continue;
    }
    const auto& in_t = in.toTensor();
    for (const auto j : c10::irange(numOutputs())) {
      auto& output_val = output(j);
      if (output_val.isTensor()) {
        checkAndCorrectOverlapWith(in_t, output_val);
      } else if (output_val.isTensorList()) {
        auto tensors = output_val.toListRef();
        for (const auto& ival : tensors) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          checkAndCorrectOverlapWith(in_t, const_cast<c10::IValue&>(ival));
        }
#ifdef FBCODE_CAFFE2
        if (outputsMemoryOverlapDetected()) {
          LOG_EVERY_MS(WARNING, 60000)
              << "Detected alias for node: " << PrintNode(node());
        }
#endif
      }
    }
  }
}

StaticRuntime::StaticRuntime(const StaticModule& sm)
    : values_(sm.valueBufferSize()), parent_module_(sm) {
  std::copy(sm.constants().begin(), sm.constants().end(), values_.data());
  block_ = std::make_unique<BlockRunner>(
      sm, values_.data(), sm.rootBlock(), /*is_root_block*/ true);
}

StaticRuntime StaticRuntime::clone() const {
  StaticRuntime runtime(parent_module_);
  if (parent_module_.opts().memory_planner_algorithm !=
      MemoryPlannerAlgorithm::kPrecomputedOffsets) {
    return runtime;
  }
  auto* values = values_.data();
  auto* new_values = runtime.values_.data();
  DCHECK_EQ(runtime.values_.size(), values_.size());
  FastMap<at::Tensor*, at::Tensor*> old_tensor_to_new;
  for (const auto i :
       c10::irange(parent_module_.constants().size(), values_.size())) {
    if (values[i].isTensor()) {
      auto& old_tensor = values[i].toTensor();
      new_values[i] = createEmptyFrom(old_tensor.sizes(), old_tensor);
      old_tensor_to_new.emplace(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<at::Tensor*>(&values[i].toTensor()),
          &new_values[i].toTensor());
    }
  }
  DCHECK(runtime.block_ != nullptr);
  DCHECK(block_ != nullptr);
  runtime.block_->maybeCloneMemoryPlanner(*block_, old_tensor_to_new);
  return runtime;
}

c10::IValue StaticRuntime::operator()(
    const std::vector<c10::IValue>& args,
    const KeywordArgs& kwargs) {
  return (*block_)(args, kwargs);
}

c10::IValue StaticRuntime::operator()(
    std::vector<c10::IValue>&& args,
    const KeywordArgs& kwargs) {
  return (*block_)(std::move(args), kwargs);
}

bool StaticRuntime::checkForMemoryLeak(bool output_returned) {
  return block_->checkForMemoryLeak(
      output_returned, /* recurse_on_sub_blocks */ true);
}

const MemoryPlanner* StaticRuntime::getMemoryPlanner() const {
  return block_->getMemoryPlanner();
}

} // namespace jit
} // namespace torch
