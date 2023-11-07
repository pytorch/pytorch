#include <torch/csrc/jit/runtime/static/fusion.h>

#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>

namespace torch::jit {

void createFusionGroups(Block* block, AliasDb* aliasDb, size_t min_size);

void fuseStaticSubgraphs(std::shared_ptr<Graph> graph, size_t min_size) {
  Inline(*graph);
  ReplaceWithCopy(graph);
  ReplaceWithMaybeCopy(graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);
  auto aliasDb = std::make_unique<AliasDb>(graph);
  createFusionGroups(graph->block(), aliasDb.get(), min_size);
  ConstantPooling(graph);
  ConstantPropagation(graph);
  torch::jit::EliminateDeadCode(graph);
}

static Operation createStaticSubgraphRuntime(const Node* node) {
  auto g = node->g(attr::Subgraph);
  auto module = std::make_shared<torch::jit::StaticModule>(g);
  auto num_inputs = module->num_inputs();
  return [module, num_inputs](Stack& stack) {
    RECORD_FUNCTION("Static Runtime", std::vector<c10::IValue>());
    auto inps = torch::jit::last(stack, num_inputs);
    // TODO maybe avoid call to vec
    auto outputs = (*module)(inps.vec(), {});
    torch::jit::drop(stack, num_inputs);

    if (module->num_outputs() > 1) {
      for (auto& o : outputs.toTupleRef().elements()) {
        push_one(stack, std::move(o));
      }
    } else {
      push_one(stack, std::move(outputs));
    }
    return 0;
  };
}

RegisterOperators StaticSubgraphOps({torch::jit::Operator(
    prim::StaticSubgraph,
    createStaticSubgraphRuntime,
    AliasAnalysisKind::INTERNAL_SPECIAL_CASE)});

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

static bool canHandle(Node* node) {
  for (Value* input : node->inputs()) {
    bool is_tensor = !!input->type()->cast<TensorType>();
    auto list_type = input->type()->cast<ListType>();
    bool is_list = list_type && list_type->getElementType()->cast<TupleType>();
    auto tuple_type = input->type()->cast<TupleType>();
    bool is_tuple = [&]() -> bool {
      if (!tuple_type) {
        return false;
      }
      for (auto& t : tuple_type->elements()) {
        if (!t->cast<TensorType>()) {
          return false;
        }
      }
      return true;
    }();
    if (!(is_tensor || is_list || is_tuple)) {
      if (input->node()->kind() != prim::Constant) {
        return false;
      }
    }
  }

  auto kind = node->kind();
  if (kind.is_prim()) {
    REQ(kind == prim::TupleConstruct || kind == prim::ListConstruct ||
        kind == prim::StaticSubgraph);
    if (kind == prim::TupleConstruct || kind == prim::ListConstruct) {
      for (Value* input : node->inputs()) {
        if (!input->type()->cast<TensorType>()) {
          return false;
        }
      }
    }
    return true;
  }

  // TODO add "canRunNatively" once memory management is audited
  return getOutOfPlaceOperation(node) != nullptr;
}

static bool canMerge(Node* consumer, Node* producer, AliasDb* aliasDb) {
  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

  // Symbolic checks
  REQ(canHandle(producer) || producer->kind() == prim::StaticSubgraph);
  TORCH_INTERNAL_ASSERT(
      consumer->kind() == prim::StaticSubgraph || canHandle(consumer));

  // Alias checks
  REQ(aliasDb->couldMoveBeforeTopologically(producer, consumer));

  // Ops that return aliases can only be folded if this is the only use.
  if (producer->kind() == aten::slice || producer->kind() == aten::unsqueeze ||
      producer->kind() == prim::ConstantChunk) {
    for (auto& use : producer->output(0)->uses()) {
      REQ(use.user == consumer);
    }
  }

  return true;
}

static Node* getOrCreateStaticSubgraph(Node* n, AliasDb* aliasDb) {
  if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::StaticSubgraph) {
    return n;
  }
  GRAPH_UPDATE("Creating a static subgraph::Group node from: ", *n);
  return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
      n, prim::StaticSubgraph, *aliasDb);
}

static value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == b) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
    return a->node()->isAfter(b->node());
  });
  return result;
}

static void debugDumpFusionGroup(const std::string& msg, Node* n) {
  // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
  GRAPH_DEBUG(msg, *n);
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  if (n->kind() == prim::StaticSubgraph) {
    GRAPH_DEBUG(*n->g(attr::Subgraph));
  }
}

static c10::optional<Node*> tryMerge(
    Node* fusion_group,
    Node* to_merge,
    AliasDb* aliasDb) {
  if (!canMerge(fusion_group, to_merge, aliasDb)) {
    return c10::nullopt;
  }

  std::vector<Node*> nodes_to_merge = {to_merge};

  if (to_merge->kind() == aten::cat) {
    Node* listconstruct = to_merge->input(0)->node();
    nodes_to_merge.push_back(listconstruct);
  }

  // First, try to move all the nodes we want to fuse next to the fusion
  // group.
  Node* move_point = fusion_group;
  for (auto n : nodes_to_merge) {
    GRAPH_UPDATE("Trying to move node next to fusion group: ", getHeader(n));
    if (!aliasDb->moveBeforeTopologicallyValid(n, move_point)) {
      GRAPH_UPDATE("Failed to move because of AliasDb checks!");
      return c10::nullopt;
    }
    move_point = n;
  }

  // Now all the nodes that we're going to fuse are moved next to the fusion
  // group, so we can safely merge them into the fusion group subgraph.
  fusion_group = getOrCreateStaticSubgraph(fusion_group, aliasDb);

  for (auto n : nodes_to_merge) {
    GRAPH_UPDATE("Merging ", getHeader(n));
    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        n, fusion_group, *aliasDb);
  }
  return fusion_group;
}

static std::pair<graph_node_list::iterator, bool> createFusionGroup(
    Node* fusion_node,
    AliasDb* aliasDb) {
  fusion_node = getOrCreateStaticSubgraph(fusion_node, aliasDb);

  GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
  auto inputs =
      sortReverseTopological(fusion_node->inputs(), fusion_node->owningBlock());
  for (auto input : inputs) {
    debugDumpFusionGroup("Current fusion group: ", fusion_node);
    GRAPH_DEBUG("Trying to merge: ", *input->node());
    if (auto maybe_fusion_group =
            tryMerge(fusion_node, input->node(), aliasDb)) {
      // we successfully merged, so the new group's `inputs` may have
      // changed. So rescan the new group for more merging opportunities.
      return std::make_pair(
          maybe_fusion_group.value()->reverseIterator(), true);
    }
  }

  return std::make_pair(++fusion_node->reverseIterator(), false);
}

static std::pair<graph_node_list::iterator, bool> scanNode(
    Node* n,
    AliasDb* aliasDb) {
  GRAPH_DEBUG("Considering node:", *n);

  if (!canHandle(n)) {
    return std::make_pair(++n->reverseIterator(), false);
  }

  return createFusionGroup(n, aliasDb);
}

static bool inlineIfTooSmall(Node* n, size_t min_size) {
  if (n->kind() != prim::StaticSubgraph) {
    return false;
  }
  auto subgraph = SubgraphUtils::getSubgraph(n);
  size_t num_nodes = std::distance(
      subgraph->block()->nodes().begin(), subgraph->block()->nodes().end());
  if (num_nodes < min_size) {
    GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
    SubgraphUtils::unmergeSubgraph(n);
    return true;
  }
  ConstantPooling(subgraph);
  ConstantPropagation(subgraph);
  return false;
}

static void inlineSmallFusionGroups(Block* block, size_t min_size) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      inlineSmallFusionGroups(b, min_size);
    }
    inlineIfTooSmall(n, min_size);
  }
}

void createFusionGroups(Block* block, AliasDb* aliasDb, size_t min_size) {
  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      bool changed;
      std::tie(it, changed) = scanNode(*it, aliasDb);
      any_changed |= changed;
    }
  }

  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      createFusionGroups(b, aliasDb, min_size);
    }
  }

  // Try to merge adjacent fusion groups together. Because we have only merged
  // by looking at graph inputs, without this we would not attempt to merge
  // adjacent fusion groups that don't have a dependency on each other

  std::vector<Node*> initial_fusion_groups;
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::StaticSubgraph) {
      initial_fusion_groups.push_back(n);
    }
  }

  Node* prev_fusion_group =
      !initial_fusion_groups.empty() ? initial_fusion_groups[0] : nullptr;

  for (const auto i : c10::irange(1, initial_fusion_groups.size())) {
    // Try merging the just created fusion group into the previous one.
    // If it did not work, then put the previous fusion group into
    // fusion_groups vector - we will not touch it anymore in this loop.
    // If merging succeeded, save the merged group as the "previous" fusion
    // group so that we can try to merge the next one into it.

    Node* fusion_group = initial_fusion_groups[i];
    debugDumpFusionGroup(
        "Trying to merge into the previous fusion group: ", prev_fusion_group);
    if (auto merged_fusion_group =
            tryMerge(prev_fusion_group, fusion_group, aliasDb)) {
      prev_fusion_group = *merged_fusion_group;
      debugDumpFusionGroup(
          "Successfully merged into the previous fusion group: ",
          prev_fusion_group);
    } else {
      GRAPH_DEBUG("Cannot merge into the previous fusion group");
      prev_fusion_group = fusion_group;
    }
  }
  inlineSmallFusionGroups(block, min_size);
}

static void inlineFallbackGraphs(std::shared_ptr<Graph> graph) {
  DepthFirstGraphNodeIterator it(graph);

  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::FallbackGraph) {
      SubgraphUtils::unmergeSubgraph(n);
    }
  }
}

void performTensorExprFusion(
    std::shared_ptr<Graph> graph,
    std::vector<IValue> sample_inputs) {
  // Enable TensorExpr fusion with dynamic shapes
  setTensorExprDynamicShapeFusionEnabled(true);
  GRAPH_DEBUG("Graph before tracing: ", *graph);
  auto traced_graph = TraceGraph(graph, sample_inputs);
  GRAPH_DEBUG("Graph after tracing: ", *traced_graph);
  FuseTensorExprs(
      traced_graph,
      /*min_group_size*/ 2,
      /*add_composed_op*/ true,
      /*fuse_to_dynamic_shapes*/ true);
  RemoveTensorTypeSpecializations(graph);
  inlineFallbackGraphs(traced_graph);
  graph->block()->clear();
  graph->block()->cloneFrom(traced_graph->block(), nullptr);
  GRAPH_DUMP("Graph after fusion: ", graph);
}

} // namespace torch::jit
