#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {
void CreateFunctionalGraphs2(const std::shared_ptr<Graph>& graph);

/**
 * TODO:
 * [ ] Add 2nd argument to prim::TypeCheck to allow chaining them.
 * [ ] Construct prim::If for fusion group.
 * [ ] Construct non-optimized graph in else-branch.
 * [ ] Remove fuser-pass based on functional subgraphs.
 * [ ] Cleanup.
 * [ ] Fix tests.
 */

namespace tensorexpr {
bool isSupported(Node* node) {
  // TODO:
  switch (node->kind()) {
    case aten::add:
    case aten::_cast_Float:
    case aten::type_as:
    case aten::sub:
    case aten::mul:
    case aten::div:
    case aten::eq:
    case aten::ne:
    case aten::ge:
    case aten::gt:
    case aten::le:
    case aten::lt:
    case aten::pow:
    case aten::clamp:
    case aten::lerp:
    case aten::log10:
    case aten::log:
    case aten::log2:
    case aten::exp:
    case aten::erf:
    case aten::erfc:
    case aten::fmod:
    case aten::cos:
    case aten::sin:
    case aten::tan:
    case aten::acos:
    case aten::asin:
    case aten::atan:
    case aten::atan2:
    case aten::cosh:
    case aten::sinh:
    case aten::tanh:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::floor:
    case aten::ceil:
    case aten::round:
    case aten::trunc:
    case aten::threshold:
    case aten::remainder:
    case aten::sigmoid:
    case aten::relu:
    case aten::addcmul:
    case aten::neg:
    case aten::reciprocal:
    case aten::expm1:
    case aten::lgamma:

    case prim::ConstantChunk:
      // TODO: The following ops require proper shape inferrence:
      // case aten::cat:
      // case prim::ListConstruct:
      // case aten::slice:
      // case aten::unsqueeze:

    case aten::frac:
    // TODO: uncomment once we can handle rand+broadcasts
    // case aten::rand_like:
    case aten::_sigmoid_backward:
    case aten::_tanh_backward:
    case aten::__and__:
    case aten::__or__:
    case aten::__xor__:
    case aten::__lshift__:
    case aten::__rshift__:
    case aten::where:
      return true;
    // Operators that can be both elementwise or reductions:
    case aten::min:
    case aten::max:
      if (node->inputs().size() != 2) {
        return false;
      }
      if (!node->inputs()[0]->type()->cast<TensorType>() ||
          !node->inputs()[1]->type()->cast<TensorType>()) {
        return false;
      }
      return true;
    default:
      return false;
  }
}
} // namespace tensorexpr

static bool texpr_fuser_enabled_ = false;
void setTensorExprFuserEnabled(bool val) {
  texpr_fuser_enabled_ = val;
}

bool tensorExprFuserEnabled() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR");
  if (!enable_c_str) {
    return texpr_fuser_enabled_;
  }
  if (std::string(enable_c_str) == "0") {
    return false;
  }
  return true;
}

const Symbol& getTensorExprSymbol() {
  static Symbol s = Symbol::fromQualString("tensorexpr::Group");
  return s;
}

value_list sortReverseTopological(
    ArrayRef<torch::jit::Value*> inputs,
    torch::jit::Block* block) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(
      result.begin(),
      result.end(),
      [&](torch::jit::Value* a, torch::jit::Value* b) {
        return a->node()->isAfter(b->node());
      });
  return result;
}

bool allShapesAreKnown(Value* v) {
  if (!v->type()->cast<TensorType>()) {
    return true;
  }
  return v->isCompleteTensor();
}

bool allShapesAreKnown(Node* node) {
  // TODO: Relax the checks to support dynamic shapes
  for (torch::jit::Value* output : node->outputs()) {
    if (!allShapesAreKnown(output)) {
      return false;
    }
  }
  for (torch::jit::Value* input : node->inputs()) {
    if (!allShapesAreKnown(input)) {
      return false;
    }
  }
  return true;
}

bool canHandle(Node* node) {
  if (node->kind() == prim::Constant) {
    if (node->output()->type()->cast<TensorType>()) {
      // TODO: add support for tensor constants.
      return false;
    }
    return true;
  }
  if (node->kind() == prim::profile) {
    return true;
  }
  if (node->kind() == prim::Loop) {
    return false; // TODO
  }
  //   if (!allShapesAreKnown(node)) {
  //     return false;
  //   }

  // Don't include nodes whose inputs are tensor constants - we cannot handle
  // them at the moment.
  // TODO: actually support tensor constants and remove this.
  for (torch::jit::Value* input : node->inputs()) {
    if (input->node()->kind() == prim::Constant &&
        input->type()->cast<TensorType>()) {
      return false;
    }
  }
  return tensorexpr::isSupported(node);
}

bool canHandle(Node* node, AliasDb& aliasDb) {
  return canHandle(node);
}

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

bool canMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb,
    const std::unordered_map<Value*, TensorTypePtr>& value_types) {
  // Only handle complete tensor types
  for (torch::jit::Value* output : consumer->outputs()) {
    REQ(output->isCompleteTensor() ||
        (value_types.count(output) && value_types.at(output)->isComplete()));
  }

  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

  // Symbolic checks
  REQ(canHandle(producer, aliasDb));
  REQ(
      (canHandle(consumer, aliasDb) ||
       consumer->kind() == getTensorExprSymbol()));

  // Alias checks
  REQ(aliasDb.couldMoveBeforeTopologically(producer, consumer));

  // Ops that return aliases can only be folded if this is the only use.
  if (producer->kind() == aten::slice || producer->kind() == aten::unsqueeze ||
      producer->kind() == prim::ConstantChunk) {
    for (auto& use : producer->output(0)->uses()) {
      REQ(use.user == consumer);
    }
  }

  if (!consumer->hasAttribute(attr::Subgraph) &&
      consumer->kind() != getTensorExprSymbol()) {
    // Don't initiate a fusion group from prim::ListConstruct
    REQ(consumer->kind() != prim::ListConstruct);
    REQ(consumer->kind() != aten::slice);
    REQ(consumer->kind() != aten::unsqueeze);
    REQ(consumer->kind() != prim::ConstantChunk);

    // Don't initiate a fusion group just for a constant operand
    REQ(producer->kind() != prim::Constant);
  }

  if (producer->kind() == aten::cat) {
    REQ(producer->inputs()[0]->node()->kind() == prim::ListConstruct);
    REQ(producer->inputs()[0]->uses().size() == 1);
    REQ(producer->inputs()[1]->node()->kind() == prim::Constant);
  } else if (consumer->kind() == aten::cat) {
    REQ(consumer->inputs()[0]->node()->kind() == prim::ListConstruct);
    REQ(consumer->inputs()[0]->uses().size() == 1);
    REQ(consumer->inputs()[1]->node()->kind() == prim::Constant);
  }

  return true;
}
#undef REQ

Node* getOrCreateTensorExprSubgraph(Node* n) {
  if (n->hasAttribute(attr::Subgraph) && n->kind() == getTensorExprSymbol()) {
    return n;
  }
  auto te_group =
      SubgraphUtils::createSingletonSubgraph(n, getTensorExprSymbol());
  GRAPH_UPDATE("getOrCreateTensorExprSubgraph: ", *te_group);
  return te_group;
}

c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb,
    std::unordered_map<Value*, TensorTypePtr>& value_types) {
  GRAPH_DEBUG(
      "Trying producer ",
      getHeader(producer),
      " and consumer ",
      getHeader(consumer),
      ":\n");

  if (!canMerge(consumer, producer, aliasDb, value_types)) {
    return c10::nullopt;
  }

  std::vector<c10::optional<TensorTypePtr>> consumer_type_info;
  for (int idx = 0; idx < consumer->outputs().size(); idx++) {
    if (value_types.count(consumer->output(idx))) {
      consumer_type_info.push_back(value_types.at(consumer->output(idx)));
    } else {
      consumer_type_info.push_back(c10::nullopt);
    }
  }
  std::vector<c10::optional<TensorTypePtr>> producer_inputs_type_info;
  std::vector<c10::optional<TensorTypePtr>> producer_outputs_type_info;
  for (int idx = 0; idx < producer->outputs().size(); idx++) {
    if (value_types.count(producer->output(idx))) {
      producer_outputs_type_info.push_back(value_types.at(producer->output(idx)));
    } else {
      producer_outputs_type_info.push_back(c10::nullopt);
    }
  }
  for (int idx = 0; idx < producer->inputs().size(); idx++) {
    if (value_types.count(producer->input(idx))) {
      producer_inputs_type_info.push_back(value_types.at(producer->input(idx)));
    } else {
      producer_inputs_type_info.push_back(c10::nullopt);
    }
  }
  Node* te_group = getOrCreateTensorExprSubgraph(consumer);
  // Propagate profiling info to the newly create node representing the TE
  // fusion group
  if (te_group != consumer) {
    for (int idx = 0; idx < te_group->outputs().size(); idx++) {
      if (consumer_type_info[idx]) {
        value_types[te_group->output(idx)] = *consumer_type_info[idx];
        GRAPH_DEBUG(
            "%",
            te_group->output(idx)->debugName(),
            " -> ",
            *value_types[te_group->output(idx)],
            "\n");
      } else {
        GRAPH_DEBUG(
            "No shape info for TE group output %",
            te_group->output(idx)->debugName(),
            "!\n");
      }
    }
    consumer = te_group;
  }

  if (producer->kind() == aten::cat) {
    Node* listconstruct = producer->inputs()[0]->node();

    aliasDb.moveBeforeTopologicallyValid(producer, consumer);
    GRAPH_UPDATE(
        "Merging ", getHeader(producer), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);

    aliasDb.moveBeforeTopologicallyValid(listconstruct, consumer);
    GRAPH_UPDATE(
        "Merging ", getHeader(listconstruct), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(listconstruct, consumer);
  } else {
    aliasDb.moveBeforeTopologicallyValid(producer, consumer);
    GRAPH_UPDATE(
        "Merging ", getHeader(producer), " into ", getHeader(consumer));
    Node* mergedNode = SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    if (mergedNode) {
      for (int idx = 0; idx < mergedNode->outputs().size(); idx++) {
        Value* v = mergedNode->output(idx);
        if (producer_outputs_type_info[idx]) {
          value_types[v] = *producer_outputs_type_info[idx];
          GRAPH_DEBUG("%", v->debugName(), " -> ", *value_types.at(v), "\n");
        } else {
          GRAPH_DEBUG(
              "No shape info for TE group output %", v->debugName(), "!\n");
        }
      }
      for (int idx = 0; idx < mergedNode->inputs().size(); idx++) {
        Value* v = mergedNode->input(idx);
        if (producer_inputs_type_info[idx]) {
          value_types[v] = *producer_inputs_type_info[idx];
          GRAPH_DEBUG("%", v->debugName(), " -> ", *value_types.at(v), "\n");
        } else {
          GRAPH_DEBUG(
              "No shape info for TE group input %", v->debugName(), "!\n");
        }
      }
    }
  }

  return consumer;
}

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb,
    std::unordered_map<Value*, TensorTypePtr>& value_types) {
  auto inputs =
      sortReverseTopological(consumer->inputs(), consumer->owningBlock());

  // Grab the iterator below consumer.  We'll use that to determine
  // where to resume iteration, even if consumer gets relocated within
  // the block.
  auto iter = --consumer->reverseIterator();
  for (auto input : inputs) {
    if (auto group = tryMerge(consumer, input->node(), aliasDb, value_types)) {
      // Resume iteration from where consumer is/used to be.
      return {++iter, true};
    }
  }

  // We know consumer didn't move, so skip over it.
  return {++(++iter), false};
}

void removeProfilingNodes_(
    Block* b,
    std::unordered_map<Value*, TensorTypePtr>& value_types) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      for (auto o : it->outputs()) {
        value_types.erase(o);
      }
      if (it->outputs().size()) {
        //         it->input()->setType(it->output()->type());
        it->output()->replaceAllUsesWith(it->input());
      }
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfilingNodes_(ib, value_types);
      }
    }
  }
}

void findValuesWithKnownSizes_(
    Block* block,
    std::unordered_map<Value*, TensorTypePtr>& value_types) {
  auto reverse_iter = block->nodes().reverse();
  for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
    Node* n = *it++;

    for (torch::jit::Value* output : n->outputs()) {
      if (allShapesAreKnown(output)) {
        if (auto tensor_ty = output->type()->cast<TensorType>()) {
          value_types[output] = tensor_ty;
        }
      }
    }

    // constants get copied into the graph
    if (n->kind() == prim::profile && n->outputs().size() == 1 &&
        allShapesAreKnown(n->output())) {
      if (auto tensor_ty = n->output()->type()->cast<TensorType>()) {
        value_types[n->input()] = tensor_ty;
      }
    }

    for (Block* b : n->blocks()) {
      findValuesWithKnownSizes_(b, value_types);
    }
  }
}

/*
*** INSERT GUARDS TRANSFORMATION ***

Original IR:
```
  %a, %b = tensorexpr::Group_0(%x, %y)
  ...
with tensorexpr::Group_0 = graph(%p : Tensor, %q : Tensor):
  %v = aten::gt(%p, %q)
  %w = aten::where(%v, %p, %q)
  return (%v, %w)
```

Should be transformed to:
```
  %x1 : Double(...), %check_x : bool = prim::TypeCheck(%x, true)
  %y1 : Double(...), %check_xy : bool = prim::TypeCheck(%y, %check_x)
  %a1, %b1 = prim::If(%check_xy)
    block0():
      %a, %b = tensorexpr::Group_0(%x1, %y1)
      -> (%a, %b)
    block1():
      %v1 = aten::gt(%x, %y)
      %w1 = aten::where(%v1, %x, %y)
      -> (%v1, %w1)
  ...
with tensorexpr::Group_0 = graph(%p : Double(...), %q : Double(...)):
  %v = aten::gt(%p, %q)
  %w = aten::where(%v, %p, %q)
  return (%v, %w)
```
 */
void insertGuards(
    Block* b,
    std::unordered_map<Value*, TensorTypePtr>& value_types) {
  for (auto it = b->nodes().begin(); it != b->nodes().end();) {
    if (it->kind() != getTensorExprSymbol()) {
      for (Block* ib : it->blocks()) {
        insertGuards(ib, value_types);
      }
      it++;
      continue;
    }
    GRAPH_DEBUG("Inserting guards for the node:\n", **it);

    // Fixup types in subgraph inputs
    auto te_g = it->g(attr::Subgraph);
    for (size_t idx = 0; idx < te_g->inputs().size(); idx++) {
      auto inp = it->input(idx);
      if (value_types.count(inp) && value_types.at(inp)->isComplete()) {
        te_g->inputs().at(idx)->setType(value_types.at(inp));
      }
    }

    // Add guard for inputs
    Value* check_result = nullptr;
    std::unordered_map<Value*, Value*> checked_values;
    Value* cond = b->owningGraph()->insertConstant(true);
    for (auto inp : it->inputs()) {
      if (value_types.count(inp) && value_types.at(inp)->isComplete()) {
        auto guard = b->owningGraph()->create(prim::TypeCheck, {inp, cond}, 2);
        auto go0 = guard->output(0);
        cond = guard->output(1);
        check_result = guard->output(1);
        go0->setType(value_types.at(inp));
        check_result->setType(BoolType::get());
        guard->insertBefore(*it);
        checked_values[inp] = go0;
      }
    }
    // checked_values = {x -> x1, y -> y1}

    if (!check_result) {
      GRAPH_DEBUG("No checks were needed.\n");
      continue;
    }
    auto fusion_group = *it;
    it++;
    auto versioning_if = b->owningGraph()->create(
        prim::If, {check_result}, fusion_group->outputs().size());

    for (size_t i = 0; i < fusion_group->outputs().size(); ++i) {
      fusion_group->output(i)->replaceAllUsesWith(versioning_if->output(i));
    }
    versioning_if->insertAfter(fusion_group);
    auto true_block = versioning_if->addBlock();
    auto false_block = versioning_if->addBlock();
    fusion_group->moveBefore(true_block->return_node());

    WithInsertPoint guard(false_block->return_node());
    const auto subgraphOutputs = insertGraph(
        *b->owningGraph(),
        *fusion_group->g(attr::Subgraph),
        fusion_group->inputs());

    for (size_t i = 0; i < fusion_group->inputs().size(); ++i) {
      Value* inp = fusion_group->input(i);
      if (value_types.count(inp) && value_types.at(inp)->isComplete()) {
        fusion_group->replaceInput(
            i, checked_values.at(fusion_group->input(i)));
      }
    }
    for (size_t i = 0; i < fusion_group->outputs().size(); ++i) {
      true_block->registerOutput(fusion_group->output(i));
      false_block->registerOutput(subgraphOutputs[i]);
      if (value_types.count(fusion_group->output(i))) {
        value_types[versioning_if->output(i)] =
            value_types.at(fusion_group->output(i));
        GRAPH_DEBUG(
            "%",
            versioning_if->output(i)->debugName(),
            " -> ",
            *value_types.at(versioning_if->output(i)),
            "\n");
      } else {
        GRAPH_DEBUG(
            "No shape info for %",
            versioning_if->output(i)->debugName(),
            "!\n");
      }
    }
      GRAPH_DEBUG("Guarded:\n", *versioning_if);
  }
}

void FuseTensorExprs(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  std::unordered_map<Value*, TensorTypePtr> value_types;

  findValuesWithKnownSizes_(graph->block(), value_types);
  removeProfilingNodes_(graph->block(), value_types);
  GRAPH_DUMP("After removing profiling nodes:", graph);
  for (const auto& kv : value_types) {
    GRAPH_DEBUG("%", kv.first->debugName(), " -> ", *kv.second, "\n");
  }

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  AliasDb aliasDb(graph);
  auto block = graph->block();
  //   CreateFunctionalGraphs2(graph);

  std::vector<std::pair<graph_node_list_iterator, graph_node_list_iterator>>
      worklist;
  std::unordered_set<torch::jit::Block*> visited_blocks;

  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    worklist.push_back({block->nodes().rbegin(), block->nodes().rend()});

    while (worklist.size()) {
      auto& it = worklist.back().first;
      auto end = worklist.back().second;

      if (it->blocks().size()) {
        Node* n = *it;
        ++it;

        if (it == end) {
          worklist.pop_back();
        }

        for (auto b : n->blocks()) {
          if (!visited_blocks.count(b)) {
            worklist.push_back({b->nodes().rbegin(), b->nodes().rend()});
            visited_blocks.insert(b);
          }
        }
      } else {
        bool changed;
        std::tie(it, changed) = scanNode(*it, aliasDb, value_types);
        any_changed |= changed;
        if (it == end) {
          worklist.pop_back();
        }
      }
    }
  }
  GRAPH_DUMP("Before inserting checks:\n", graph);
  insertGuards(graph->block(), value_types);
  GRAPH_DUMP("After inserting checks:\n", graph);

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}

Operation createTensorExprOp(const Node* node) {
  auto kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(node->g(attr::Subgraph));
  return [kernel](Stack* stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    if (!tensorexpr::fallbackAllowed()) {
      kernel->run(*stack);
      return 0;
    }

    try {
      kernel->run(*stack);
    } catch (const std::runtime_error& e) {
      kernel->fallback(*stack);
    }
    return 0;
  };
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        getTensorExprSymbol(),
        createTensorExprOp,
        AliasAnalysisKind::PURE_FUNCTION),
});

} // namespace jit
} // namespace torch
