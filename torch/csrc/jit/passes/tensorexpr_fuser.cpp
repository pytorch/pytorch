#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

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

bool canHandle(Node* node, AliasDb& aliasDb) {
  // TODO: actually support some ops
  return false;
}

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

bool canMerge(Node* consumer, Node* producer, AliasDb& aliasDb) {
  // Only handle complete tensor types
  for (torch::jit::Value* output : consumer->outputs()) {
    REQ(output->isCompleteTensor());
  }

  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

  // Symbolic checks
  REQ(canHandle(producer, aliasDb));
  REQ(
      (canHandle(consumer, aliasDb) ||
       consumer->kind() == getTensorExprSymbol()));

  // Alias checks
  REQ(aliasDb.couldMoveAfterTopologically(consumer, producer));

  return true;
}
#undef REQ

Node* getOrCreateTensorExprSubgraph(Node* n) {
  if (n->hasAttribute(attr::Subgraph) && n->kind() == getTensorExprSymbol()) {
    return n;
  }
  return SubgraphUtils::createSingletonSubgraph(n, getTensorExprSymbol());
}

c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb) {
  GRAPH_DEBUG(
      "Trying producer ",
      producer->kind().toQualString(),
      " and consumer ",
      consumer->kind().toQualString(),
      ":\n");

  if (!canMerge(consumer, producer, aliasDb)) {
    return c10::nullopt;
  }

  consumer = getOrCreateTensorExprSubgraph(consumer);

  aliasDb.moveAfterTopologicallyValid(consumer, producer);
  SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);

  return consumer;
}

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb) {
  auto inputs =
      sortReverseTopological(consumer->inputs(), consumer->owningBlock());

  // Grab the iterator below consumer.  We'll use that to determine
  // where to resume iteration, even if consumer gets relocated within
  // the block.
  auto iter = --consumer->reverseIterator();
  for (auto input : inputs) {
    if (auto group = tryMerge(consumer, input->node(), aliasDb)) {
      // Resume iteration from where consumer is/used to be.
      return {++iter, true};
    }
  }

  // We know consumer didn't move, so skip over it.
  return {++(++iter), false};
}

Operation createTensorExprOp(const Node* node) {
  // TODO: actually compile the fusion group.
  return [](Stack& stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    return 0;
  };
}

c10::AliasAnalysisKind getAliasAnalysisOption(AliasAnalysisKind k) {
  return k;
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        getTensorExprSymbol(),
        createTensorExprOp,
        getAliasAnalysisOption(AliasAnalysisKind::PURE_FUNCTION)),
});

void fuseTensorExprs(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  AliasDb aliasDb(graph);
  auto block = graph->block();

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
        std::tie(it, changed) = scanNode(*it, aliasDb);
        any_changed |= changed;
        if (it == end) {
          worklist.pop_back();
        }
      }
    }
  }

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}

void registerTensorExprFuser() {
  static bool already_registered = false;
  if (!already_registered) {
    RegisterPass pass(fuseTensorExprs);
    already_registered = true;
  }
}
} // namespace jit
} // namespace torch
