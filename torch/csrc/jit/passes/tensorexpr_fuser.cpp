#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {

static bool texpr_fuser_enabled = true;
void setTensorExprFuserEnabled(bool val) {
  texpr_fuser_enabled = val;
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
    case aten::min:
    case aten::max:
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
    case prim::ConstantChunk:
    case aten::cat:
    case prim::ListConstruct:
    case aten::sigmoid:
    case aten::relu:
    case aten::addcmul:
    case aten::neg:
    case aten::reciprocal:
    case aten::expm1:
    case aten::lgamma:
    case aten::slice:
    case aten::unsqueeze:
    case aten::frac:
    case aten::rand_like:
    case aten::_sigmoid_backward:
    case aten::_tanh_backward:
    case aten::__and__:
    case aten::__or__:
    case aten::__xor__:
    case aten::__lshift__:
    case aten::__rshift__:
    case aten::where:
      return true;
    default:
      return false;
  }
}

bool canHandle(Node* node, AliasDb& aliasDb) {
  if (node->kind() == prim::Constant) {
    return true;
  }
  if (node->kind() == prim::Loop) {
    return false; // TODO
  }
  return isSupported(node);
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
    AliasDb& aliasDb) {
  GRAPH_DEBUG(
      "Trying producer ",
      getHeader(producer),
      " and consumer ",
      getHeader(consumer),
      ":\n");

  if (!canMerge(consumer, producer, aliasDb)) {
    return c10::nullopt;
  }

  consumer = getOrCreateTensorExprSubgraph(consumer);

  if (producer->kind() == aten::cat) {
    Node* listconstruct = producer->inputs()[0]->node();

    aliasDb.moveAfterTopologicallyValid(consumer, producer);
    GRAPH_UPDATE(
        "Merging ", getHeader(producer), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);

    aliasDb.moveAfterTopologicallyValid(consumer, listconstruct);
    GRAPH_UPDATE(
        "Merging ", getHeader(listconstruct), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(listconstruct, consumer);
  } else {
    aliasDb.moveAfterTopologicallyValid(consumer, producer);
    GRAPH_UPDATE(
        "Merging ", getHeader(producer), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

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

void fuseTensorExprs(std::shared_ptr<Graph>& graph) {
  if (!texpr_fuser_enabled) {
    return;
  }
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

Operation createTensorExprOp(const Node* node) {
  auto kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(node->g(attr::Subgraph));
  return [kernel](Stack& stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    try {
      kernel->run(stack);
    } catch (const std::runtime_error& e) {
      kernel->fallback(stack);
    }
    return 0;
  };
}

c10::OperatorOptions getAliasAnalysisOption(AliasAnalysisKind k) {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(k);
  return options;
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        getTensorExprSymbol(),
        createTensorExprOp,
        getAliasAnalysisOption(AliasAnalysisKind::PURE_FUNCTION)),
});

void registerTensorExprFuser() {
  static bool already_registered = false;
  if (!already_registered) {
    RegisterPass pass(fuseTensorExprs);
    already_registered = true;
  }
}
} // namespace jit
} // namespace torch
