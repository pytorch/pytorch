#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

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

Node* getOrCreateTensorExprSubgraph(Node* n) {
  if (n->hasAttribute(attr::Subgraph) && n->kind() == getTensorExprSymbol()) {
    return n;
  }
  GRAPH_UPDATE("Creating a tensorexpr::Group node from: ", *n);
  auto te_group =
      SubgraphUtils::createSingletonSubgraph(n, getTensorExprSymbol());
  return te_group;
}

struct nodesComparator {
  bool operator()(Node* a, Node* b) const {
    return a->isAfter(b);
  }
};

class TensorExprFuser {
 public:
  TensorExprFuser(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    createFusionGroups(graph_->block());
  }

 private:
  // Add unvisited input nodes to the queue for further merging into the fusion
  // group.
  void updateQueue(
      Node* fusion_group,
      std::set<Node*, nodesComparator>& queue,
      const std::unordered_set<Node*>& visited) {
    for (auto input : fusion_group->inputs()) {
      if (!visited.count(input->node())) {
        queue.insert(input->node());
      }
    }
  }

  // Create a fusion group starting from the node N.
  // We then try to pull inputs into the fusion group and repeat that process
  // until there is nothing we can pull in.
  Node* createFusionGroup(Node* n) {
    // Queue of the nodes we should consider for merging into the fusion groups
    // (those nodes are usually inputs of the fusion group).
    // We use an ordered set here to visit them in the right order: the fusion
    // group is closer to the end of the block and we are trying to pull later
    // nodes first.
    // NB: the order in the list in theory could stale if we move nodes around.
    // However, this should only happen to the nodes we could not fuse, and
    // hence it should not be a problem.
    std::set<Node*, nodesComparator> queue;
    std::unordered_set<Node*> visited_nodes;

    Node* fusion_group = n;
    if (minSubgraphSize_ == 1) {
      fusion_group = getOrCreateTensorExprSubgraph(n);
    }

    updateQueue(fusion_group, queue, visited_nodes);

    GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
    while (!queue.empty()) {
      GRAPH_DEBUG("Current fusion group: ", *fusion_group);
      GRAPH_DEBUG(queue.size(), " nodes are in the queue.\n");

      Node* input_node = *queue.begin();
      queue.erase(queue.begin());

      GRAPH_DEBUG("Trying to merge: ", *input_node);
      fusion_group = tryMerge(fusion_group, input_node);
      visited_nodes.insert(input_node);
      updateQueue(fusion_group, queue, visited_nodes);
    }

    return fusion_group;
  }

  // Merge fusible nodes into subgraphs in tensorexpr::Group nodes.
  void createFusionGroups(Block* block) {
    std::vector<Node*> fusion_groups;
    auto reverse_iter = block->nodes().reverse();
    for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
      Node* n = *it;

      for (auto b : n->blocks()) {
        createFusionGroups(b);
      }

      if (!canHandle(n)) {
        it++;
        continue;
      }
      // There are some nodes that we can support, but we don't want to start a
      // fusion group from - skip them.
      if (n->kind() == prim::ListConstruct || n->kind() == aten::slice ||
          n->kind() == aten::unsqueeze || n->kind() == prim::ConstantChunk ||
          n->kind() == prim::Constant) {
        it++;
        continue;
      }

      Node* fusion_group = createFusionGroup(n);
      fusion_groups.push_back(fusion_group);
      it = fusion_group->reverseIterator();
      it++;
    }

    for (auto n : fusion_groups) {
      inlineIfTooSmall(n);
    }
  }

  size_t blockSize(Block* block) {
    size_t num = 0;
    for (Node* n : block->nodes()) {
      // Don't count prim::Constants and prim::ListConstructs as these are nodes
      // we only pull in along with another, "main", node. E.g. the
      // ListConstruct nodes would also be pulled into a fusion group if they
      // are inputs of an aten::cat node.
      if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
        continue;
      }
      for (Block* b : n->blocks()) {
        num += blockSize(b);
      }
      num++;
    }
    return num;
  }

  bool inlineIfTooSmall(Node* n) {
    if (n->kind() != getTensorExprSymbol()) {
      return false;
    }
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_modes = blockSize(subgraph->block());
    if (num_modes < minSubgraphSize_) {
      GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    return false;
  }

  Node* tryMerge(Node* fusion_group, Node* to_merge) {
    if (!canMerge(fusion_group, to_merge)) {
      return fusion_group;
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
      if (!aliasDb_->moveBeforeTopologicallyValid(n, move_point)) {
        GRAPH_UPDATE("Failed to move because of AliasDB checks!");
        return fusion_group;
      }
      move_point = n;
    }

    // Now all the nodes that we're going to fuse are moved next to the fusion
    // group, so we can safely merge them into the fusion group subgraph.
    fusion_group = getOrCreateTensorExprSubgraph(fusion_group);
    for (auto n : nodes_to_merge) {
      GRAPH_UPDATE("Merging ", getHeader(n));
      SubgraphUtils::mergeNodeIntoSubgraph(n, fusion_group);
    }
    return fusion_group;
  }

  bool allShapesAreKnown(Value* v) {
    if (!v->type()->cast<TensorType>()) {
      return true;
    }
    return v->isCompleteTensor();
  }

  bool allShapesAreKnown(Node* node) {
    // TODO: Relax the checks to support dynamic shapes
    for (Value* input : node->inputs()) {
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
    if (!allShapesAreKnown(node)) {
      return false;
    }

    // Don't include nodes whose inputs are tensor constants - we cannot handle
    // them at the moment.
    // TODO: actually support tensor constants and remove this.
    for (Value* input : node->inputs()) {
      if (input->node()->kind() == prim::Constant &&
          input->type()->cast<TensorType>()) {
        return false;
      }
    }
    return tensorexpr::isSupported(node);
  }

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

  bool canMerge(Node* consumer, Node* producer) {
    // Only fuse within a block
    REQ(consumer->owningBlock() == producer->owningBlock());

    // Symbolic checks
    REQ(canHandle(producer));
    TORCH_INTERNAL_ASSERT(
        canHandle(consumer) || consumer->kind() == getTensorExprSymbol());

    // Alias checks
    REQ(aliasDb_->couldMoveBeforeTopologically(producer, consumer));

    // Ops that return aliases can only be folded if this is the only use.
    if (producer->kind() == aten::slice ||
        producer->kind() == aten::unsqueeze ||
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
      REQ(producer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(producer->input(0)->uses().size() == 1);
      REQ(producer->input(1)->node()->kind() == prim::Constant);
    } else if (consumer->kind() == aten::cat) {
      REQ(consumer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(consumer->input(0)->uses().size() == 1);
      REQ(consumer->input(1)->node()->kind() == prim::Constant);
    }

    return true;
  }
#undef REQ

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  size_t minSubgraphSize_ = 2;
};

void FuseTensorExprs(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  TensorExprFuser fuser(graph);
  fuser.run();

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
