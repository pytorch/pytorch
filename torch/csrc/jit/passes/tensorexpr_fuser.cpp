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

bool isSupportedForBlock(Node* node) {
  switch (node->kind()) {
    case aten::add:
    case aten::mul:
      return true;
    default:
      return false;
  }
}

namespace tensorexpr {
bool isSupported(Node* node) {
  // For Block codegen we allow limited ops.
  if (tensorexpr::getTEGenerateBlockCode()) {
    return isSupportedForBlock(node);
  }
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
    case aten::slice:
      // TODO: Shape inference is not implemented for this op yet
      return false;
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

struct nodesComparator {
  bool operator()(Node* a, Node* b) const {
    return a->isAfter(b);
  }
};

class TensorExprFuser {
 public:
  TensorExprFuser(std::shared_ptr<Graph> graph, size_t min_group_size)
      : graph_(std::move(graph)), min_group_size_(min_group_size) {}

  void run() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    fillTypeInfoMap(graph_->block());
    printTypeInfoMap();
    removeProfilingNodes(graph_->block());
    printTypeInfoMap();
    GRAPH_DUMP("After removing profiling nodes: ", graph_);
    createFusionGroups(graph_->block());
    printTypeInfoMap();
    GRAPH_DUMP("After creating fusion groups: ", graph_);
    guardFusionGroups(graph_->block());
  }

 private:
  Node* getOrCreateTensorExprSubgraph(Node* n) {
    if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::TensorExprGroup) {
      return n;
    }
    std::unordered_map<Value*, Value*> vmap;
    GRAPH_UPDATE("Creating a prim::TensorExprGroup node from: ", *n);
    auto te_group =
        SubgraphUtils::createSingletonSubgraph(n, prim::TensorExprGroup, vmap);
    updateTypeinfoMapWithVmap(vmap);
    return te_group;
  }

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
    if (min_group_size_ == 1) {
      fusion_group = getOrCreateTensorExprSubgraph(n);
    }

    updateQueue(fusion_group, queue, visited_nodes);

    GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
    while (!queue.empty()) {
      debugDumpFusionGroup("Current fusion group: ", fusion_group);
      GRAPH_DEBUG(queue.size(), " nodes are in the queue.\n");

      Node* input_node = *queue.begin();
      queue.erase(queue.begin());

      GRAPH_DEBUG("Trying to merge: ", *input_node);
      fusion_group = tryMerge(fusion_group, input_node);
      visited_nodes.insert(input_node);
      updateQueue(fusion_group, queue, visited_nodes);
    }

    // Update typeinfo for inputs and outputs of the fusion group
    if (fusion_group->kind() == prim::TensorExprGroup) {
      auto subgraph = SubgraphUtils::getSubgraph(fusion_group);
      for (size_t idx = 0; idx < subgraph->inputs().size(); idx++) {
        if (typeinfo_map_.count(subgraph->inputs()[idx])) {
          auto ty = typeinfo_map_[subgraph->inputs()[idx]];
          typeinfo_map_[fusion_group->input(idx)] = ty;
          subgraph->inputs()[idx]->setType(ty);
          GRAPH_DEBUG(
              "TYPEINFO ADD: %",
              fusion_group->input(idx)->debugName(),
              " --> ",
              *ty);
        }
      }
      for (size_t idx = 0; idx < subgraph->outputs().size(); idx++) {
        if (typeinfo_map_.count(subgraph->outputs()[idx])) {
          auto ty = typeinfo_map_[subgraph->outputs()[idx]];
          typeinfo_map_[fusion_group->output(idx)] = ty;
          GRAPH_DEBUG(
              "TYPEINFO ADD: %",
              fusion_group->output(idx)->debugName(),
              " --> ",
              *ty);
        }
      }
    }

    return fusion_group;
  }

  static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    GRAPH_DEBUG(msg, *n);
    if (n->kind() == prim::TensorExprGroup) {
      GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
  }

  // Merge fusible nodes into subgraphs in prim::TensorExprGroup nodes.
  void createFusionGroups(Block* block) {
    std::vector<Node*> fusion_groups;
    auto reverse_iter = block->nodes().reverse();
    for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
      Node* n = *it;
      GRAPH_DEBUG("Considering node:", *n)

      for (Block* b : n->blocks()) {
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

    for (Node* n : fusion_groups) {
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
    if (n->kind() != prim::TensorExprGroup) {
      return false;
    }
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_modes = blockSize(subgraph->block());
    if (num_modes < min_group_size_) {
      GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    return false;
  }

  void updateTypeinfoMapWithVmap(
      const std::unordered_map<Value*, Value*>& vmap) {
    for (const auto& kv : vmap) {
      if (typeinfo_map_.count(kv.first)) {
        auto ty = typeinfo_map_.at(kv.first);
        typeinfo_map_[kv.second] = ty;
        GRAPH_DEBUG("TYPEINFO UPDATE: %", kv.second->debugName(), " --> ", *ty);
      }
    }
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
      std::unordered_map<Value*, Value*> vmap;
      SubgraphUtils::mergeNodeIntoSubgraph(n, fusion_group, vmap);
      updateTypeinfoMapWithVmap(vmap);
    }
    return fusion_group;
  }

  bool allShapesAreKnown(Value* v) {
    if (!v->type()->cast<TensorType>()) {
      return true;
    }
    if (typeinfo_map_.count(v)) {
      return typeinfo_map_.at(v)->isComplete();
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
      // TODO: add support for tensor constants.
      return false;
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
        canHandle(consumer) || consumer->kind() == prim::TensorExprGroup);

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
        consumer->kind() != prim::TensorExprGroup) {
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

  void fillTypeInfoMap(Block* block) {
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::profile) {
        if (auto tensor_ty = n->ty(attr::profiled_type)->cast<TensorType>()) {
          GRAPH_DEBUG(
              "TYPEINFO ADD: %", n->input()->debugName(), " --> ", *tensor_ty);
          typeinfo_map_[n->input()] = tensor_ty;
        }
      }

      for (Block* b : n->blocks()) {
        fillTypeInfoMap(b);
      }
    }
  }
  void printTypeInfoMap() {
    GRAPH_DEBUG("Typeinfo map:");
    for (const auto& kv : typeinfo_map_) {
      // GRAPH_DEBUG("%", kv.first->debugName(), " --> ", *kv.second);
      GRAPH_DEBUG("%??? --> ", *kv.second);
    }
  }

  void removeProfilingNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      if (it->kind() == prim::profile) {
        GRAPH_DEBUG("TYPEINFO ERASE: %", it->output()->debugName());
        typeinfo_map_.erase(it->output());
        it->output()->replaceAllUsesWith(it->input());
        it.destroyCurrent();
      } else {
        for (Block* ib : it->blocks()) {
          removeProfilingNodes(ib);
        }
      }
    }
  }

  void guardFusionGroup(Node* fusion_group) {
    GRAPH_DEBUG("Inserting a typecheck guard for a node", *fusion_group);
    auto subgraph = SubgraphUtils::getSubgraph(fusion_group);

    // Fixup types of the subgraph inputs
    std::vector<Value*> inputs_to_check;
    for (size_t idx = 0; idx < subgraph->inputs().size(); idx++) {
      // TODO: If a type of the input was explicitly set in IR (in constrast to
      // us expecting it from a hint given by a prim::profile node), we don't
      // need to check that input in the typecheck.

      // We only check inputs of the fusion group and expect NNC to infer
      // intermediates and outputs shapes
      if (typeinfo_map_.count(subgraph->inputs()[idx])) {
        subgraph->inputs()[idx]->setType(
            typeinfo_map_.at(subgraph->inputs()[idx]));
        inputs_to_check.push_back(fusion_group->input(idx));
      }
    }
    if (!inputs_to_check.size()) {
      return;
    }

    // Add prim::TypeCheck node
    //
    // TypeCheck nodes  look like the following:
    //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
    //   prim::TypeCheck(%inp1 : Tensor, %inp2 : Tensor)
    //
    // They have N inputs whose types we are going to check and N+1 outputs. The
    // first N outputs specify expected types and N+1-th output holds the result
    // of the check (bool).
    Node* typecheck_node =
        fusion_group->owningGraph()
            ->create(
                prim::TypeCheck, inputs_to_check, inputs_to_check.size() + 1)
            ->insertBefore(fusion_group);
    Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

    // Fixup types of the typecheck node outputs
    std::unordered_map<Value*, Value*> typechecked_inputs;
    size_t output_idx = 0;
    for (size_t idx = 0; idx < subgraph->inputs().size(); idx++) {
      if (typeinfo_map_.count(subgraph->inputs()[idx])) {
        typechecked_inputs[fusion_group->input(idx)] =
            typecheck_node->output(output_idx);
        typecheck_node->output(output_idx++)
            ->setType(typeinfo_map_.at(subgraph->inputs()[idx]));
      } else if (
          auto tt = subgraph->inputs()[idx]->type()->cast<TensorType>()) {
        TORCH_INTERNAL_ASSERT(tt->isComplete());
      }
    }
    typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());

    // Insert if
    auto versioning_if =
        fusion_group->owningGraph()
            ->create(
                prim::If, {typecheck_result}, fusion_group->outputs().size())
            ->insertAfter(typecheck_node);
    for (size_t idx = 0; idx < fusion_group->outputs().size(); ++idx) {
      versioning_if->output(idx)->setType(fusion_group->output(idx)->type());
      fusion_group->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
    }
    auto true_block = versioning_if->addBlock();
    auto false_block = versioning_if->addBlock();

    // Fill in the false block. It should contain the unoptimized
    // copy of the fused subgraph.
    WithInsertPoint guard(false_block->return_node());
    const auto subgraph_outputs = insertGraph(
        *fusion_group->owningGraph(), *subgraph, fusion_group->inputs());
    for (Value* output : subgraph_outputs) {
      false_block->registerOutput(output);
    }

    // Fill in the true block. It has all inputs type-checked and its
    // body should be the fusion group node.
    fusion_group->moveBefore(true_block->return_node());
    for (size_t idx = 0; idx < fusion_group->inputs().size(); ++idx) {
      if (typechecked_inputs.count(fusion_group->input(idx))) {
        fusion_group->replaceInput(
            idx, typechecked_inputs.at(fusion_group->input(idx)));
      }
    }
    for (Value* output : fusion_group->outputs()) {
      true_block->registerOutput(output);
    }
  }

  void guardFusionGroups(Block* block) {
    std::vector<Node*> fusion_groups;
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        guardFusionGroups(b);
      }
      if (n->kind() == prim::TensorExprGroup) {
        fusion_groups.push_back(n);
      }
    }
    for (Node* fusion_group : fusion_groups) {
      guardFusionGroup(fusion_group);
    }
  }

  // A typeinfo map hold profile information (which currently is a type
  // information for the values in the graph). In the beginning the values
  // represented in the map are the values from the original graph for which we
  // had prim::profile nodes. As we start constructing fusion groups, we start
  // adding values from subgraphs of these groups into the map as well.
  std::unordered_map<Value*, TensorTypePtr> typeinfo_map_;

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  // Minimal size of a fusion group
  size_t min_group_size_;
};

void FuseTensorExprs(std::shared_ptr<Graph>& graph, size_t min_group_size) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // Temporary change for Block code generation.
  if (tensorexpr::getTEGenerateBlockCode()) {
    min_group_size = 1;
  }

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  TensorExprFuser fuser(graph, min_group_size);
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
        prim::TensorExprGroup,
        createTensorExprOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

} // namespace jit
} // namespace torch
