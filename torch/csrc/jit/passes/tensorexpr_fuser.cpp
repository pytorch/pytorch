#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>
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
    case aten::sum:
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

  // TODO: if a value has differently typed uses, temporarrily insert a node
  // specializing the type for each use and later remove, instead of bailing
  bool profiledWithDifferentTypes(Value* v) {
    std::vector<TypePtr> types;
    for (const auto& use : v->uses()) {
      if (use.user->kind() == prim::profile) {
        types.push_back(use.user->ty(attr::profiled_type));
      }
    }
    for (size_t i = 1; i < types.size(); ++i) {
      if (types.at(i - 1) != types.at(i)) {
        return true;
      }
    }
    return false;
  }

  void removeProfileNodesAndSpecializeTypes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      if (it->kind() == prim::profile) {
        GRAPH_DEBUG("Removing prim::profile: %", it->output()->debugName());
        it->output()->replaceAllUsesWith(it->input());
        if (!profiledWithDifferentTypes(it->input())) {
          it->input()->setType(it->ty(attr::profiled_type));
        } else {
          GRAPH_DEBUG(
              "Ignoring value with differently typed profiles :%",
              it->output()->debugName());
        }
        it.destroyCurrent();
      } else {
        for (Block* ib : it->blocks()) {
          removeProfileNodesAndSpecializeTypes(ib);
        }
      }
    }
  }

  void removeTensorTypeSpecialization(Value* v) {
    if (!v->type()->cast<TensorType>()) {
      return;
    }
    // Constants & TensorExprGroup will always produce specialized tensor type,
    // TypeCheck are inserted by this pass and only used by fusion groups that
    // insert proper guards
    if (v->node()->kind() == prim::Constant ||
        v->node()->kind() == prim::TypeCheck ||
        v->node()->kind() == prim::TensorExprGroup) {
      return;
    }
    v->setType(TensorType::get());
  }

  void removeTensorTypeSpecializations(Block* block) {
    for (Value* v : block->inputs()) {
      removeTensorTypeSpecialization(v);
    }
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        removeTensorTypeSpecializations(b);
      }
      for (Value* v : n->outputs()) {
        removeTensorTypeSpecialization(v);
      }
    }
  }

  void run() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    RemoveRedundantProfiles(graph_);
    GRAPH_DUMP("After removing redundant profile nodes: ", graph_);
    removeProfileNodesAndSpecializeTypes(graph_->block());
    GRAPH_DUMP(
        "After removing profiling nodes and specializing types: ", graph_);
    RemoveTensorMutation(graph_);
    createFusionGroupsNew(graph_->block());
    GRAPH_DUMP("After creating fusion groups: ", graph_);
    guardFusionGroups(graph_->block());
    GRAPH_DUMP("After guarding fusion groups: ", graph_);
    removeTensorTypeSpecializations(graph_->block());
    GRAPH_DUMP("After removing tensor type specializations: ", graph_);
  }

 private:
  static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    GRAPH_DEBUG(msg, *n);
    if (n->kind() == prim::TensorExprGroup) {
      GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
  }

  void fuseNodes(
      Block* block,
      std::vector<Node*> fusion_group,
      std::vector<Value*> inputs,
      std::vector<Value*> outputs) {
    auto graph = block->owningGraph();
    auto subgraph_node = graph->create(prim::TensorExprGroup, 0);
    subgraph_node->g_(
        attr::Subgraph, std::make_shared<Graph>(graph->current_scope()));
    std::shared_ptr<Graph> subgraph = subgraph_node->g(attr::Subgraph);

    // map node input to subgraph input
    std::map<Value*, Value*> value_map;
    std::vector<Value*> copied_in;
    for (const auto& v : inputs) {
      auto value = toIValue(v);
      if (value && !value->isTensor()) {
        value_map[v] = subgraph->insertConstant(*value);
        copied_in.emplace_back(v);
      } else {
        value_map[v] = subgraph->addInput();
      }
      value_map.at(v)->setType(v->type());
    }
    for (const auto& v : copied_in) {
      auto it = std::find(inputs.begin(), inputs.end(), v);
      TORCH_INTERNAL_ASSERT(it != inputs.end());
      inputs.erase(it);
    }

    // Over this loop the "value_map" is updated
    for (const auto& n : fusion_group) {
      auto new_n =
          subgraph->createClone(n, [&](Value* v) { return value_map.at(v); });
      // Add the newly created subgraph versions of the outputs to the map
      TORCH_INTERNAL_ASSERT(new_n->outputs().size() == n->outputs().size());
      for (auto i = 0; i < new_n->outputs().size(); ++i) {
        value_map[n->outputs().at(i)] = new_n->outputs()[i];
      }
      subgraph->appendNode(new_n);
    }

    std::map<Value*, size_t> index_map;
    for (const auto& v : outputs) {
      index_map[v] = subgraph->registerOutput(value_map.at(v));
      auto output = subgraph_node->addOutput();
      output->setType(v->type());
    }

    Node* insert_point = inputs.front()->node();
    for (const auto& input : inputs) {
      if (input->node()->isAfter(insert_point)) {
        insert_point = input->node();
      }
    }

    Node* after_point = outputs.front()->node();
    for (const auto& output : outputs) {
      if (output->node()->isBefore(after_point)) {
        after_point = output->node();
      }
    }

    // TODO This is not true in the general case!!
    // CANNOT BE LANDED WITH THIS HERE
    subgraph_node->insertBefore(after_point);

    // Set inputs to subgraph node
    for (const auto& v : inputs) {
      subgraph_node->addInput(v);
    }

    // Hook up outputs of subgraph node
    for (const auto& n : block->nodes()) {
      for (const auto& v : n->inputs()) {
        if (!index_map.count(v)) {
          continue;
        }
        auto new_v = subgraph_node->outputs().at(index_map.at(v));
        n->replaceInputWith(v, new_v);
      }
    }
    size_t index = 0;
    std::vector<std::pair<size_t, Value*>> outputReplacement;
    for (const auto& v : block->outputs()) {
      if (index_map.count(v)) {
        auto new_v = subgraph_node->outputs().at(index_map.at(v));
        outputReplacement.emplace_back(std::make_pair(index, new_v));
      }
      index++;
    }
    for (auto kv : outputReplacement) {
      block->replaceOutput(kv.first, kv.second);
    }

    // we have to destroy in reverse order
    for (auto it = fusion_group.rbegin(); it != fusion_group.rend(); ++it) {
      auto n = *it;
      n->destroy();
    }
  }

  std::tuple<std::vector<Node*>, std::vector<Value*>, std::vector<Value*>>
  findCandidates(Block* block) {
    std::vector<Node*> fusion_group;
    std::set<Node*> fusion_group_set;
    // In the fusion group
    std::set<Value*> fused;
    // Not in the fusion group, but can be an input to it
    std::set<Value*> fusion_inputs;
    // Cannot be an input to the fusion group
    std::set<Value*> fusion_outputs;

    // First, let's prep the inputs:
    for (const auto& v : block->inputs()) {
      fusion_inputs.insert(v);
    }

    for (const auto& n : block->nodes()) {
      // check if we can fuse.
      // 1) inputs are either fusion_inputs or fused
      // 2) canHandle == true (includes shape/type checks)

      // 1)
      bool can_fuse = true;
      for (const auto& v : n->inputs()) {
        if (fusion_outputs.count(v)) {
          can_fuse = false;
          break;
        }
      }
      if (!can_fuse) {
        for (const auto& v : n->outputs()) {
          fusion_outputs.insert(v);
        }
        continue;
      }

      // 2)
      if (!canHandle(n)) {
        bool is_output = false;
        for (const auto& v : n->inputs()) {
          if (fused.count(v)) {
            is_output = true;
          }
        }
        for (const auto& v : n->outputs()) {
          if (is_output) {
            fusion_outputs.insert(v);
          } else {
            fusion_inputs.insert(v);
          }
        }
        continue;
      }

      // We can handle it!
      for (const auto& v : n->outputs()) {
        fused.insert(v);
      }
      fusion_group.emplace_back(n);
      fusion_group_set.insert(n);
    }

    std::vector<Value*> inputs;
    for (const auto& n : fusion_group) {
      for (const auto& v : n->inputs()) {
        if (fused.count(v)) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(fusion_inputs.count(v));
        inputs.emplace_back(v);
      }
    }
    std::vector<Value*> outputs;
    for (const auto& n : block->nodes()) {
      if (fusion_group_set.count(n)) {
        continue;
      }
      for (const auto& v : n->inputs()) {
        if (fused.count(v)) {
          outputs.emplace_back(v);
        }
      }
    }
    for (const auto& v : block->outputs()) {
      if (fused.count(v)) {
        outputs.emplace_back(v);
      }
    }

    return std::make_tuple(fusion_group, inputs, outputs);
  }

  void createFusionGroupsNew(Block* block) {
    std::vector<Node*> fusion_group;
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::tie(fusion_group, inputs, outputs) = findCandidates(block);
    GRAPH_DEBUG(
        "Found ",
        fusion_group.size(),
        " candidate nodes for fusion in the original graph:\n");
    for (const auto& n : fusion_group) {
      GRAPH_DEBUG(" ", *n);
    }

    if (fusion_group.size() < min_group_size_) {
      return;
    }

    fuseNodes(block, fusion_group, inputs, outputs);

    graph_->lint();
  }

  bool allShapesAreKnown(Node* node) {
    // TODO: Relax the checks to support dynamic shapes
    for (Value* input : node->inputs()) {
      if (input->type()->cast<TensorType>() && !input->isCompleteTensor()) {
        return false;
      }
    }
    return true;
  }

  bool canHandle(Node* node) {
    if (node->kind() == prim::Constant) {
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

  void guardFusionGroup(Node* fusion_group) {
    GRAPH_DEBUG("Inserting a typecheck guard for a node", *fusion_group);
    auto subgraph = SubgraphUtils::getSubgraph(fusion_group);

    // Fixup types of the subgraph inputs
    std::vector<Value*> inputs_to_check;
    for (Value* input : fusion_group->inputs()) {
      // We only check inputs of the fusion group and expect NNC to infer
      // intermediates and outputs shapes
      if (!input->type()->cast<TensorType>()) {
        continue;
      }

      // fusion outputs are already guarded
      if (input->node()->kind() == prim::Constant ||
          input->node()->kind() == prim::FusionGroup) {
        continue;
      }
      inputs_to_check.push_back(input);
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

    std::unordered_map<Value*, Value*> typechecked_inputs;
    for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
      typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
    }

    // Fixup types of the typecheck node outputs, which are used by the op in
    // execution
    typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
    for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
      typecheck_node->output(i)->setType(typecheck_node->input(i)->type());
    }

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
