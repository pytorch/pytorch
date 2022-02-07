#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <ATen/core/interned_strings.h>
#include <ATen/core/symbol.h>
#include <ATen/record_function.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/jit_opt_limit.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/utils/memory.h>

// NOLINTNEXTLINE
C10_DEFINE_bool(
    torch_jit_disable_cat,
    false,
    "disable aten::cat in TE fusion groups");

C10_DEFINE_bool(
    torch_jit_enable_dynamic_shape_fusion,
    false,
    "enable TE fusion using dynamic shapes");

namespace torch {
namespace jit {

static bool texpr_reductions_enabled = false;

bool isSupportedForBlock(Node* node) {
  switch (node->kind()) {
    case aten::add:
    case aten::mul:
      return true;
    default:
      return false;
  }
}

bool usedOnlyInSize(Value* v) {
  const auto& uses = v->uses();
  return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
    return u.user->matches("aten::size(Tensor self) -> int[]");
  });
}

Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db) {
  AT_ASSERT(!sizes.empty());
  Graph* graph = sizes[0]->owningGraph();
  Node* broadcast_n =
      graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
  broadcast_n->output()->setType(ListType::ofInts());
  db->createValue(broadcast_n->output());
  return broadcast_n->output();
}

namespace tensorexpr {

static const OperatorSet& supported_non_eltwise_set() {
  // clang-format off
  static const OperatorSet supported_non_eltwise_set{
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      "aten::matmul(Tensor self, Tensor other) -> Tensor",
  };
  // clang-format on
  return supported_non_eltwise_set;
};

bool isSupported(Node* node) {
  // For Block codegen we allow limited ops.
  if (tensorexpr::getTEGenerateBlockCode()) {
    return isSupportedForBlock(node);
  }

  static const OperatorSet supported_reduction_set{
      "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
      "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
      "aten::softmax.int(Tensor self, int dim , ScalarType? dtype=None) -> Tensor",
      "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  };
  static const OperatorSet supported_misc_set{
      "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
      "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
  };
  // clang-format on

  if (get_tensorexpr_elementwise_set().contains(node) ||
      node->isMemberOf(supported_non_eltwise_set()) ||
      node->isMemberOf(supported_misc_set) ||
      (texpr_reductions_enabled && node->isMemberOf(supported_reduction_set))) {
    // We only insert guards on Tensor types, so we rely on the output
    // of a node being uniquely determined by its input types.
    // bail if any non-Tensor input affects the output type
    // and cannot be reasoned about statically

    // Value is either an int or a float (can occur from .item())
    for (Value* v : node->inputs()) {
      if (v->type()->cast<NumberType>()) {
        return false;
      }
    }

    // non-const dtype / device
    for (auto arg_name : {"dtype", "device"}) {
      if (auto index = node->schema().argumentIndexWithName(arg_name)) {
        if (!toIValue(node->input(*index))) {
          return false;
        }
      }
    }

    if (FLAGS_torch_jit_disable_cat && node->kind() == aten::cat) {
      return false;
    }

    return true;
  }

  // unschematized ops
  switch (node->kind()) {
    case prim::ConstantChunk:
    case prim::ListConstruct:
    case prim::TensorExprGroup:
      return true;
  }

  return false;
}

} // namespace tensorexpr

static bool texpr_fuser_enabled_ = true;

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

bool tensorExprDynamicShapeFusionEnabled() {
  return FLAGS_torch_jit_enable_dynamic_shape_fusion;
}

void setTensorExprDynamicShapeFusionEnabled(bool val) {
  FLAGS_torch_jit_enable_dynamic_shape_fusion = val;
}

bool setTexprReductionsEnabled(bool value) {
  bool old_value = texpr_reductions_enabled;
  texpr_reductions_enabled = value;
  return old_value;
}

bool texprReductionsEnabled() {
  return texpr_reductions_enabled;
}

void removeProfileNodesAndSpecializeTypes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      GRAPH_DEBUG("Removing prim::profile: %", it->output()->debugName());
      it->output()->replaceAllUsesWith(it->input());
      auto profiled_type = it->ty(attr::profiled_type)->expect<TensorType>();

      // A value can be profiled with differently typed uses.
      // This can occur from:
      // - having a use which is not executed, so the type will be
      // TensorType::get()
      // - control-flow that depends on tensor type:
      //   if x.size() == 2 op(x) else op(x)
      // - mutation of the value on a field represented in the tensor type
      //   op(x); x.resize_([...]); op(x)

      // The most common case today with num_profiles = 1 is from the first
      // case. Here we can just ignore non-profiled uses, and choose any of the
      // profiled uses. Because we guard all tensor types in the runtime, even
      // if we set a Value to have a profiled type from one use and then execute
      // a use with a different profiled type, we will still be correct.
      // In the future we could consider unifying the types of uses, or adding a
      // type refinement node so uses can have the correct corresponding type.
      if (profiled_type == TensorType::get()) {
        continue;
      }
      // If we encounter non-identical profiled types for the same value, merge
      // them.  This situation can happen if, e.g., loop unrolling duplicates
      // profiled types in a loop body in a manner that isn't logically
      // consistent (see TestTEFuser.test_unrolled_cat).
      auto input_type = it->input()->type()->expect<TensorType>();
      if (input_type == TensorType::get()) {
        it->input()->setType(profiled_type);
      } else {
        it->input()->setType(input_type->merge(*profiled_type));
      }
      it.destroyCurrent();

    } else {
      for (Block* ib : it->blocks()) {
        removeProfileNodesAndSpecializeTypes(ib);
      }
    }
  }
}

void RemoveProfileNodesAndSpecializeTypes(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("Before removeProfileNodesAndSpecializeTypes:\n", *graph);
  removeProfileNodesAndSpecializeTypes(graph->block());
  GRAPH_DEBUG("After removeProfileNodesAndSpecializeTypes:\n", *graph);
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

void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph) {
  removeTensorTypeSpecializations(graph->block());
}

void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    Symbol kind) {
  GRAPH_DEBUG("Inserting a typecheck guard for a node", *guarded_node);
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (Value* input : guarded_node->inputs()) {
    // We only check inputs of the guarded nodes and expect user to infer
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
    guard_types.push_back(type_converter(input->type()->expect<TensorType>()));
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
      guarded_node->owningGraph()
          ->create(kind, inputs_to_check, inputs_to_check.size() + 1)
          ->insertBefore(guarded_node);
  typecheck_node->tys_(attr::types, guard_types);
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
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);
  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // types get copied to the fallback graph, so remove specializations before
  // replacing
  removeTensorTypeSpecializations(false_block);
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  guarded_node->moveBefore(true_block->return_node());
  for (size_t idx = 0; idx < guarded_node->inputs().size(); ++idx) {
    if (typechecked_inputs.count(guarded_node->input(idx))) {
      guarded_node->replaceInput(
          idx, typechecked_inputs.at(guarded_node->input(idx)));
    }
  }
  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }
}

namespace {
bool has_unsupported_pin_memory(const Node* node) {
  // cant support non-constant pin_memory or pin_memory = True
  if (auto maybe_index = node->schema().argumentIndexWithName("pin_memory")) {
    int index = *maybe_index;
    auto inp = node->input(index);
    if (inp->type() != NoneType::get() &&
        constant_as<bool>(inp).value_or(true)) {
      return true;
    }
  }
  return false;
}
} // namespace

class TensorExprFuser {
 public:
  TensorExprFuser(
      std::shared_ptr<Graph> graph,
      size_t min_group_size,
      bool add_composed_op,
      bool fuse_to_dynamic_shapes)
      : graph_(std::move(graph)),
        min_group_size_(min_group_size),
        add_composed_op_(add_composed_op),
        fuse_to_dynamic_shapes_(fuse_to_dynamic_shapes) {
    parseTENotFuseOption();
  }

  // Builds up expressions that compute shapes of all intermediates (and
  // outputs) of the fusion group, based on the sizes of inputs. You should run
  // DCE to remove those that you end up not using.
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    GRAPH_DUMP("buildShapeExpressions for ", fusion_group->g(attr::Subgraph));
    WithInsertPoint insert_guard{fusion_group->next()};
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    for (const auto i : c10::irange(inputs.size())) {
      if (inputs[i]->type()->isSubtypeOf(*TensorType::get())) {
        Value* soutput = graph->insert(aten::size, {inputs[i]});
        aliasDb_->createValue(soutput);
        GRAPH_DEBUG(
            "Adding a mapping for %",
            sinputs[i]->debugName(),
            " ",
            getHeader(soutput->node()));
        shape_of[sinputs[i]] = soutput;
      }
    }

    // When we have a guarantee that an output won't be removed, because it's
    // used in expressions that don't involve size checks, we can use its size
    // instead of computing a long chain of broadcasts, starting from the
    // beginning of the kernel.
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      if (usedOnlyInSize(outputs[i]))
        continue;
      Value* soutput = graph->insert(aten::size, {outputs[i]});
      aliasDb_->createValue(soutput);
      shape_of[soutputs[i]] = soutput;
    }

    for (Node* n : subgraph->nodes()) {
      if (n->kind() == prim::ConstantChunk) {
        Node* sizes_node = graph->insertNode(
            graph->create(prim::ChunkSizes, shape_of.at(n->input()), 2));
        sizes_node->i_(attr::dim, n->i(attr::dim));
        sizes_node->i_(attr::chunks, n->i(attr::chunks));
        for (Value* output : sizes_node->outputs()) {
          aliasDb_->createValue(output);
        }
        Value* regular_size = sizes_node->outputs().at(0);
        Value* last_size = sizes_node->outputs().at(1);
        regular_size->setType(ListType::ofInts());
        last_size->setType(ListType::ofInts());
        auto outputs = n->outputs();
        for (Value* o : outputs.slice(0, outputs.size() - 1)) {
          shape_of.emplace(o, regular_size);
        }
        shape_of.emplace(outputs.at(outputs.size() - 1), last_size);
        continue;
      }

      // we only support shape calculations for elementwise, some
      // non-elementwise like batch_norm, conv, matmul, and
      // a few exceptions (e.g. prim::ConstantChunk, etc) listed above
      if (!(get_tensorexpr_elementwise_set().contains(n)) &&
          !n->isMemberOf(tensorexpr::supported_non_eltwise_set())) {
        continue;
      }

      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        return v->type()->isSubtypeOf(*TensorType::get());
      });
      GRAPH_DEBUG("Building sizes for ", getHeader(n));
      bool all_inputs_have_sizes = true;
      auto shapes = fmap(tensor_inputs, [&](Value* v) {
        GRAPH_DEBUG("Getting aten::size for %", v->debugName());
        all_inputs_have_sizes &= shape_of.count(v);
        return shape_of.count(v) != 0 ? shape_of.at(v) : nullptr;
      });

      if (!all_inputs_have_sizes) {
        GRAPH_DEBUG(
            "Not all tensor arguments have sizes available to compute the broadcasted size",
            getHeader(n));
        continue;
      }
      shape_of.emplace(
          n->output(),
          shapes.size() == 1 ? shapes[0]
                             : broadcastSizes(shapes, aliasDb_.get()));
    }
    return shape_of;
  }

  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    if (fusion_group->kind() != prim::TensorExprGroup)
      return;
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto shape_of = buildShapeExpressions(fusion_group);
    auto outputs = fusion_group->outputs().vec();
    auto soutputs = subgraph->outputs().vec();
    // XXX: Iterating in this order is not only good for performance reasons!
    // It is also crucial for correctness (i has to reflect the current true
    // index of outputs[i])!
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      auto output = outputs[i];
      auto soutput = soutputs[i];
      if (usedOnlyInSize(output) && shape_of.count(soutput) > 0) {
        auto uses = output->uses();
        for (Use u : uses) {
          AT_ASSERT(u.user->matches("aten::size(Tensor self) -> int[]"));
          u.user->output()->replaceAllUsesWith(shape_of.at(soutput));
          u.user->destroy();
        }
        fusion_group->eraseOutput(i);
        subgraph->eraseOutput(i);
      }
    }
  }

  void run() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    RemoveRedundantProfiles(graph_);
    GRAPH_DUMP("After removing redundant profile nodes: ", graph_);
    createFusionGroups(graph_->block());
    GRAPH_DUMP("After creating fusion groups: ", graph_);
    // we maintain alias db correctness during initial fusion, but it is
    // difficult to maintain correctness after inlining so inline only after
    // fusion is done.
    inlineSmallFusionGroups(graph_->block());
    GRAPH_DUMP("After inlining small fusion groups: ", graph_);
    if (fuse_to_dynamic_shapes_) {
      VLOG(1) << "TensorExpr fusion with dynamic shapes is enabled"
              << std::endl;
      generalizeFusionGroups(graph_->block());
      GRAPH_DUMP("After generalizing fusion groups: ", graph_);
    } else {
      prepareFusionGroupAndGuardOutputs(graph_->block());
      GRAPH_DUMP("After guarding fusion groups: ", graph_);
      removeTensorTypeSpecializations(graph_->block());
      GRAPH_DUMP("After removing tensor type specializations: ", graph_);
    }
  }

 private:
  Node* getOrCreateTensorExprSubgraph(Node* n) {
    if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::TensorExprGroup) {
      return n;
    }
    GRAPH_UPDATE("Creating a tensorexpr::Group node from: ", *n);
    return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, prim::TensorExprGroup, *aliasDb_);
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
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

  // Create a fusion group starting from the node N.
  // We then try to pull inputs into the fusion group and repeat that process
  // until there is nothing we can pull in.
  std::pair<graph_node_list::iterator, bool> createFusionGroup(
      Node* fusion_node) {
    // Allow single-node groups containing conv2d, since we'll only select
    // those in cases where the tensorexpr implementation is faster than the
    // aten implementation.
    if (min_group_size_ == 1 || fusion_node->kind() == aten::conv2d) {
      fusion_node = getOrCreateTensorExprSubgraph(fusion_node);
    }

    GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
    auto inputs = sortReverseTopological(
        fusion_node->inputs(), fusion_node->owningBlock());
    for (auto input : inputs) {
      debugDumpFusionGroup("Current fusion group: ", fusion_node);
      GRAPH_DEBUG("Trying to merge: ", *input->node());
      if (auto maybe_fusion_group = tryMerge(fusion_node, input->node())) {
        // we successfully merged, so the new group's `inputs` may have
        // changed. So rescan the new group for more merging opportunities.
        return std::make_pair(
            maybe_fusion_group.value()->reverseIterator(), true);
      }
    }

    return std::make_pair(++fusion_node->reverseIterator(), false);
  }

  static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
    GRAPH_DEBUG(msg, *n);
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (n->kind() == prim::TensorExprGroup) {
      GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
  }

  // No Ops in eager shouldn't be outputs of Fusion Groups because it
  // will degrade perf and change aliasing relationships
  static bool unexecutedEagerOp(Node* n) {
    if (n->kind() != aten::to) {
      return false;
    }

    return *n->input(0)->type()->expect<TensorType>() ==
        *n->output()->type()->expect<TensorType>();
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* n) {
    GRAPH_DEBUG("Considering node:", *n)

    if (!canHandle(n)) {
      return std::make_pair(++n->reverseIterator(), false);
    }
    // There are some nodes that we can support, but we don't want to start a
    // fusion group from - skip them.
    if (n->kind() == prim::ListConstruct || n->kind() == aten::slice ||
        n->kind() == aten::unsqueeze || n->kind() == prim::ConstantChunk ||
        n->kind() == prim::Constant || unexecutedEagerOp(n)) {
      return std::make_pair(++n->reverseIterator(), false);
    }
    return createFusionGroup(n);
  }

  // Merge fusible nodes into subgraphs in prim::TensorExprGroup nodes.
  void createFusionGroups(Block* block) {
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        createFusionGroups(b);
      }
    }

    // Try to merge adjacent fusion groups together. Because we have only merged
    // by looking at graph inputs, without this we would not attempt to merge
    // adjacent fusion groups that don't have a depdency on each other

    std::vector<Node*> initial_fusion_groups;
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::TensorExprGroup) {
        initial_fusion_groups.push_back(n);
      }
    }

    Node* prev_fusion_group =
        initial_fusion_groups.size() ? initial_fusion_groups[0] : nullptr;

    for (const auto i : c10::irange(1, initial_fusion_groups.size())) {
      // Try merging the just created fusion group into the previous one.
      // If it did not work, then put the previous fusion group into
      // fusion_groups vector - we will not touch it anymore in this loop.
      // If merging suceeded, save the merged group as the "previous" fusion
      // group so that we can try to merge the next one into it.

      Node* fusion_group = initial_fusion_groups[i];
      debugDumpFusionGroup(
          "Trying to merge into the previous fusion group: ",
          prev_fusion_group);
      if (auto merged_fusion_group =
              tryMerge(prev_fusion_group, fusion_group)) {
        prev_fusion_group = *merged_fusion_group;
        debugDumpFusionGroup(
            "Successfully merged into the previous fusion group: ",
            prev_fusion_group);
      } else {
        GRAPH_DEBUG("Cannot merge into the previous fusion group");
        prev_fusion_group = fusion_group;
      }
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

  bool hasConv(Block* block) {
    for (Node* n : block->nodes()) {
      if (n->kind() == aten::conv2d) {
        return true;
      }
    }
    return false;
  }

  bool inlineIfTooSmall(Node* n) {
    if (n->kind() != prim::TensorExprGroup) {
      return false;
    }
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_nodes = blockSize(subgraph->block());
    // Allow small subgraphs containing conv2d, since we'll only select those
    // in cases where the tensorexpr implementation is faster than the aten
    // implementation.
    if (num_nodes < min_group_size_ && !hasConv(subgraph->block())) {
      GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    // Cleanup the subgraph from duplicated constants while we're at it.
    ConstantPooling(subgraph);
    return false;
  }

  void inlineSmallFusionGroups(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++;

      for (Block* b : n->blocks()) {
        inlineSmallFusionGroups(b);
      }
      inlineIfTooSmall(n);
    }
  }

  c10::optional<Node*> tryMerge(Node* fusion_group, Node* to_merge) {
    if (!canMerge(fusion_group, to_merge)) {
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
      if (!aliasDb_->moveBeforeTopologicallyValid(n, move_point)) {
        GRAPH_UPDATE("Failed to move because of AliasDB checks!");
        return c10::nullopt;
      }
      move_point = n;
    }

    // Now all the nodes that we're going to fuse are moved next to the fusion
    // group, so we can safely merge them into the fusion group subgraph.
    fusion_group = getOrCreateTensorExprSubgraph(fusion_group);

    for (auto n : nodes_to_merge) {
      GRAPH_UPDATE("Merging ", getHeader(n));
      SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
          n, fusion_group, *aliasDb_);
    }
    return fusion_group;
  }

  bool shapeIsKnown(Value* v) {
    if (v->type()->cast<TensorType>()) {
      if (!v->isCompleteTensor()) {
        return false;
      }
    }
    return true;
  }

  bool allShapesAreKnown(Node* node) {
    // TODO: Relax the checks to support dynamic shapes
    for (Value* input : node->inputs()) {
      if (!shapeIsKnown(input)) {
        return false;
      }
      if (input->node()->kind() == prim::ListConstruct) {
        if (!allShapesAreKnown(input->node())) {
          return false;
        }
      }
    }
    for (Value* output : node->outputs()) {
      if (!shapeIsKnown(output)) {
        return false;
      }
    }
    return true;
  }

  bool canFuseOnDevice(Value* v) {
    auto type = v->type()->cast<TensorType>();
    if (!type) {
      return true;
    }
    auto device = type->device();
    if (!device) {
      return false;
    }
    if (device->is_cpu()) {
      return canFuseOnCPU();
    } else if (device->is_cuda()) {
      return canFuseOnGPU();
    } else if (device->is_xpu()) {
      return false;
    } else {
      TORCH_CHECK_NOT_IMPLEMENTED(false, "Unknown device for tensorexpr fuser")
    }
  }

  bool isFusableOnDevice(Node* node) {
    for (const auto& input : node->inputs()) {
      if (input->node()->kind() == prim::ListConstruct) {
        if (!isFusableOnDevice(input->node())) {
          return false;
        }
      }
      if (!canFuseOnDevice(input)) {
        return false;
      }
    }
    return true;
  }

  bool typesAreSupported(Node* node) {
    // clang-format off
    // breaks up the schema strings so they are no longer discoverable with ctrl-F
    static const OperatorSet float_only_operator_set{
      "aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor",
    };
    static const OperatorSet int_only_operator_set{
      "aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor",
    };
    static const OperatorSet cpu_compute_heavy_set{
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      "aten::matmul(Tensor self, Tensor other) -> Tensor",
    };
    static const OperatorSet gpu_only_operator_set{
      // On CPU, these are slower and less accurate than ATen kernels, because
      // ATen is able to use MKL-VML, whereas the fuser currently can't.  The
      // fuser uses sleef instead because sleef provides functions that operate
      // on vectors, instead of large buffers.
      "aten::erf(Tensor self) -> Tensor",
      "aten::erfc(Tensor self) -> Tensor",
    };
    static const OperatorSet pow{
      "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
    };
    // clang-format on

    // Check types of input values.
    for (const Value* v : node->inputs()) {
      if (auto const& tt = v->type()->cast<TensorType>()) {
        auto const& st = tt->scalarType();
        auto const& device = tt->device();

        // All tensors must be typed.
        if (!st || !device) {
          return false;
        }

        // Byte tensors introduce too many corner cases in type promotion.
        // Better not to try to handle them.
        if (*st == c10::ScalarType::Byte) {
          return false;
        }

        // Float16 support has some issues (see e.g. #61336 and #61382), so for
        // now it's disabled. There seem to be some problems in HalfRewriter,
        // but on top of that Float16 has a few kinks on LLVM.  Thus, on CPU we
        // additionally disable it until we either move to a more stable version
        // or find workarounds.
        if ((*st == c10::ScalarType::Half ||
             *st == c10::ScalarType::BFloat16) &&
            *device == c10::kCPU) {
          return false;
        }

        // These operators only support floats, because integer divisors need to
        // raise ZeroDivisionError.
        if (node->isMemberOf(float_only_operator_set) && !isFloatingType(*st)) {
          return false;
        }

        // These operators have complicated casting rules for floats.
        if (node->isMemberOf(int_only_operator_set) && isFloatingType(*st)) {
          return false;
        }
      } else if (node->isMemberOf(float_only_operator_set)) {
        // Check scalar operands of float-only ops.
        if (!v->type()->cast<FloatType>()) {
          return false;
        }
      } else if (node->isMemberOf(int_only_operator_set)) {
        if (!v->type()->cast<IntType>()) {
          return false;
        }
      }
    }

    // aten::pow has special rules to avoid complicated integer cases.  We
    // expect the first arg to be a floating point tensor, and if that's the
    // case the type of the scalar exponent doesn't matter.
    if (node->isMemberOf(pow)) {
      auto const& tt = node->input(0)->type()->cast<TensorType>();
      if (!tt) {
        return false;
      }
      auto const& st = tt->scalarType();
      if (!st || !isFloatingType(*st)) {
        return false;
      }
    }

    // Operator is only supported on CPU.
    if (node->isMemberOf(cpu_compute_heavy_set)) {
      if (fuse_to_dynamic_shapes_) {
        return false;
      }

      auto device = tensorexpr::pickDeviceType(node->inputs());
      if (!device) {
        device = tensorexpr::pickDeviceType(node->outputs());
      }
      if (!device || !device->is_cpu()) {
        return false;
      }
    }

    // Operator is only supported on GPU.
    if (node->isMemberOf(gpu_only_operator_set)) {
      auto device = tensorexpr::pickDeviceType(node->inputs());
      if (!device) {
        device = tensorexpr::pickDeviceType(node->outputs());
      }
      if (!device || !device->is_cuda()) {
        return false;
      }
    }

    if (node->kind() == aten::to) {
      // only support same-device conversion
      auto device = tensorexpr::pickDeviceType(node->inputs());
      auto output_device = tensorexpr::pickDeviceType(node->outputs());
      if (!device || !output_device || *device != *output_device) {
        return false;
      }
      // non_blocking only applies in cross-device conversion, which we bail on
      // copy arg only applies if op is a no-op, which we dont start fusion
      // group from memory format is separately handled in NNC output

      // all non-Tensor arguments must be constant
      for (size_t i = 1; i < node->inputs().size(); i++) {
        if (node->inputs().at(i)->node()->kind() != prim::Constant) {
          return false;
        }
      }

      if (has_unsupported_pin_memory(node)) {
        return false;
      }
    }

    if (node->kind() == aten::_autocast_to_reduced_precision ||
        node->kind() == aten::_autocast_to_full_precision) {
      for (auto i : c10::irange(1, node->inputs().size())) {
        if (node->inputs().at(i)->node()->kind() != prim::Constant) {
          return false;
        }
      }

      if (has_unsupported_pin_memory(node)) {
        return false;
      }
    }

    if (node->kind() == aten::unsqueeze) {
      // `dim` argument must be a constant.
      if (node->input(1)->node()->kind() != prim::Constant) {
        return false;
      }
    }

    if (node->kind() == aten::conv2d) {
      if (!tensorexpr::conv2dIsSupportedJit(node)) {
        GRAPH_DEBUG("Params of conv2d are not supported");
        return false;
      }
    }
    if (node->kind() == aten::matmul) {
      if (!tensorexpr::matmulIsSupported(node)) {
        GRAPH_DEBUG("Shapes of matmul inputs are not supported");
        return false;
      }
    }
    return true;
  }

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

  bool canHandle(Node* node) {
    REQ(allShapesAreKnown(node));
    REQ(isFusableOnDevice(node));
    REQ(operators_not_to_fuse.find(node->kind()) ==
        operators_not_to_fuse.end());

    for (Value* input : node->inputs()) {
      if (auto const& tt = input->type()->cast<TensorType>()) {
        auto st = tt->scalarType();
        if (!st) {
          // All tensor types should be known.
          return false;
        }
        if (c10::isComplexType(*st) || c10::isQIntType(*st)) {
          return false;
        }
      }
    }
    if (node->kind() == aten::cat) {
      REQ(node->input(0)->node()->kind() == prim::ListConstruct);
      REQ(node->input(0)->uses().size() == 1);
      REQ(node->input(1)->node()->kind() == prim::Constant);
      auto const& listconstruct = node->input(0)->node();
      REQ(tensorexpr::pickDeviceType(listconstruct->inputs()));
    } else {
      REQ(tensorexpr::pickDeviceType(node->inputs()));
    }

    // Only fuse aten::batch_norm when the parameter 'training' is false
    if (node->kind() == aten::batch_norm) {
      REQ(node->input(5)->node()->kind() == prim::Constant);
      REQ(!toIValue(node->input(5)).value().toBool());
    }

    REQ(tensorexpr::isSupported(node));
    REQ(typesAreSupported(node));

    // A hook to optimizations limitter to allow bisecting the pass
    REQ(JIT_OPT_ALLOWED);

    if (fuse_to_dynamic_shapes_) {
      // Allow only if the node has a shape function defined.
      // ListConstruct node is an exception since that is needed to fuse
      // aten::cat, though it does not have a shape function.
      REQ(node->kind() == prim::ListConstruct ||
          node->kind() == prim::TensorExprGroup ||
          (node->maybeSchema() && shapeComputeGraphForSchema(node->schema())));
    }

    return true;
  }

  bool canMerge(Node* consumer, Node* producer) {
    // Only fuse within a block
    REQ(consumer->owningBlock() == producer->owningBlock());

    // Symbolic checks
    REQ(canHandle(producer) || producer->kind() == prim::TensorExprGroup);
    TORCH_INTERNAL_ASSERT(
        consumer->kind() == prim::TensorExprGroup || canHandle(consumer));

    // nvrtc has a limit on the number of arguments allowed in a CUDA kernel.
    // The specific limit is a function of constant memory size, amount
    // available to pass arguments, and some implementation dependence. Select a
    // safe limit here.
    constexpr size_t subgraphArgLimit = 128;
    auto const nInputs = consumer->inputs().size() +
        consumer->outputs().size() + producer->inputs().size() +
        producer->outputs().size();
    REQ(nInputs <= subgraphArgLimit);

    // Device checks
    if (consumer->kind() != aten::cat && producer->kind() != aten::cat) {
      // aten::cat needs a special handling because it takes a Tensor[] as its
      // input We deal with that in the code below.
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(consumer_device);
      auto producer_device = tensorexpr::pickDeviceType(producer->inputs());
      REQ(producer_device);
      REQ(*consumer_device == *producer_device);
    }

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
      auto const& listConstruct = producer->input(0)->node();
      // We're merging listconstruct->cat->consumer. cat is the producer here
      // and we cannot determine its device type - we should use device of the
      // listconstruct instead
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(listconstruct_device);
      REQ(consumer_device);
      REQ(*listconstruct_device == *consumer_device);
      for (auto const& input : listConstruct->inputs()) {
        REQ(isFusableOnDevice(input->node()));
      }
      REQ((nInputs + listConstruct->inputs().size()) <= subgraphArgLimit);
    } else if (consumer->kind() == aten::cat) {
      REQ(consumer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(consumer->input(0)->uses().size() == 1);
      REQ(consumer->input(1)->node()->kind() == prim::Constant);
      auto const& listConstruct = consumer->input(0)->node();
      // We're merging listconstruct->cat. cat is the consumer and listconstruct
      // is the producer. cat doesn't have its device type and thus the only
      // thing we should check is that listconstruct's device is well defined
      // (e.g. all its inputs has the same device).
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      REQ(listconstruct_device);
      REQ((nInputs + listConstruct->inputs().size()) <= subgraphArgLimit);
    } else {
      REQ(isFusableOnDevice(producer));
    }

    return true;
  }
#undef REQ

  void prepareFusionGroupAndGuardOutputs(Block* block) {
    std::vector<Node*> fusion_groups;
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        prepareFusionGroupAndGuardOutputs(b);
      }
      if (n->kind() == prim::TensorExprGroup) {
        fusion_groups.push_back(n);
      }
    }
    for (Node* fusion_group : fusion_groups) {
      removeOutputsUsedOnlyInSize(fusion_group);
      insertTypeGuard(
          fusion_group,
          [](const TensorTypePtr& t) { return t; },
          prim::TypeCheck);
    }
  }

  void generalizeFusionGroups(Block* block) {
    std::vector<Node*> fusion_groups;
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        generalizeFusionGroups(b);
      }
      if (n->kind() == prim::TensorExprGroup) {
        fusion_groups.push_back(n);
      }
    }
    for (Node* fusion_group : fusion_groups) {
      removeOutputsUsedOnlyInSize(fusion_group);
      VLOG(1) << "GenerateGuard for fusion group: " << *fusion_group;
      if (!GenerateGuard(fusion_group, add_composed_op_)) {
        VLOG(1) << "  Unfusing the fusion group because GenerateGuard failed"
                << std::endl;
        SubgraphUtils::unmergeSubgraph(fusion_group);
      }
    }
  }

  // This function parses the option provided by the environment variable
  // "PYTORCH_TENSOREXPR_DONT_FUSE".
  // This variable allows users to disable fusion on a list of specified
  // operators that are separated by ':'. e.g.,
  // 'PYTORCH_TENSOREXPR_DONT_FUSE="clamp:mul:add"' disables fusion on
  // aten::clamp, aten::mul and aten::add.
  void parseTENotFuseOption() {
    const char* option = std::getenv("PYTORCH_TENSOREXPR_DONT_FUSE");
    std::stringstream in_ss;
    if (option) {
      in_ss << option;
    }

    std::string line;
    while (std::getline(in_ss, line, ':')) {
      if (line.size() == 0) {
        continue;
      }
      operators_not_to_fuse.insert(c10::Symbol::aten(line));
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  std::set<NodeKind> operators_not_to_fuse;
  // Minimal size of a fusion group
  size_t min_group_size_;
  // compose Runtime Type Guard and Kernel in one op
  bool add_composed_op_;
  // generalize static shapes to dynamic shapes
  bool fuse_to_dynamic_shapes_;
};

void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size,
    bool add_composed_op,
    bool fuse_to_dynamic_shapes) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // Temporary change for Block code generation.
  if (tensorexpr::getTEGenerateBlockCode()) {
    min_group_size = 1;
  }

  if (add_composed_op) {
    TORCH_INTERNAL_ASSERT(
        fuse_to_dynamic_shapes, "Fusing static shapes with composed op NYI");
  }

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  TensorExprFuser fuser(
      graph, min_group_size, add_composed_op, fuse_to_dynamic_shapes);
  fuser.run();

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}

Operation createTensorExprOp(const Node* node) {
  bool dynamic_shape_fusion_node =
      node->hasAttribute(attr::striding_inputs_desc);
  if (!dynamic_shape_fusion_node) {
    auto kernel =
        std::make_shared<tensorexpr::TensorExprKernel>(node->g(attr::Subgraph));
    return [kernel](Stack& stack) {
      RECORD_FUNCTION(kernel->getKernelName(), std::vector<c10::IValue>());
      kernel->run(stack);
      return 0;
    };
  }

  // Handle the case when dynamic shape fusion is enabled.
  VLOG(1) << "Compiling a new kernel for " << *node;
  std::vector<int64_t> sym_shapes;
  if (node->hasAttribute(attr::symbolic_shape_inputs)) {
    sym_shapes = node->is(attr::symbolic_shape_inputs);
  }
  bool allow_stack_outputs = false;
  if (node->hasAttribute(attr::allow_stack_outputs)) {
    allow_stack_outputs = node->i(attr::allow_stack_outputs) == 1;
  }

  std::unordered_map<c10::Symbol, tensorexpr::NNCLoweringFunction>
      custom_lowerings;
  auto subgraph = node->g(attr::Subgraph);
  IValue sym_strides = node->ival(attr::striding_inputs_desc);

  // Striding Descriptor is serialized on the node as a vector of vector of
  // strings, translate back to StrideInput enum
  std::vector<std::vector<std::string>> sym_strides_strs =
      sym_strides.to<std::vector<std::vector<std::string>>>();
  std::vector<std::vector<StrideInput>> striding_inputs;
  for (const auto& vec : sym_strides_strs) {
    std::vector<StrideInput> input_desc;
    input_desc.reserve(vec.size());
    for (const std::string& str : vec) {
      input_desc.push_back(strideInputFromString(str));
    }
    striding_inputs.push_back(input_desc);
  }
  std::unordered_map<const Value*, std::vector<StrideInput>> stride_map;
  size_t index = 0;
  for (Value* v : subgraph->inputs()) {
    if (!v->type()->cast<TensorType>()) {
      continue;
    }
    stride_map[v] = striding_inputs[index];
    index++;
  }
  std::vector<std::string> output_desc =
      node->ival(attr::striding_outputs_desc).to<std::vector<std::string>>();
  for (size_t i = 0; i < subgraph->outputs().size(); ++i) {
    stride_map[subgraph->outputs().at(i)] = {
        strideInputFromString(output_desc.at(i))};
  }

  std::shared_ptr<tensorexpr::TensorExprKernel> kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(
          subgraph,
          custom_lowerings,
          sym_shapes,
          /*pre_alloc*/ false,
          stride_map);

  auto num_subgraph_inputs = subgraph->inputs().size();
  return [kernel, num_subgraph_inputs, allow_stack_outputs](Stack& stack) {
    RECORD_FUNCTION(kernel->getKernelName(), std::vector<c10::IValue>());

    // Stack contents:
    //   [<outputs>] <inputs>
    //
    // If the number of graph inputs is same as the stack size, then no
    // outputs are being passed in. Otherwise, output tensors are passed in
    // at the bottom of the stack. So, we call the appropriate run function
    // in TensorExprKernel.
    if (num_subgraph_inputs == stack.size() || !allow_stack_outputs) {
      kernel->run(stack);
    } else {
      kernel->runWithAllocatedOutputs(stack);
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
