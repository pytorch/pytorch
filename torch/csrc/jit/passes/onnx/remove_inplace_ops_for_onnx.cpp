#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>

#include <limits>

namespace torch {
namespace jit {

namespace {

const std::set<c10::Symbol> inplace_ops =
    {aten::append, aten::index_put_, aten::pop, aten::insert, aten::Delete};

struct InplaceConverter {
  InplaceConverter(
      std::shared_ptr<Graph> graph,
      MutationRemover* mr,
      Module* model = nullptr)
      : graph_(graph), mr_(mr), module_(model) {}

  // new api
  void convertMutationForONNX();

  // old api
  void RegisterInplaceOpAsBlockOutputs();

  // old code
  void RegisterInplaceNodeInBlocks(Value* orig_data, Value* new_data);
  bool isInplaceNode(const Node* n) const;

 private:

  struct ValueTracker {
    ValueTracker() : graph_(nullptr), root_block_(nullptr) {}

    void init(const std::shared_ptr<Graph>& graph);
    void registerSetValue(Value* old_v, Value* new_v);
    Value* findAliasForValueAtNode(Value* v, const Node* n) const;
    void passUpdateValueUse(Block* block);
    std::string toString() const;

   private:
    std::vector<std::tuple<Value*, Node*, Block*>> sortAliasOfValue(const Value* v) const;

    std::shared_ptr<Graph> graph_;
    Block* root_block_;

    // Map from aliases to ground truth value.
    // A single value can have multiple aliases throughout the graph,
    // created by inplace operators, and preserved through loop carried input/output.
    std::unordered_map<Value*, Value*> alias_to_value_;

    struct aliasComp {
      bool operator() (const Value* a, const Value* b) {
        auto* n_a = a->node();
        auto* n_b = b->node();
        if (n_a == n_b) {
          return false;
        }
        auto a_b = n_a->isBefore(n_b);
        auto b_a = n_b->isBefore(n_a);
        if (a_b == b_a) {
          return a->unique() < b->unique();
        }
        return a_b;
      }
    };
    // Map from ground truth value to aliases sorted by their order in graph.
    std::unordered_map<Value*, std::set<Value*, aliasComp>> value_to_sorted_aliases_;
  };

  // new code
  void gatherAttrNameInitialValueMap(Block* block,
      std::unordered_map<std::string, Value*>& attr_name_value_map,
      std::unordered_map<Node*, std::string>& attr_node_fullname_map);
  void replaceAttrWithInplaceOps(Block* block,
      const std::unordered_map<std::string, Value*>& attr_name_value_map,
      const std::unordered_map<Node*, std::string>& attr_node_fullname_map);

  void convertInplaceOps();
  void convertInplaceOps(Block* block);

  // old code
  Value* MatchIfBlocksOutputForValue(
      Value* orig_data,
      Block* outer_block,
      Value* origOutput);

  void RegisterInplaceNodeInIfBlocks(
      Value* orig_data,
      Value* new_data,
      const std::string& output_name);

  void RegisterInplaceNodeInLoopBlocks(Value* orig_data, Value* new_data);

  void convertGetSetAttrToInplaceOps(Block* block);
  std::unordered_map<std::string, Value*> registerInplaceOpAsBlockOutputs(Block* block);

  void trackAndRegisterAttributesInBlocks(
      Node* n,
      const Module& module_,
      std::unordered_map<std::string, Value*>& nextSetAttrValues);

  Value* registerSetAttrInBlocks(
      Block* block,
      Node* cloneNode,
      Value* origValue,
      const std::string& output_name);

  void PrintSnapshotForDebug(const Node* n) {
    GRAPH_UPDATE("Print Snapshot for debugging.\nGraph: ", graph_->toString());
    GRAPH_UPDATE("Current node: ", *n);

    auto print_map = [](const std::unordered_map<std::string, Value*>& map) {
      for (auto iter : map) {
        GRAPH_UPDATE("Key: ", iter.first);
        GRAPH_UPDATE("Value: ", iter.second->debugName());
      }
    };

    GRAPH_UPDATE("All attribute values map:");
    print_map(allAttrValues_);

    GRAPH_UPDATE("Set attribute values map:");
    print_map(setAttrValues_);

    GRAPH_UPDATE("All attribute modules map:");
    print_map(allAttrModules_);
  }

  std::shared_ptr<Graph> graph_;
  MutationRemover* mr_;
  Module* module_;
  ValueTracker vt_;
  // A map of names and values of referenced attributes, to avoid duplicates.
  std::unordered_map<std::string, Value*> allAttrValues_ = {};
  // A map of names and values of set attributes, to track mutations.
  std::unordered_map<std::string, Value*> setAttrValues_ = {};
  // A map of names and values of attribute modules, to create SetAttr node.
  std::unordered_map<std::string, Value*> allAttrModules_ = {};
};

bool InplaceConverter::isInplaceNode(const Node* n) const {
  if (inplace_ops.find(n->kind()) != inplace_ops.end()) {
    return true;
  }

  if (n->kind() == Symbol::fromQualString("onnx::Placeholder") &&
      n->s(attr::name) == "index_put_") {
    return true;
  }

  return false;
}

Node* addDummyClone(
    Graph* graph,
    Value* orig_data,
    bool insertBefore,
    Node* referenceNode) {
  Node* newNode = nullptr;
  if (orig_data->type()->kind() == TypeKind::ListType) {
    newNode = graph->create(aten::list, /*num_outputs =*/1);
    newNode->addInput(orig_data);
    newNode->output()->setType(orig_data->type());
    if (insertBefore)
      newNode->insertBefore(referenceNode);
    else
      referenceNode->owningBlock()->prependNode(newNode);
  } else if (
      orig_data->type()->kind() == TypeKind::TensorType ||
      orig_data->type()->kind() == TypeKind::IntType ||
      orig_data->type()->kind() == TypeKind::FloatType ||
      orig_data->type()->kind() == TypeKind::BoolType) {
    auto* noneNode = graph->create(prim::Constant);
    noneNode->output()->setType(NoneType::get());
    newNode = graph->create(aten::clone, /*num_outputs =*/1);
    newNode->addInput(orig_data);
    newNode->addInput(noneNode->output());
    newNode->output()->setType(orig_data->type());
    if (insertBefore)
      newNode->insertBefore(referenceNode);
    else
      referenceNode->owningBlock()->prependNode(newNode);
    noneNode->insertBefore(newNode);
  }
  return newNode;
}

// Check If then/else blocks to match the number of outputs.
// If the number of block outputs do not match, insert a dummy
// constant of corresponding shape and type.
Value* InplaceConverter::MatchIfBlocksOutputForValue(
    Value* orig_data,
    Block* outer_block,
    Value* origOutput) {
  if (outer_block->owningNode()->kind() == prim::Loop)
    return outer_block->owningNode()->outputs().at(
        outer_block->owningNode()->outputs().size() - 1);

  if (outer_block->owningNode()->kind() != prim::If)
    return nullptr;
  size_t output_size = outer_block->outputs().size();

  for (size_t i = 0; i < output_size - 1; i++) {
    if (outer_block->outputs().at(i)->debugNameBase() ==
        origOutput->debugNameBase()) { // Check debug names
      outer_block->replaceOutput(i, outer_block->outputs().at(output_size - 1));
      outer_block->eraseOutput(output_size - 1);
      outer_block->owningNode()->eraseOutput(output_size - 1);
      return outer_block->owningNode()->outputs().at(i);
    }
  }

  for (Block* b : outer_block->owningNode()->blocks()) {
    if (b->outputs().size() < output_size) {
      auto clone_node =
          addDummyClone(graph_.get(), orig_data, false, b->return_node());
      b->registerOutput(clone_node->output());
      b->outputs()
          .at(b->outputs().size() - 1)
          ->copyMetadata(
              outer_block->outputs().at(output_size - 1)); // Copy debug names
    }
  }
  return outer_block->owningNode()->outputs().at(output_size - 1);
}

// clang-format off
// Register inplace op node inputs/outputs through the blocks.
// Eg. The IR before updating:
//%23 : bool = aten::eq(%22, %13)
// = prim::If(%23)
//  block0():
//    %24 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1, %spatial_size_1.1)
//    %25 : Tensor = aten::ones(%24, %12, %12, %12, %12)
//    %26 : Tensor = aten::slice(%state.1, %13, %13, %10, %11)
//    %27 : Tensor = aten::copy_(%26, %25, %9)
//    -> ()
//  block1():
//    %28 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1, %spatial_size_1.1)
//    %29 : Tensor = aten::randn(%28, %12, %12, %12, %12)
//    %30: Tensor = aten::slice(%state.1, %13, %13, %10, %11)
//    %31 : Tensor = aten::copy_(%30, %29, %9)
//    -> ()
// After updating:
//%23 : bool = aten::eq(%22, %13)
//%51 : Tensor = prim::If(%23)
//  block0():
//    %24 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1, %spatial_size_1.1)
//    %25 : Tensor = aten::ones(%24, %12, %12, %12, %12)
//    %26 : Tensor = aten::slice(%state.1, %13, %13, %10, %11)
//    %32 : Tensor?[] = prim::ListConstruct()
//    %33 : Tensor = aten::expand_as(%25, %26)
//    %38 : int = prim::Constant[value=0]()
//    %39 : int = aten::size(%state.1, %38)
//    %40 : int = prim::Constant[value=4]()
//    %41 : None = prim::Constant()
//    %42 : None = prim::Constant()
//    %43 : None = prim::Constant()
//    %44 : Tensor = aten::arange(%39, %40, %41, %42, %43)
//    %45 : int = prim::Constant[value=0]()
//    %46 : Tensor = aten::slice(%44, %45, %13, %10, %11)
//    %47 : int[] = prim::Constant[value=[-1]]()
//    %48 : Tensor = aten::view(%46, %47)
//    %49 : Tensor?[] = prim::ListConstruct(%48)
//    %50 : Tensor = aten::index_put(%state.1, %49, %33, %9)
//    -> (%50)
//  block1():
//    %28 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1, %spatial_size_1.1)
//    %29 : Tensor = aten::randn(%28, %12, %12, %12, %12)
//    %30 : Tensor = aten::slice(%state.1, %13, %13, %10, %11)
//    %35 : Tensor?[] = prim::ListConstruct()
//    %36 : Tensor = aten::expand_as(%29, %30)
//    %52 : int = prim::Constant[value=0]()
//    %53 : int = aten::size(%state.1, %52)
//    %54 : int = prim::Constant[value=4]()
//    %55 : None = prim::Constant()
//    %56 : None = prim::Constant()
//    %57 : None = prim::Constant()
//    %58 : Tensor = aten::arange(%53, %54, %55, %56, %57)
//    %59 : int = prim::Constant[value=0]()
//    %60 : Tensor = aten::slice(%58, %59, %13, %10, %11)
//    %61 : int[] = prim::Constant[value=[-1]]()
//    %62 : Tensor = aten::view(%60, %61)
//    %63 : Tensor?[] = prim::ListConstruct(%62)
//    %64 : Tensor = aten::index_put(%state.1, %63, %36, %9)
//    -> (%64)
// clang-format on
void InplaceConverter::RegisterInplaceNodeInIfBlocks(
    Value* orig_data,
    Value* new_data,
    const std::string& output_name) {
  auto outer_block = new_data->node()->owningBlock();
  auto initial_block_node = outer_block->owningNode();

  if ((nullptr == initial_block_node) ||
      (initial_block_node->kind() != prim::If)) {
    return;
  }

  auto next_block_node = initial_block_node;
  new_data->setDebugName("_output_" + output_name);
  outer_block->registerOutput(new_data);
  // Block has a new output. Add the output for the prim::If node.
  if (next_block_node->outputs().size() < outer_block->outputs().size())
    next_block_node->addOutput()->copyMetadata(new_data);

  auto next_block = next_block_node->owningBlock();
  while (nullptr != next_block->owningNode() &&
         next_block != orig_data->node()->owningBlock()) {
    // TODO:
    //    1. output(0): 0 is wrong, should be -1, since the last output is the one added,
    //       but should use a better way of keeping track of this output.
    //    2. This is recursively updating outer block, if and loop should be handled differently.
    next_block->registerOutput(next_block_node->output(0));
    next_block_node = next_block->owningNode();
    // Block has a new output. Add the output for the prim::If node.
    if (next_block_node->outputs().size() < next_block->outputs().size())
      next_block_node->addOutput()->setType(new_data->type());
    next_block = next_block_node->owningBlock();
  }
  orig_data->replaceAllUsesAfterNodeWith(
      next_block_node,
      next_block_node->outputs().at(next_block_node->outputs().size() - 1));
}

// clang-format off
// Register inplace op node inputs/outputs through the blocks.
// Eg. The IR before updating:
//   = prim::Loop(%10, %27)
//    block0(%stream_idx.1 : int):
//       = prim::Loop(%9, %27)
//        block0(%i.1 : int):
//          %36 : Tensor = aten::select(%bias.1, %26, %stream_idx.1)
//          %41 : Tensor = aten::copy_(%37, %40, %25)
//          -> (%27)
//      -> (%27)
//  After updating:
// %62 : Tensor = prim::Loop(%10, %27, %bias.2)
//    block0(%stream_idx.1 : int, %bias.3 : Tensor):
//      %61 : Tensor = prim::Loop(%9, %27, %bias.3)
//        block0(%i.1 : int, %bias.1 : Tensor):
//          %36 : Tensor = aten::select(%bias.1, %26, %stream_idx.1)
//          %59 : Tensor?[] = prim::ListConstruct(%55, %58)
//          %60 : Tensor = aten::index_put(%bias.1, %59, %45, %25)
//          -> (%27, %60)
//      -> (%27, %61)
// clang-format on
void InplaceConverter::RegisterInplaceNodeInLoopBlocks(Value* orig_data, Value* new_data) {
  Node* inplace_node = new_data->node();
  Block* outer_block = inplace_node->owningBlock();
  Node* outer_block_node = outer_block->owningNode();

  if (nullptr == outer_block_node) {
    return;
  }

  if (outer_block_node->kind() != prim::Loop)
    return;

  outer_block->registerOutput(new_data);
  std::vector<std::pair<Block*, Node*>> node_list = {
      std::make_pair(outer_block, outer_block_node)};

  outer_block_node->addOutput()->setType(new_data->type());
  auto next_block = outer_block_node->owningBlock();
  auto next_node = outer_block_node;

  while (nullptr != next_block->owningNode() &&
         next_block != orig_data->node()->owningBlock()) {
    outer_block = next_block;
    outer_block->registerOutput(
        next_node->outputs().at(next_node->outputs().size() - 1));
    next_node = outer_block->owningNode();
    next_node->addOutput()->setType(new_data->type());
    next_block = next_node->owningBlock();
    if (next_node->kind() == prim::Loop) // Do not register input if nested in
                                         // If block. Register in Loop blocks.
      node_list.emplace_back(std::make_pair(outer_block, next_node));
  }

  // Register inplace node inputs through the blocks.
  auto next_data = orig_data;
  while (!node_list.empty()) {
    auto cur_pair = node_list.back();
    // Add input to current node.
    cur_pair.second->addInput(next_data);
    // Add input to current block.
    auto cur_input = cur_pair.first->addInput();
    cur_input->setType(next_data->type());
    next_data = cur_input;
    node_list.pop_back();
  }

  // Update inplace node inputs inside the outer most block.
  outer_block_node = outer_block->owningNode();
  auto prev_data =
      outer_block_node->inputs().at(outer_block_node->inputs().size() - 1);
  for (auto node : inplace_node->owningBlock()->nodes()) {
    size_t idx = 0;
    for (auto inputs_ : node->inputs()) {
      if (inputs_ == prev_data) {
        node->replaceInput(idx, next_data);
        break;
      }
      idx++;
    }
  }

  orig_data->replaceAllUsesAfterNodeWith(
      next_node->outputs().at(0)->node(),
      next_node->outputs().at(next_node->outputs().size() - 1));
}

// void RegisterInplaceNodeThroughBlocks
//     Value* orig_data,
//     Value* new_data,
//     const std::string& output_name) {
//   Node* inplace_node = new_data->node();
//   Block* outer_block = inplace_node->owningBlock();
//   Node* outer_block_node = outer_block->owningNode();

//   if (nullptr == outer_block_node) {
//     return;
//   }

//   new_data->setDebugName("_output_" + output_name);
//   outer_block->registerOutput(new_data);


// }

// Register inplace op node inputs/outputs through the blocks.
void InplaceConverter::RegisterInplaceNodeInBlocks(Value* orig_data, Value* new_data) {
  Node* inplace_node = new_data->node();
  Block* outer_block = inplace_node->owningBlock();
  Node* outer_block_node = outer_block->owningNode();

  if (outer_block_node == nullptr)
    return;

  // Check if the value is already registered in the block
  bool registered = false;
  while (isInplaceNode(orig_data->node())) {
    // TODO: another edge case. will overtrace orig_data
    //    eg.
    //        a = tensor
    //        b = a.inplace_op()
    //        loop
    //          c = a.inplace_op()
    //          loop
    //            d = a.inplace_op()
    //        return a
    //
    //  In code handling "b = a.inplace_op()", all later usage of a are
    //  converted to b. So the below code will use `a` as `orig_data`, while
    //  in the graph, all the rest of inplace ops actually uses `b`.
    orig_data = orig_data->node()->inputs().at(0);
  }
  for (auto use : orig_data->uses()) {
    if ((use.user->owningBlock() == outer_block) &&
        (use.user->isAfter(inplace_node))) {
      size_t idx = 0;
      // TODO: GetAttr afterwards are not counted as use of same orig_data.
      for (auto input_ : use.user->inputs()) {
        if (input_ == orig_data) {
          use.user->replaceInput(idx, new_data);
          registered = true;
        }
        idx++;
      }
    }
  }
  if (registered)
    return;

  // Register inplace node outputs through the blocks.
  // RegisterInplaceNodeThroughBlocks(orig_data, new_data, orig_data->debugName());

  RegisterInplaceNodeInLoopBlocks(orig_data, new_data);

  RegisterInplaceNodeInIfBlocks(orig_data, new_data, orig_data->debugName());

  while (nullptr != outer_block->owningNode() &&
         outer_block != orig_data->node()->owningBlock()) {
    MatchIfBlocksOutputForValue(orig_data, outer_block, new_data);
    outer_block = outer_block->owningNode()->owningBlock();
  }
}

void PrepareIndexPutForONNX(Node* node, InplaceConverter* ic) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::index_put || node->kind() == aten::index_put_);
  auto placeholder_node = EncapsulatePatternIntoSubblock(node).value();
  if (node->kind() == aten::index_put_) {
    auto orig_data = placeholder_node->input();
    auto new_data = placeholder_node->output();

    // avoid confusing "registered" logic
    node->removeAllInputs();

    if (nullptr == placeholder_node->owningBlock()->owningNode()) {
      orig_data->replaceAllUsesAfterNodeWith(placeholder_node, new_data);
      return;
    }
    ic->RegisterInplaceNodeInBlocks(orig_data, new_data);
  }
}

void PrepareCopyForONNX(Node* node, InplaceConverter* ic) {
  if (node->kind() == aten::copy_) {
    // aten::copy_ can be viewed as a special case of index_put, where the
    // tensor indices input is empty.
    // Remove aten::copy_, and replace it with index_put.
    // 1. create an empty listConstruct node as indices input for index_put.
    // 2. create index_put node.

    // Tracing aten::copy_ broadcasts the rhs values.
    // 3. Apply broadcasting for scripting.
    WithInsertPoint guard(node);
    auto graph = node->owningGraph();
    auto dummy_list =
        graph->insertNode(graph->createList(OptionalType::ofTensor(), {}))
            ->output();

    auto expanded_value =
        graph->insert(aten::expand_as, {node->input(1), node->input(0)});
    expanded_value->node()->setSourceRange(node->sourceRange());
    expanded_value->copyMetadata(node->input(1));

    auto index_put = graph->insert(
        aten::index_put_,
        {node->input(0), dummy_list, expanded_value, node->input(2)});
    index_put->node()->setSourceRange(node->sourceRange());
    index_put->copyMetadata(node->output());
    node->output()->replaceAllUsesWith(index_put);

    // avoid confusing "registered" logic
    node->removeAllInputs();

    PrepareIndexPutForONNX(index_put->node(), ic);
  }
}

std::pair<Value*, Value*> PrepareIndexPutForONNX(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::index_put || node->kind() == aten::index_put_);
  auto placeholder_node = EncapsulatePatternIntoSubblock(node).value();
  node->destroy();
  return std::make_pair(placeholder_node->input(0), placeholder_node->output());
}

std::pair<Value*, Value*> PrepareCopyForONNX(Node* node) {
  TORCH_INTERNAL_ASSERT(node->kind() == aten::copy_);
  // aten::copy_ can be viewed as a special case of index_put, where the
  // tensor indices input is empty.
  // Remove aten::copy_, and replace it with index_put.
  // 1. create an empty listConstruct node as indices input for index_put.
  // 2. create index_put node.

  // Tracing aten::copy_ broadcasts the rhs values.
  // 3. Apply broadcasting for scripting.
  WithInsertPoint guard(node);
  auto graph = node->owningGraph();
  auto dummy_list =
      graph->insertNode(graph->createList(OptionalType::ofTensor(), {}))
          ->output();

  auto expanded_value =
      graph->insert(aten::expand_as, {node->input(1), node->input(0)});
  expanded_value->node()->setSourceRange(node->sourceRange());
  expanded_value->copyMetadata(node->input(1));

  auto index_put = graph->insert(
      aten::index_put_,
      {node->input(0), dummy_list, expanded_value, node->input(2)});
  index_put->node()->setSourceRange(node->sourceRange());
  index_put->copyMetadata(node->output());
  node->output()->replaceAllUsesWith(index_put);

  auto orig_data = node->input(0);
  node->destroy();

  return PrepareIndexPutForONNX(index_put->node());
}

void PrepareInplaceOpsInBlocksForONNX(Node* node, InplaceConverter* ic) {
  if (!node->kind().is_aten())
    return;

  auto name = node->schema().name();
  bool inplace_op = name.at(name.size() - 1) == '_';
  if (!inplace_op)
    return;

  auto new_schema = name.substr(0, name.size() - 1);

  Node* input_node = node->inputs().at(0)->node();
  // if (input_node->kind() != aten::select && input_node->kind() != aten::slice)
  //   return;

  auto graph = node->owningGraph();
  auto new_node = graph->create(Symbol::fromQualString(new_schema), 1);
  for (Value* input : node->inputs()) {
    new_node->addInput(input);
  }
  new_node->output()->setType(node->output()->type());
  new_node->insertBefore(node);
  new_node->setSourceRange(node->sourceRange());

  auto false_val_ = graph->insertConstant(false);

  auto new_copy = graph->create(aten::copy_, 1);
  new_copy->addInput(node->inputs().at(0));
  new_copy->addInput(new_node->output());
  new_copy->addInput(false_val_);
  new_copy->insertBefore(node);
  new_copy->setSourceRange(node->sourceRange());

  PrepareCopyForONNX(new_copy, ic);
}

std::pair<Value*, Value*> PrepareInplaceOpsInBlocksForONNX(Node* node) {
  if (!node->kind().is_aten())
    return {};

  auto name = node->schema().name();
  bool inplace_op = name.at(name.size() - 1) == '_';
  if (!inplace_op)
    return {};

  auto new_schema = name.substr(0, name.size() - 1);

  Node* input_node = node->inputs().at(0)->node();

  auto graph = node->owningGraph();
  auto new_node = graph->create(Symbol::fromQualString(new_schema), 1);
  for (Value* input : node->inputs()) {
    new_node->addInput(input);
  }
  new_node->output()->setType(node->output()->type());
  new_node->insertBefore(node);
  new_node->setSourceRange(node->sourceRange());
  node->replaceAllUsesWith(new_node);
  node->destroy();

  if (input_node->kind() == aten::select || input_node->kind() == aten::slice) {
    // Cases from a[i] = x. Convert to copy_ and eventually index_put_.
    WithInsertPoint guard(new_node);
    auto false_val_ = graph->insertConstant(false);

    auto new_copy = graph->create(aten::copy_, 1);
    new_copy->addInput(new_node->inputs().at(0));
    new_copy->addInput(new_node->output());
    new_copy->addInput(false_val_);
    new_copy->insertAfter(new_node);
    new_copy->setSourceRange(new_node->sourceRange());

    return PrepareCopyForONNX(new_copy);
  } else {
    // Direct aliasing.
    return std::make_pair(new_node->input(0), new_node->output());
  }
}

// aten::pop is inplace. The tensor list input is updated.
// This pass creates an aten::__getitem__ op to return the original output from
// aten::pop. Then it makes the original aten::pop operator return the updated
// tensor list, and replaces all later uses of that tensor list with this new
// output.
static std::pair<Value*, Value*> PrepareListPopForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::pop);
  //   %ten : Tensor = aten::pop(%seq, %pos)
  // Convert to
  //   %ten : Tensor = aten::__getitem__(%seq, %pos)
  //   %new_seq : Tensor[] = aten::pop(%seq, %pos)
  // And replace all uses of %seq afterwards with %new_seq
  Node* getitem_node =
      n->owningGraph()->create(aten::__getitem__, {n->inputs()});
  getitem_node->output()->setType(n->output()->type());
  getitem_node->insertBefore(n);
  n->output()->replaceAllUsesWith(getitem_node->output());
  n->output()->setType(n->inputs().at(0)->type());

  return std::make_pair(n->input(0), n->output());
}

static std::pair<Value*, Value*> PrepareListDeleteForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::Delete);
  n->addOutput();
  n->output()->setType(n->inputs().at(0)->type());

  return std::make_pair(n->input(0), n->output());
}

static std::pair<Value*, Value*> PrepareListAppendAndInsertForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::insert || n->kind() == aten::append);
  if (n->outputs().size() == 0) {
    n->addOutput();
    n->output()->setType(n->inputs().at(0)->type());
  }
  return std::make_pair(n->input(0), n->output());
}

static std::pair<Value*, Value*> PrepareListSetItemForONNX(Node* n) {
  TORCH_INTERNAL_ASSERT(n->kind() == aten::_set_item);
  return std::make_pair(n->input(0), n->output());
}

// Remove Mutation pass does not handle mutation on block inputs.
// To fix this, insert a clone node following the graph input:
// Example for graph input node %0:
// Before:
// graph(%0 : Tensor):
//   %5 : Tensor = aten::zero_(%0)
//   ...
// After:
// graph(%0 : Tensor):
//   %2 : None = prim::Constant()
//   %3 : Tensor = aten::clone(%0, %2)
//   %5 : Tensor = aten::zero_(%3)
//   ...

static void PrepareForRemoveMutations(MutationRemover& mr, Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareForRemoveMutations(mr, child_block);
    }
  }

  for (auto input : b->inputs()) {
    for (auto use : input->uses()) {
      Node* node = use.user;
      if (!mr.inplaceOpVariant(node)) {
        continue;
      }
      auto it = std::find(node->inputs().begin(), node->inputs().end(), input);
      if (it != node->inputs().end()) {
        int index = std::distance(node->inputs().begin(), it);
        std::cerr << "Warning: ONNX Preprocess - Removing mutation from node "
                  << node->kind().toQualString() << " on block input: '"
                  << (*it)->debugName() << "'. This changes graph semantics."
                  << std::endl;

        Node* newNode =
            addDummyClone(b->owningGraph(), input, false, b->return_node());
        TORCH_INTERNAL_ASSERT(nullptr != newNode);
        node->replaceInput(index, newNode->output());
        input->replaceAllUsesAfterNodeWith(node, newNode->output());
      }
    }
  }
}

// findSubModuleAttr function chases getAttr chains backwards to locate the
// submodules. For example: module M {
//   attributes {
//     A = <SubModule at ...>
//   }
//   ...
//   %A = prim::GetAttr[name="A"](%self)
//   ...
//   %B = prim::GetAttr[name="B"](%A)
//   ...
//   %weight = prim::GetAttr[name="scale"](%B)
//   ...
std::deque<std::string> findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    const std::shared_ptr<Graph>& graph) {
  Node* node = input->node();
  std::deque<std::string> moduleNames;

  // Loop starts from inner submodule and follows the chain until reaches the
  // top module.

  auto selfNode = graph->nodes().begin();
  auto n = *selfNode;
  while (node->outputs().at(0)->type() != n->output()->type()) {
    if (node->kind() == prim::GetAttr) {
      moduleNames.push_front(node->s(attr::name));
      node = node->inputs()[0]->node();
    } else {
      break;
    }
  }
  // Assign the inner module to attrModule.
  for (auto& moduleName : moduleNames) {
    attrModule = attrModule.attr(moduleName).toModule();
  }
  return moduleNames;
}

Value* findArgumentAsInputParam(
    const std::shared_ptr<Graph>& graph,
    std::string& name,
    IValue& attr) {
  for (auto input : graph->inputs()) {
    if (input->debugName() == name)
      return input;
  }
  throw std::runtime_error(
      "Attribute is not part of model parameters. Cannot handle SetAttr and GetAttr nodes for : " +
      name);
}

Value* InplaceConverter::registerSetAttrInBlocks(
    Block* block,
    Node* cloneNode,
    Value* origValue,
    const std::string& output_name) {
  // TODO: add check for registered
  auto orig_data = origValue;
  auto new_data = cloneNode->output();

  // Check if the value is already registered in the block
  bool registered = false;
  while (isInplaceNode(orig_data->node())) {
    orig_data = orig_data->node()->inputs().at(0);
  }
  for (auto use : orig_data->uses()) {
    if ((use.user->owningBlock() == block) &&
        (use.user->isAfter(cloneNode))) {
      size_t idx = 0;
      for (auto input_ : use.user->inputs()) {
        if (input_ == orig_data) {
          use.user->replaceInput(idx, new_data);
          registered = true;
        }
        idx++;
      }
    }
  }

  for (auto n : block->nodes()) {
    if (n->isAfter(cloneNode) && n->kind() == prim::SetAttr) {
      // Check if it is SetAttr node on this value, if so, postpone registration till then.
      // if (n->) {
      //   registered = true;
      // }
    }
  }

  if (registered)
    return nullptr;

  RegisterInplaceNodeInLoopBlocks(origValue, cloneNode->output());
  RegisterInplaceNodeInIfBlocks(origValue, cloneNode->output(), output_name);

  Value* output = nullptr;
  while (nullptr != block->owningNode() &&
         block != origValue->node()->owningBlock()) {
    output = MatchIfBlocksOutputForValue(origValue, block, cloneNode->output());
    block = block->owningNode()->owningBlock();
  }
  return output;
}

// clang-format off
// The trackAndRegisterAttributesInBlocks function tracks any instances
// of getAttr and setAttr in a sub-block and capture these nodes as inpalce
// read/write ops. This pass captures the output of setAttr in sub-block outputs
// so that it gets reflected into the outer block.
// Also, the pass matched the number of If sub-block outputs
// if a value is updated in one branch, but no updated on the other branch.
// For example:
//= prim::If(%12)
//    block0():
//      %13 : __torch__.torch.nn.modules.conv.___torch_mangle_9.Conv1d = prim::GetAttr[name="conv"](%3)
//      %b.1 : Tensor? = prim::GetAttr[name="bias"](%13)
//      ...
//      %18 : __torch__.torch.nn.modules.conv.___torch_mangle_9.Conv1d = prim::GetAttr[name="conv"](%3)
//      %19 : Tensor = aten::add(%anchors.1, %b, %6)
//       = prim::SetAttr[name="bias"](%18, %19)
//     -> ()
//    block1():
//      %20 : __torch__.torch.nn.modules.conv.___torch_mangle_9.Conv1d = prim::GetAttr[name="conv"](%3)
//      %21 : __torch__.torch.nn.modules.conv.___torch_mangle_9.Conv1d = prim::GetAttr[name="conv"](%3)
//      %22 : Tensor = prim::GetAttr[name="weight"](%21)
//      %23 : Tensor = aten::slice(%22, %7, %7, %8, %6)
//       = prim::SetAttr[name="bias"](%20, %23)
//     -> ()
// After the pass
//%_output_conv.bias.3 : Tensor = prim::If(%12)
//    block0():
//     ...
//      %18 : __torch__.torch.nn.modules.conv.___torch_mangle_9.Conv1d = prim::GetAttr[name="conv"](%3)
//      %19 : Tensor = aten::add(%anchors.1, %b, %6)
//      %_output_conv.bias.2 : Tensor = aten::clone(%19, %26)
//     -> (%_output_conv.bias.2)
//    block1():
//      %20 : __torch__.torch.nn.modules.conv.___torch_mangle_9.Conv1d = prim::GetAttr[name="conv"](%3)
//      %23 : Tensor = aten::slice(%conv.weight, %7, %7, %8, %6)
//      %31 : None = prim::Constant()
//      %_output_conv.bias.4 : Tensor = aten::clone(%23, %31)
//     -> (%_output_conv.bias.4)
// clang-format on
void InplaceConverter::trackAndRegisterAttributesInBlocks(
    Node* n,
    const Module& module_,
    std::unordered_map<std::string, Value*>& nextSetAttrValues) {
  if (n->kind() != prim::GetAttr && n->kind() != prim::SetAttr)
    return;

  auto name = n->s(attr::name);
  auto attrModule = module_;
  Value* paramConst = nullptr;

  auto moduleNames =
      findSubModuleAttr(n->inputs().at(0), name, attrModule, graph_);

  std::string fullName("");
  for (auto& name : moduleNames) {
    fullName += name + '.';
  }
  fullName += name;

  if (allAttrValues_.find(fullName) == allAttrValues_.end() &&
      attrModule.hasattr(name)) {
    auto attr = attrModule.attr(name);
    auto type = attrModule.type();
    auto slot = *type->findAttributeSlot(name);

    // Add model_parameters and model_buffers as model inputs. Order is
    // preserved based on the appearance in the graph.
    if (type->is_parameter(slot) || type->is_buffer(slot) ||
        (attr.isObject() && !attr.toObjectRef().type()->is_module())) {
      if (allAttrValues_.find(fullName) == allAttrValues_.end()) {
        paramConst = findArgumentAsInputParam(graph_, fullName, attr);
        allAttrValues_.insert({fullName, paramConst});
      }
    } else if (auto attrVal = tryInsertConstant(*graph_, attr)) {
      for (size_t i = 0; i < type->getAttributes().size(); i++) {
        if (type->getAttributeName(i) == name) {
          paramConst = *attrVal;
          allAttrValues_.insert({fullName, paramConst});
        }
      }
    } else {
      GRAPH_DEBUG(
          attr.type()->cast<ClassType>() ? "" : "attribute: ",
          name,
          " is not materializable.");
      return;
    }
  }

  if (n->kind() == prim::SetAttr) { // Handle SetAttr node
    if (attrModule.hasattr(name)) {
      // If inside a block, keep the output value to register in block
      // output.
      auto block_ = n->owningBlock();
      Node* cloneNode =
          addDummyClone(graph_.get(), n->inputs().at(1), true, n);
      if (block_->owningNode() &&
          (block_->owningNode()->kind() == prim::If ||
           block_->owningNode()->kind() == prim::Loop)) {
        auto attrValue = (setAttrValues_.find(fullName) != setAttrValues_.end())
            ? setAttrValues_[fullName]
            : allAttrValues_[fullName];

        auto blockOutput = registerSetAttrInBlocks(
            block_, cloneNode, attrValue, fullName);

        if (nullptr != blockOutput) {
          nextSetAttrValues[fullName] = blockOutput;
        }
      }
      // SetAttr writes a value to an attr. Keep this
      // in the setAttrValues map.
      setAttrValues_[fullName] = cloneNode->output();
    }
  } else if (n->kind() == prim::GetAttr) { // Handle GetAttr node
    allAttrModules_[fullName] = n->input(0);
    if (setAttrValues_.find(fullName) != setAttrValues_.end()) {
      // Attr has been set earlier in the graph.
      // Read its value from setAttrValues map.
      auto set_attr_node_input = setAttrValues_[fullName];
      // Clone SetAttr input
      n->output()->replaceAllUsesAfterNodeWith(n, set_attr_node_input);
    } else if (allAttrValues_.find(fullName) != allAttrValues_.end()) {
      // Attr has not been set earlier in the graph. Replace it with the
      // graph parameter if exists.
      n->output()->replaceAllUsesWith(allAttrValues_[fullName]);
      n->removeAllInputs();
    }
  }
}

// clang-format off
// The registerInplaceOpAsBlockOutputs function tracks inplace op
// (like aten::copy_ or aten::append) outputs as sub-block output.
// Also, match the number of If sub-block outputs
// if a value is updated in one branch, but no updated on the other branch.
// For example:
// = prim::If(%30)
//    block0():
//      ...
//      %35 : Tensor = aten::copy_(%state_copy.1, %33, %12)
//      -> ()
//    block1():
//      ...
//      %40 : Tensor = aten::copy_(%state.1, %38, %12)
//      -> ()
//
// After the pass
//%_output_state_copy.1 : Tensor, %_output_state.1 : Tensor = prim::If(%30)
//    block0():
//      %_output_state.2 : Tensor = aten::clone(%state.1, %59)
//      ...
//      %_output_state_copy.3 : Tensor = onnx::Placeholder[name="index_put_"](%state_copy.1)...
//      ...
//      -> (%_output_state_copy.3, %_output_state.2)
//    block1():
//      %50 : None = prim::Constant()
//      %_output_state_copy.2 : Tensor = aten::clone(%state_copy.1, %50)
//      ...
//      %_output_state.3 : Tensor = onnx::Placeholder[name="index_put_"](%state.1)...
//       ...
//      -> (%_output_state_copy.2, %_output_state.3)
std::unordered_map<std::string, Value*> InplaceConverter::registerInplaceOpAsBlockOutputs(Block* block) {
  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);
  std::unordered_map<std::string, Value*> nextSetAttrValues = {};

  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed
    // printf("before node: ");
    // n->dump();
    // printf("pre graph: %s\n", n->owningGraph()->toString().c_str());
    if (nullptr != module_ &&
        (n->kind() == prim::GetAttr || n->kind() == prim::SetAttr)) {
      Module moduleClone = (*module_);
      trackAndRegisterAttributesInBlocks(
          n,
          moduleClone,
          nextSetAttrValues);
    } else if (n->kind() == aten::copy_) {
      PrepareCopyForONNX(n, this);
      GRAPH_UPDATE("After copy:", graph_->toString());
    } else if (n->kind() == aten::index_put || n->kind() == aten::index_put_) {
      PrepareIndexPutForONNX(n, this);
    } else if (mr_->inplaceOpVariant(n)) {
      // TODO: not sure the effectiveness of this.
      //       Since aliasDb is not out of sync with current graph.
      PrepareInplaceOpsInBlocksForONNX(n, this);
      GRAPH_UPDATE("After inplace op:", graph_->toString());
    } else if (n->kind() == aten::pop) {
      PrepareListPopForONNX(n, this);
    } else if (n->kind() == aten::insert || n->kind() == aten::append) {
      GRAPH_UPDATE("Before registering inplace node in blocks");
      PrintSnapshotForDebug(n);
      PrepareListAppendAndInsertForONNX(n, this);
      // Check if input of n is something in setAttrValues, and update it.
      auto orig_data = n->input(0);
      GRAPH_UPDATE("orig_data for list append: ", orig_data->debugName());
      PrintSnapshotForDebug(n);
      for (auto iter : allAttrValues_) {
        if (iter.second == orig_data) {
          auto attr_full_name = iter.first;
          auto attr_name = attr_full_name.substr(attr_full_name.rfind('.') + 1);
          // TODO: find module of this value.
          //       need to create a map for this.
          auto set_attr_node = graph_->create(prim::SetAttr, {allAttrModules_[attr_full_name], n->output()}, 0);
          set_attr_node->s_(attr::name, attr_name);
          set_attr_node->insertAfter(n);
          break;
        }
      }
    } else if (n->kind() == aten::Delete) {
      PrepareListDeleteForONNX(n, this);
    } else if (n->kind() == aten::_set_item) {
      PrepareListSetItemForONNX(n, this);
    } else { // for prim::If and prim::Loop nodes with blocks.
      for (Block* sub_block : n->blocks()) {
        std::unordered_map<std::string, Value*> map_ =
            registerInplaceOpAsBlockOutputs(sub_block);
        std::unordered_map<std::string, Value*>::iterator mapIt;
        for (mapIt = map_.begin(); mapIt != map_.end(); mapIt++) {
          setAttrValues_[mapIt->first] = mapIt->second;
        }
      }
    }
    // printf("after graph: %s\n", n->owningGraph()->toString().c_str());
  }
  return nextSetAttrValues;
}

void InplaceConverter::ValueTracker::init(const std::shared_ptr<Graph>& graph) {
  alias_to_value_ = {};
  value_to_sorted_aliases_ = {};
  graph_ = graph;
  root_block_ = graph->block();
}

std::string InplaceConverter::ValueTracker::toString() const {
  std::stringstream ss;

  // ss << "Current graph: " << graph_->toString() << std::endl;
  ss << "Tracking " << value_to_sorted_aliases_.size() << " individual values." << std::endl;
  ss << "value_to_sorted_aliases_: " << std::endl;
  size_t idx = 0;
  for (auto it : value_to_sorted_aliases_) {
    ss << "Value[" << idx << "]: " << it.first->debugName() << std::endl;
    ss << "  Mapping to ";
    for (auto v : it.second) {
      ss << v->debugName() << " ";
    }
    ss << std::endl;
    idx++;
  }

  ss << "alias_to_value_: " << std::endl;
  for (auto it : alias_to_value_) {
    ss << "  Alias " << it.first->debugName();
    ss << " map to " << it.second->debugName() << std::endl;
  }

  return ss.str();
}

std::vector<std::tuple<Value*, Node*, Block*>> InplaceConverter::ValueTracker::sortAliasOfValue(const Value* v) const {
  std::vector<std::tuple<Value*, Node*, Block*>> res = {};

  // TORCH_INTERNAL_ASSERT(value_to_sorted_aliases_.find(v) != value_to_sorted_aliases.end());
  // for (auto v : value_to_aliases_[v]) {

  // }

  return res;
}

// TODO: maybe don't need n, is it true that for all cases n should be just new_v->node()?
void InplaceConverter::ValueTracker::registerSetValue(Value* old_v, Value* new_v) {
  GRAPH_UPDATE("Calling registerSetValue with old_v: ", old_v->debugName(), " new_v: ", new_v->debugName());
  GRAPH_UPDATE(this->toString());
  auto* n = new_v->node();
  auto* owning_block = n->owningBlock();

  if (alias_to_value_.find(old_v) == alias_to_value_.end()) {
    alias_to_value_[old_v] = old_v;
    value_to_sorted_aliases_[old_v] = {old_v};
  }

  auto root_v = alias_to_value_[old_v];
  alias_to_value_[new_v] = root_v;
  // auto alias_order = sortAliasOfValue(root_v);
  auto &sorted_alias = value_to_sorted_aliases_[root_v];
  sorted_alias.insert(new_v);

  // check if new_v is registered as block output for if & loop subblock.
  // NOTE: minor thought, if v is actually not used, dce probably not able to pick up this block output.
  if (owning_block == root_block_) {
    return;
  }
  auto* owning_blocknode = owning_block->owningNode();
  TORCH_INTERNAL_ASSERT(nullptr != owning_blocknode);
  auto owning_block_nkind = owning_blocknode->kind();
  if (owning_block_nkind != prim::Loop && owning_block_nkind != prim::If) {
    return;
  }

  bool registered = std::any_of(owning_block->outputs().begin(), owning_block->outputs().end(), [&sorted_alias](Value* out) {
    return std::any_of(sorted_alias.begin(), sorted_alias.end(), [&out](Value* alias) {
      return alias == out;
    });
  });

  if (!registered) {
    if (owning_block_nkind == prim::Loop) {
      owning_block->registerOutput(new_v);
      auto new_block_in = owning_block->addInput();
      sorted_alias.insert(new_block_in);
      alias_to_value_[new_block_in] = root_v;
      owning_blocknode->addInput(root_v);
      auto* new_blocknode_out = owning_blocknode->addOutput();
      registerSetValue(root_v, new_blocknode_out);
    } else if (owning_block_nkind == prim::If) {
      // Only register as output, if there this value comes from outer block.
      auto isAncestor = [](const Block* a, const Block* b) {
        while (b && b->owningNode()) {
          if (a == b) {
            return true;
          }
          b = b->owningNode()->owningBlock();
        }
        return a == b;
      };
      bool from_outer = std::any_of(sorted_alias.begin(), sorted_alias.end(), [&owning_blocknode, isAncestor](Value* alias) {
        return isAncestor(alias->node()->owningBlock(), owning_blocknode->owningBlock());
      });
      if (from_outer) {
        for (auto* if_sub_block : owning_blocknode->blocks()) {
          if (owning_block == if_sub_block) {
            if_sub_block->registerOutput(new_v);
          } else {
            if_sub_block->registerOutput(root_v);
          }
        }
        auto* new_blocknode_out = owning_blocknode->addOutput();
        registerSetValue(root_v, new_blocknode_out);
      }
    }
  }

  GRAPH_UPDATE("After registerSetValue for in: ", old_v->debugName(), ", out: ", new_v->debugName(), ". tracker status:");
  GRAPH_UPDATE(this->toString());
}

void InplaceConverter::ValueTracker::passUpdateValueUse(Block* block) {
  auto updateValueUse = [this](Node* n) {
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      auto* in = n->input(i);
      auto* alias = this->findAliasForValueAtNode(in, n);
      if (alias != in) {
        n->replaceInput(i, alias);
        GRAPH_UPDATE("Replacing ", in->debugName(), " with ", alias->debugName(), " for ", *n);
      }
    }
  };

  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    updateValueUse(n);

    auto nkind = n->kind();
    if (nkind == prim::If || nkind == prim::Loop) {
      for (auto* sub_block : n->blocks()) {
        passUpdateValueUse(sub_block);
      }
    }
  }
  updateValueUse(block->return_node());
}

Value* InplaceConverter::ValueTracker::findAliasForValueAtNode(Value* v, const Node* n) const {
  GRAPH_UPDATE("Finding alias for value:", v->debugName(), " at node ", *n);
  if (alias_to_value_.find(v) == alias_to_value_.end()) {
    // This value was not affected by any inplace operator.
    return v;
  }

  auto* root_v = alias_to_value_.find(v)->second;
  TORCH_INTERNAL_ASSERT(value_to_sorted_aliases_.find(root_v) != value_to_sorted_aliases_.end());
  const auto& aliases = value_to_sorted_aliases_.find(root_v)->second;

  // alias is accessible only if
  // 1. alias owning block is ancestor of n.
  // 2. alias owning node is before n.
  // return the last alias that satisfies this condition.
  auto isAncestor = [](const Block* a, const Block* b) {
    while (b && b->owningNode()) {
      if (a == b) {
        return true;
      }
      b = b->owningNode()->owningBlock();
    }
    return a == b;
  };
  Value* found_alias = nullptr;
  for (auto* alias : aliases) {
    auto* alias_n = alias->node();
    if (alias_n->isBefore(n) && isAncestor(alias_n->owningBlock(), n->owningBlock())) {
      found_alias = alias;
    }
  }

  TORCH_INTERNAL_ASSERT(nullptr != found_alias);

  return found_alias;
}

void InplaceConverter::gatherAttrNameInitialValueMap(
    Block* block,
    std::unordered_map<std::string, Value*>& attr_name_value_map,
    std::unordered_map<Node*, std::string>& attr_node_fullname_map) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    for (auto* sub_block : n->blocks()) {
      gatherAttrNameInitialValueMap(sub_block, attr_name_value_map, attr_node_fullname_map);
    }

    if (n->kind() != prim::GetAttr && n->kind() != prim::SetAttr)
      continue;

    auto name = n->s(attr::name);
    auto attrModule = *module_;
    Value* paramConst = nullptr;

    auto moduleNames =
        findSubModuleAttr(n->inputs().at(0), name, attrModule, graph_);

    std::string fullName("");
    for (auto& name : moduleNames) {
      fullName += name + '.';
    }
    fullName += name;

    attr_node_fullname_map.insert({n, fullName});

    if (attr_name_value_map.find(fullName) == attr_name_value_map.end() &&
        attrModule.hasattr(name)) {
      auto attr = attrModule.attr(name);
      auto type = attrModule.type();
      auto slot = *type->findAttributeSlot(name);

      // Add model_parameters and model_buffers as model inputs. Order is
      // preserved based on the appearance in the graph.
      WithInsertPoint guard(graph_->nodes().front());
      if (type->is_parameter(slot) || type->is_buffer(slot) ||
          (attr.isObject() && !attr.toObjectRef().type()->is_module())) {
        paramConst = findArgumentAsInputParam(graph_, fullName, attr);
        attr_name_value_map.insert({fullName, paramConst});
      } else if (auto attrVal = tryInsertConstant(*graph_, attr)) {
        for (size_t i = 0; i < type->getAttributes().size(); i++) {
          if (type->getAttributeName(i) == name) {
            paramConst = *attrVal;
            attr_name_value_map.insert({fullName, paramConst});
          }
        }
      } else {
        // TODO: error out or continue?
        GRAPH_DEBUG(
            attr.type()->cast<ClassType>() ? "" : "attribute: ",
            name,
            " is not materializable.");
      }
    }

    // Create dummy initial value.
    if (attr_name_value_map.find(fullName) == attr_name_value_map.end()) {
      auto* noneNode = graph_->create(prim::Constant);
      noneNode->output()->setType(NoneType::get());
      noneNode->insertBefore(graph_->nodes().front());
      attr_name_value_map.insert({fullName, noneNode->output()});
    }
  }

}

void InplaceConverter::replaceAttrWithInplaceOps(Block* block,
    const std::unordered_map<std::string, Value*>& attr_name_value_map,
    const std::unordered_map<Node*, std::string>& attr_node_fullname_map) {

  for (auto pair : attr_node_fullname_map) {
    auto* n = pair.first;
    auto fullName = pair.second;
    auto find_init_val = attr_name_value_map.find(fullName);
    TORCH_INTERNAL_ASSERT(find_init_val != attr_name_value_map.end());

    TORCH_INTERNAL_ASSERT(n->kind() == prim::GetAttr || n->kind() == prim::SetAttr);
    if (n->kind() == prim::SetAttr) {
      // auto* copyNode = graph_->create(aten::copy_, 1);
      // WithInsertPoint guard(graph_->nodes().front());
      // auto false_val_ = graph_->insertConstant(false);

      // NOTE: directly create index_put_ to avoid expanding for copy_.
      WithInsertPoint guard(n);
      auto false_val_ = graph_->insertConstant(false);
      auto dummy_list =
          graph_->insertNode(graph_->createList(OptionalType::ofTensor(), {}))
              ->output();

      auto* index_put_node = graph_->create(aten::index_put_, 1);
      index_put_node->addInput(find_init_val->second);
      index_put_node->addInput(dummy_list);
      index_put_node->addInput(n->input(1));
      index_put_node->addInput(false_val_);
      index_put_node->setSourceRange(n->sourceRange());

      index_put_node->insertBefore(n);


      // copyNode->addInput(find_init_val->second);
      // copyNode->addInput(n->input(1));
      // copyNode->addInput(false_val_);
      // copyNode->insertBefore(n);
      // copyNode->setSourceRange(n->sourceRange());
    } else {
      // prim::GetAttr
      n->output()->replaceAllUsesWith(find_init_val->second);
    }

    n->destroy();
  }
}

void InplaceConverter::convertGetSetAttrToInplaceOps(Block* block) {
  // First pass over graph, to gather all attribute names,
  // and their intial values.
  // Create dummy initial values for attributes.
  // In the end of this pass,
  // these dummy initial values should have zero uses, and safely removed.
  // Otherwise it will imply error in model for using uninitialized values.
  std::unordered_map<std::string, Value*> attr_name_value_map = {};
  std::unordered_map<Node*, std::string> attr_node_fullname_map = {};
  gatherAttrNameInitialValueMap(block, attr_name_value_map, attr_node_fullname_map);
  GRAPH_UPDATE("Graph after gatherAttrNameInitialValueMap", graph_->toString());

  // Second pass over graph,
  // replace GetAttr with initial value,
  // and replace SetAttr with aten::copy_(initial_value, new_value)
  replaceAttrWithInplaceOps(block, attr_name_value_map, attr_node_fullname_map);
}



// Register Inplace Ops As Block Outputs
// Inplace operations like aten::copy_ or aten::append that are inside
// sub-blocks would require the output of the operation to be captured
// as sub-block output, so that the inplace operation would be visible
// to the outer block.
// We also consider setAttr node an inplace op, and handle those
// similarly by tracking the output as sub-block outputs.
void InplaceConverter::RegisterInplaceOpAsBlockOutputs() {
  convertGetSetAttrToInplaceOps(graph_->block());
  GRAPH_UPDATE("Graph after convertGetSetAttrToInplaceOps", graph_->toString());
  registerInplaceOpAsBlockOutputs(graph_->block());
}

void InplaceConverter::convertInplaceOps(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    auto nkind = n->kind();
    if (n->kind() == aten::copy_) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareCopyForONNX(n);
      vt_.registerSetValue(orig_data, new_out);
    } else if (n->kind() == aten::index_put || n->kind() == aten::index_put_) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareIndexPutForONNX(n);
      if (nkind == aten::index_put_) {
        vt_.registerSetValue(orig_data, new_out);
      }
    } else if (n->kind() == aten::insert || n->kind() == aten::append) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareListAppendAndInsertForONNX(n);
      vt_.registerSetValue(orig_data, new_out);
    } else if (mr_->inplaceOpVariant(n)) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareInplaceOpsInBlocksForONNX(n);
      if (nullptr != new_out) {
        vt_.registerSetValue(orig_data, new_out);
      }
    } else if (n->kind() == aten::pop) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareListPopForONNX(n);
      vt_.registerSetValue(orig_data, new_out);
    } else if (n->kind() == aten::Delete) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareListDeleteForONNX(n);
      vt_.registerSetValue(orig_data, new_out);
    } else if (n->kind() == aten::_set_item) {
      Value *orig_data, *new_out;
      std::tie(orig_data, new_out) = PrepareListSetItemForONNX(n);
      vt_.registerSetValue(orig_data, new_out);
    } else { // for prim::If and prim::Loop nodes with blocks.
      // All outputs are replacing some outer values.
      // For those already captured, it implies that new values are assigned to the alias.
      for (Block* sub_block : n->blocks()) {
        convertInplaceOps(sub_block);
      }
    }
  }
}

void InplaceConverter::convertInplaceOps() {
  convertInplaceOps(graph_->block());
  GRAPH_UPDATE("Graph after convertInplaceOps: ", graph_->toString());
  GRAPH_UPDATE(vt_.toString());
}

void InplaceConverter::convertMutationForONNX() {
  convertGetSetAttrToInplaceOps(graph_->block());
  GRAPH_UPDATE("Graph after convertGetSetAttrToInplaceOps", graph_->toString());
  vt_.init(graph_);
  convertInplaceOps();
  vt_.passUpdateValueUse(graph_->block());
}

} // namespace

void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model = nullptr) {
  MutationRemover mr(graph);
  ImplicitCastForBinaryInplaceOps(graph->block());
  PrepareForRemoveMutations(mr, graph->block());
  RemoveTensorMutation(graph);
  RemoveListMutation(graph);
  InplaceConverter ic(graph, &mr, model);
  ic.convertMutationForONNX();
}

} // namespace jit
} // namespace torch
