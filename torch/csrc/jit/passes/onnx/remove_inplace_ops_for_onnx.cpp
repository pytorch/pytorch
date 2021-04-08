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

bool IsInplaceNode(const Node* n) {
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
Value* MatchIfBlocksOutputForValue(
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
          addDummyClone(b->owningGraph(), orig_data, false, b->return_node());
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
void RegisterInplaceNodeInIfBlocks(
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
void RegisterInplaceNodeInLoopBlocks(Value* orig_data, Value* new_data) {
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

// Register inplace op node inputs/outputs through the blocks.
void RegisterInplaceNodeInBlocks(Value* orig_data, Value* new_data) {
  Node* inplace_node = new_data->node();
  Block* outer_block = inplace_node->owningBlock();
  Node* outer_block_node = outer_block->owningNode();

  if (outer_block_node == nullptr)
    return;

  // Check if the value is already registered in the block
  bool registered = false;
  while (IsInplaceNode(orig_data->node())) {
    orig_data = orig_data->node()->inputs().at(0);
  }
  for (auto use : orig_data->uses()) {
    if ((use.user->owningBlock() == outer_block) &&
        (use.user->isAfter(inplace_node))) {
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
  if (registered)
    return;

  // Register inplace node outputs through the blocks.
  RegisterInplaceNodeInLoopBlocks(orig_data, new_data);

  RegisterInplaceNodeInIfBlocks(orig_data, new_data, orig_data->debugName());

  while (nullptr != outer_block->owningNode() &&
         outer_block != orig_data->node()->owningBlock()) {
    MatchIfBlocksOutputForValue(orig_data, outer_block, new_data);
    outer_block = outer_block->owningNode()->owningBlock();
  }
}

void PrepareIndexPutForONNX(Node* node) {
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::index_put || node->kind() == aten::index_put_);
  auto placeholder_node = EncapsulatePatternIntoSubblock(node).value();
  if (node->kind() == aten::index_put_) {
    auto orig_data = placeholder_node->input();
    auto new_data = placeholder_node->output();

    if (nullptr == placeholder_node->owningBlock()->owningNode()) {
      orig_data->replaceAllUsesAfterNodeWith(placeholder_node, new_data);
      return;
    }
    RegisterInplaceNodeInBlocks(orig_data, new_data);
  }
}

void PrepareCopyForONNX(Node* node) {
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

    PrepareIndexPutForONNX(index_put->node());
  }
}

void PrepareInplaceOpsInBlocksForONNX(Node* node) {
  if (!node->kind().is_aten())
    return;

  auto name = node->schema().name();
  bool inplace_op = name.at(name.size() - 1) == '_';
  if (!inplace_op)
    return;

  auto new_schema = name.substr(0, name.size() - 1);

  Node* input_node = node->inputs().at(0)->node();
  if (input_node->kind() != aten::select && input_node->kind() != aten::slice)
    return;

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
  new_copy->addInput(input_node->output());
  new_copy->addInput(new_node->output());
  new_copy->addInput(false_val_);
  new_copy->insertBefore(node);
  new_copy->setSourceRange(node->sourceRange());

  PrepareCopyForONNX(new_copy);
}

// aten::pop is inplace. The tensor list input is updated.
// This pass creates an aten::__getitem__ op to return the original output from
// aten::pop. Then it makes the original aten::pop operator return the updated
// tensor list, and replaces all later uses of that tensor list with this new
// output.
static void PrepareListPopForONNX(Node* n) {
  if (n->kind() == aten::pop) {
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

    if (nullptr == n->owningBlock()->owningNode()) {
      n->inputs().at(0)->replaceAllUsesAfterNodeWith(n, n->output());
      return;
    }
    RegisterInplaceNodeInBlocks(n->inputs().at(0), n->output());
  }
}

static void PrepareListDeleteForONNX(Node* n) {
  if (n->kind() == aten::Delete) {
    n->addOutput();
    n->output()->setType(n->inputs().at(0)->type());

    if (nullptr == n->owningBlock()->owningNode()) {
      n->inputs().at(0)->replaceAllUsesAfterNodeWith(n, n->output());
      return;
    }
    RegisterInplaceNodeInBlocks(n->inputs().at(0), n->output());
  }
}

static void PrepareListAppendAndInsertForONNX(Node* n) {
  if (n->kind() == aten::insert || n->kind() == aten::append) {
    if (n->outputs().size() == 0) {
      n->addOutput();
      n->output()->setType(n->inputs().at(0)->type());
    }

    if (nullptr == n->owningBlock()->owningNode()) {
      n->inputs().at(0)->replaceAllUsesAfterNodeWith(n, n->output());
      return;
    }
    RegisterInplaceNodeInBlocks(n->inputs().at(0), n->output());
  }
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

Value* registerSetAttrInBlocks(
    const std::shared_ptr<Graph>& graph,
    Block* block,
    Node* cloneNode,
    Value* origValue,
    const std::string& output_name) {
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
void trackAndRegisterAttributesInBlocks(
    Node* n,
    const std::shared_ptr<Graph>& graph,
    const Module& module_,
    std::unordered_map<std::string, Value*>& allAttrValues,
    std::unordered_map<std::string, Value*>& setAttrValues,
    std::unordered_map<std::string, Value*>& nextSetAttrValues) {
  if (n->kind() != prim::GetAttr && n->kind() != prim::SetAttr)
    return;

  auto name = n->s(attr::name);
  auto attrModule = module_;
  Value* paramConst = nullptr;

  auto moduleNames =
      findSubModuleAttr(n->inputs().at(0), name, attrModule, graph);

  std::string fullName("");
  for (auto& name : moduleNames) {
    fullName += name + '.';
  }
  fullName += name;

  if (allAttrValues.find(fullName) == allAttrValues.end() &&
      attrModule.hasattr(name)) {
    auto attr = attrModule.attr(name);
    auto type = attrModule.type();
    auto slot = *type->findAttributeSlot(name);

    // Add model_parameters and model_buffers as model inputs. Order is
    // preserved based on the appearance in the graph.
    if (type->is_parameter(slot) || type->is_buffer(slot) ||
        (attr.isObject() && !attr.toObjectRef().type()->is_module())) {
      if (allAttrValues.find(fullName) == allAttrValues.end()) {
        paramConst = findArgumentAsInputParam(graph, fullName, attr);
        allAttrValues.insert({fullName, paramConst});
      }
    } else if (auto attrVal = tryInsertConstant(*graph, attr)) {
      for (size_t i = 0; i < type->getAttributes().size(); i++) {
        if (type->getAttributeName(i) == name) {
          paramConst = *attrVal;
          allAttrValues.insert({fullName, paramConst});
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
          addDummyClone(block_->owningGraph(), n->inputs().at(1), true, n);
      if (block_->owningNode() &&
          (block_->owningNode()->kind() == prim::If ||
           block_->owningNode()->kind() == prim::Loop)) {
        auto attrValue = (setAttrValues.find(fullName) != setAttrValues.end())
            ? setAttrValues[fullName]
            : allAttrValues[fullName];

        auto blockOutput = registerSetAttrInBlocks(
            graph, block_, cloneNode, attrValue, fullName);

        nextSetAttrValues[fullName] = blockOutput;
      }
      // SetAttr writes a value to an attr. Keep this
      // in the setAttrValues map.
      setAttrValues[fullName] = cloneNode->output();
    }
  } else if (n->kind() == prim::GetAttr) { // Handle GetAttr node
    if (setAttrValues.find(fullName) != setAttrValues.end()) {
      // Attr has been set earlier in the graph.
      // Read its value from setAttrValues map.
      auto set_attr_node_input = setAttrValues[fullName];
      // Clone SetAttr input
      n->output()->replaceAllUsesAfterNodeWith(n, set_attr_node_input);
    } else if (allAttrValues.find(fullName) != allAttrValues.end()) {
      // Attr has not been set earlier in the graph. Replace it with the
      // graph parameter if exists.
      n->output()->replaceAllUsesWith(allAttrValues[fullName]);
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
std::unordered_map<std::string, Value*> registerInplaceOpAsBlockOutputs(
    Block* block,
    const std::shared_ptr<Graph>& graph,
    std::unordered_map<std::string, Value*>& allAttrValues,
    std::unordered_map<std::string, Value*>& setAttrValues,
    MutationRemover& mr,
    Module* module_ = nullptr) {
  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);
  std::unordered_map<std::string, Value*> nextSetAttrValues = {};

  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    if (nullptr != module_ &&
        (n->kind() == prim::GetAttr || n->kind() == prim::SetAttr)) {
      Module moduleClone = (*module_);
      trackAndRegisterAttributesInBlocks(
          n,
          graph,
          moduleClone,
          allAttrValues,
          setAttrValues,
          nextSetAttrValues);
    } else if (n->kind() == aten::copy_) {
      PrepareCopyForONNX(n);
    } else if (n->kind() == aten::index_put || n->kind() == aten::index_put_) {
      PrepareIndexPutForONNX(n);
    } else if (mr.inplaceOpVariant(n)) {
      PrepareInplaceOpsInBlocksForONNX(n);
    } else if (n->kind() == aten::pop) {
      PrepareListPopForONNX(n);
    } else if (n->kind() == aten::insert || n->kind() == aten::append) {
      PrepareListAppendAndInsertForONNX(n);
    } else if (n->kind() == aten::Delete) {
      PrepareListDeleteForONNX(n);
    } else { // for prim::If and prim::Loop nodes with blocks.
      for (Block* sub_block : n->blocks()) {
        std::unordered_map<std::string, Value*> map_ =
            registerInplaceOpAsBlockOutputs(
                sub_block, graph, allAttrValues, setAttrValues, mr, module_);
        std::unordered_map<std::string, Value*>::iterator mapIt;
        for (mapIt = map_.begin(); mapIt != map_.end(); mapIt++) {
          setAttrValues[mapIt->first] = mapIt->second;
        }
      }
    }
  }
  return nextSetAttrValues;
}

// Register Inplace Ops As Block Outputs
// Inplace operations like aten::copy_ or aten::append that are inside
// sub-blocks would require the output of the operation to be captured
// as sub-block output, so that the inplace operation would be visible
// to the outer block.
// We also consider setAttr node an inplace op, and handle those
// similarly by tracking the output as sub-block outputs.
void RegisterInplaceOpAsBlockOutputs(
    Module* module,
    const std::shared_ptr<Graph>& graph,
    MutationRemover& mr) {
  // A map of names and values of referenced attributes, to avoid duplicates.
  std::unordered_map<std::string, Value*> allAttrValues = {};
  // A map of names and values of set attributes, to track mutations.
  std::unordered_map<std::string, Value*> setAttrValues = {};

  registerInplaceOpAsBlockOutputs(
      graph->block(), graph, allAttrValues, setAttrValues, mr, module);
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
  RegisterInplaceOpAsBlockOutputs(model, graph, mr);
}

} // namespace jit
} // namespace torch
