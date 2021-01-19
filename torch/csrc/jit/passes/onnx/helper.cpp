#include <aten/src/ATen/InitialTensorOptions.h>
#include <onnx/onnx_pb.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;

}

ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  for (auto& input : b->inputs()) {
    auto it = paramsDict.find(input->debugName());
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  return valsToParamsMap;
}

void eraseUnusedBlockInputs(Block* b) {
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}

void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  paramsDict.clear();
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

c10::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type) {
  switch (onnx_type) {
    case ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      return at::ScalarType::Undefined;
    case ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return at::kFloat;
    case ::ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return at::kByte;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return at::kChar;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return at::kShort;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return at::kInt;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return at::kLong;
    case ::ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return at::kBool;
    case ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return at::kHalf;
    case ::ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return at::kDouble;
    case ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
      return at::kComplexFloat;
    case ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      return at::kComplexDouble;
    case ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return at::kBFloat16;
    default:
      TORCH_CHECK("unexpected tensor scalar type");
  }
  return c10::optional<at::ScalarType>{};
}

Node* addNodeToBlock(Block* block, Symbol kind, ArrayRef<Value*> inputs) {
  auto new_node = block->appendNode(block->owningGraph()->create(kind));
  for (auto input : inputs) {
    auto new_input = new_node->addInput(input);
  }
  return new_node;
}

Value* addInputToBlock(Block* block) {
  return block->addInput();
}

void addDummyConstantBlockOutput(Block* b, Value* orig_data) {
  auto graph = b->owningGraph();
  auto newNode = graph->create(aten::clone, /*num_outputs =*/1);
  newNode->addInput(orig_data);

  auto* noneNode = graph->create(prim::Constant);
  noneNode->output()->setType(NoneType::get());
  newNode->addInput(noneNode->output());
  newNode->output()->copyMetadata(orig_data);

  newNode->insertBefore(b->return_node());
  noneNode->insertBefore(newNode);
  b->registerOutput(newNode->output());
}

bool isValueUsedInBlock(Block* b, Value* val) {
  for (auto node : b->nodes()) {
    for (auto inputs_ : node->inputs()) {
      if (inputs_ == val) {
        return true;
      }
    }
    for (auto inner_block : node->blocks()) {
      if (isValueUsedInBlock(inner_block, val))
        return true;
    }
  }
  return false;
}

// Check If then/else blocks to match the number of outputs.
// If the number of block outputs do not match, insert a dummy
// constant of corresponding shape and type.
void matchIfBlockOutputs(
    Value* orig_data,
    Node* block_node,
    Block* outer_block,
    Node* next_node) {
  auto prev_data = block_node->input(0);

  int output_size = outer_block->outputs().size();
  for (Block* b : outer_block->owningNode()->blocks()) {
    if (b->outputs().size() < output_size && !isValueUsedInBlock(b, prev_data))
      addDummyConstantBlockOutput(b, orig_data);
  }
}

// Register inplace op node inputs/outputs through the blocks.
// Eg. The IR before updating:
//%23 : bool = aten::eq(%22, %13)
// = prim::If(%23) # test/onnx/test_pytorch_onnx_onnxruntime.py:6243:12
//  block0():
//    %24 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1,
//    %spatial_size_1.1) %25 : Tensor = aten::ones(%24, %12, %12, %12, %12) %26
//    : Tensor = aten::slice(%state.1, %13, %13, %10, %11) %27 : Tensor =
//    aten::copy_(%26, %25, %9)
//    -> ()
//  block1():
//    %28 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1,
//    %spatial_size_1.1) %29 : Tensor = aten::randn(%28, %12, %12, %12, %12) %30
//    : Tensor = aten::slice(%state.1, %13, %13, %10, %11) %31 : Tensor =
//    aten::copy_(%30, %29, %9)
//    -> ()
// After updating:
//%23 : bool = aten::eq(%22, %13)
//%51 : Tensor = prim::If(%23)
//  block0():
//    %24 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1,
//    %spatial_size_1.1) %25 : Tensor = aten::ones(%24, %12, %12, %12, %12) %26
//    : Tensor = aten::slice(%state.1, %13, %13, %10, %11) %32 : Tensor?[] =
//    prim::ListConstruct() %33 : Tensor = aten::expand_as(%25, %26) %38 : int =
//    prim::Constant[value=0]() %39 : int = aten::size(%state.1, %38) %40 : int
//    = prim::Constant[value=4]() %41 : None = prim::Constant() %42 : None =
//    prim::Constant() %43 : None = prim::Constant() %44 : Tensor =
//    aten::arange(%39, %40, %41, %42, %43) %45 : int =
//    prim::Constant[value=0]() %46 : Tensor = aten::slice(%44, %45, %13, %10,
//    %11) %47 : int[] = prim::Constant[value=[-1]]() %48 : Tensor =
//    aten::view(%46, %47) %49 : Tensor?[] = prim::ListConstruct(%48) %50 :
//    Tensor = aten::index_put(%state.1, %49, %33, %9)
//    -> (%50)
//  block1():
//    %28 : int[] = prim::ListConstruct(%batch_size.1, %6, %spatial_size_0.1,
//    %spatial_size_1.1) %29 : Tensor = aten::randn(%28, %12, %12, %12, %12) %30
//    : Tensor = aten::slice(%state.1, %13, %13, %10, %11) %35 : Tensor?[] =
//    prim::ListConstruct() %36 : Tensor = aten::expand_as(%29, %30) %52 : int =
//    prim::Constant[value=0]() %53 : int = aten::size(%state.1, %52) %54 : int
//    = prim::Constant[value=4]() %55 : None = prim::Constant() %56 : None =
//    prim::Constant() %57 : None = prim::Constant() %58 : Tensor =
//    aten::arange(%53, %54, %55, %56, %57) %59 : int =
//    prim::Constant[value=0]() %60 : Tensor = aten::slice(%58, %59, %13, %10,
//    %11) %61 : int[] = prim::Constant[value=[-1]]() %62 : Tensor =
//    aten::view(%60, %61) %63 : Tensor?[] = prim::ListConstruct(%62) %64 :
//    Tensor = aten::index_put(%state.1, %63, %36, %9)
//    -> (%64)
void RegisterInplaceNodeInIfBlocks(
    Value* orig_data,
    Value* new_inplace_node,
    Node* block_node,
    Block* outer_block,
    Node* initial_node) {
  if (initial_node->kind() != prim::If)
    return;

  auto next_node = initial_node;
  outer_block->registerOutput(new_inplace_node);
  if (next_node->outputs().size() == 0)
    next_node->addOutput()->copyMetadata(new_inplace_node);

  auto next_block = next_node->owningBlock();
  while (nullptr != next_block->owningNode()) {
    for (auto block_output : next_block->outputs()) {
      if (block_output->debugName() == next_node->output(0)->debugName())
        return;
    }
    next_block->registerOutput(next_node->output(0));
    next_node = next_block->owningNode();
    if (next_node->outputs().size() == 0)
      next_node->addOutput()->copyMetadata(new_inplace_node);
    next_block = next_node->owningBlock();
  }

  orig_data->replaceAllUsesAfterNodeWith(
      next_node->output(0)->node(), next_node->output(0));
}

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
void RegisterInplaceNodeInLoopBlocks(
    Value* orig_data,
    Value* new_inplace_node,
    Node* block_node,
    Block* outer_block,
    Node* next_node) {
  if (next_node->kind() != prim::Loop)
    return;

  outer_block->registerOutput(new_inplace_node);
  std::vector<std::pair<Block*, Node*>> node_list = {
      std::make_pair(outer_block, next_node)};

  next_node->addOutput()->copyMetadata(new_inplace_node);
  auto next_block = next_node->owningBlock();

  while (nullptr != next_block->owningNode()) {
    outer_block = next_block;
    outer_block->registerOutput(next_node->output(0));
    next_node = outer_block->owningNode();
    next_node->addOutput()->copyMetadata(new_inplace_node);
    next_block = next_node->owningBlock();
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
    cur_input->copyMetadata(next_data);
    next_data = cur_input;
    node_list.pop_back();
  }

  // Update inplace node inputs inside the inner most block.
  auto prev_data = block_node->input(0);
  for (auto node : block_node->owningBlock()->nodes()) {
    size_t idx = 0;
    for (auto inputs_ : node->inputs()) {
      if (inputs_ == prev_data) {
        node->replaceInput(idx, next_data);
        idx++;
        break;
      }
    }
  }

  orig_data->replaceAllUsesAfterNodeWith(
      next_node->output(0)->node(), next_node->output(0));
}

// Register inplace op node inputs/outputs through the blocks.
void RegisterInplaceNodeInBlocks(
    Value* orig_data,
    Value* new_inplace_node,
    Node* block_node,
    Block* outer_block,
    Node* next_node) {
  auto cur_node = next_node;

  while (nullptr != cur_node) {
    if (cur_node->kind() != prim::Loop && cur_node->kind() != prim::If)
      return;
    cur_node = cur_node->owningBlock()->owningNode();
  }

  for (auto block_input : outer_block->inputs()) {
    if (block_input->debugName() == orig_data->debugName()) {
      AT_ERROR(
          "More than one inplace mutation in a subblock are not supported.");
    }
  }

  for (auto block_output : outer_block->outputs()) {
    if (block_output->debugName() == new_inplace_node->debugName())
      return;
  }

  // Register inplace node outputs through the blocks.

  RegisterInplaceNodeInLoopBlocks(
      orig_data, new_inplace_node, block_node, outer_block, next_node);

  RegisterInplaceNodeInIfBlocks(
      orig_data, new_inplace_node, block_node, outer_block, next_node);

  matchIfBlockOutputs(orig_data, block_node, outer_block, next_node);
}

} // namespace jit
} // namespace torch
