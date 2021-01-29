#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <limits>

namespace torch {
namespace jit {

namespace {

Value* CreateSizeOfDim(Value* input, int64_t dim, Node* insertBefore) {
  auto graph = input->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto size = graph->insert(aten::size, {input, dim});
  return size;
}

Value* ConvertSelectToIndex(Value* index, Node* insertBefore) {
  // Create index tensor based on index input of aten::select node.
  auto graph = insertBefore->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto idx_tensor = graph->createNumToTensor(index);
  graph->insertNode(idx_tensor);
  return graph->insert(aten::unsqueeze, {idx_tensor->output(), 0});
}

Value* ConvertSliceToIndex(Node* slice, Value* size, Node* insertBefore) {
  // Create index tensor based on aten::slice node.
  const int64_t int_max = std::numeric_limits<int>::max();
  auto graph = slice->owningGraph();
  WithInsertPoint guard(insertBefore);
  TORCH_INTERNAL_ASSERT((slice->inputs()).size() == 5);
  auto start = slice->inputs()[2];
  auto end = slice->inputs()[3];
  auto step = slice->inputs()[4];
  auto index =
      graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
  auto sliced_index =
      graph->insert(aten::slice, {index, {0}, start, end, step});
  return sliced_index;
}

Value* CreateCompleteIndexTensor(Value* size, Node* insertBefore) {
  // Create index tensor of size.
  // The result is torch.tensor([0, 1, 2, ..., size - 1])
  auto graph = size->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto index =
      graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
  return index;
}

bool IsSameSource(const Node* n, const Node* m) {
  const auto& source_n = n->sourceRange().source();
  const auto& source_m = m->sourceRange().source();
  return (
      (source_n->text() == source_m->text()) &&
      (source_n->starting_line_no() == source_m->starting_line_no()));
}

// Trace back all the slice & select nodes associated with the index_put node.
// E.g. The IR for x[1:3, 0] = update
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
//
// We collect %11 and %8, to construct the index tensors.
// The vector slice_and_select_node contains all the associated slice and
// select node, in the reversed order.
std::vector<Node*> FetchSliceAndSelect(const Node* index_put_node) {
  std::vector<Node*> slice_and_select_node;
  auto src_node = index_put_node->input(0)->node();
  while (src_node) {
    if ((src_node->kind() == aten::slice || src_node->kind() == aten::select) &&
        IsSameSource(src_node, index_put_node)) {
      slice_and_select_node.emplace_back(src_node);
      src_node = src_node->input(0)->node();
    } else {
      src_node = nullptr;
    }
  }
  return slice_and_select_node;
}

struct ConvertedIndex {
  ConvertedIndex(Value* index, c10::Symbol orig_node_kind)
      : index(index), orig_node_kind(orig_node_kind) {}

  Value* index = nullptr;
  c10::Symbol orig_node_kind;
};

std::unordered_map<int64_t, ConvertedIndex> MergeSliceAndSelectToIndices(
    Graph* graph,
    Node* index_put_node,
    const std::vector<Node*>& slice_and_select_nodes,
    Value* orig_data) {
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map;

  // Loop over fetched slice and select nodes and convert them to index tensors.
  // keep track of which dimension the current slice/select node is applying to.
  int64_t cur_dim = 0;
  int64_t dim_offset = 0;
  const auto orig_tensor_indices = index_put_node->input(1)->node()->inputs();
  for (auto it = slice_and_select_nodes.rbegin();
       it != slice_and_select_nodes.rend();
       ++it) {
    auto node = *it;
    // select does not keep dims,
    // this creates offset for latter slice and select nodes.
    auto dim = node->get(attr::dim)->toInt();
    if (dim < 0) {
      auto input_type = orig_data->type()->expect<TensorType>();
      if (input_type->dim().has_value()) {
        auto rank = input_type->dim().value();
        // Rank of original tensor to index on.
        // Minus the offset created by select operators.
        dim = dim + rank - dim_offset;
      } else {
        std::cerr
            << "Error: ONNX Remove Inplace Ops - Cannot export ellipsis indexing for input "
            << "of unknown rank.";
      }
    }
    dim = dim + dim_offset;

    while (cur_dim < dim) {
      // Handle skipped dims, these are created from ..., or tensor indices
      // E.g.: x[torch.tensor([1, 0]), ..., 0] = update, where x has rank 3.
      // Both torch.tensor([1, 0]) and ... are skipped, we only observe
      // aten::select node with dim == 2. Tensor indices will be handled later.
      // Ellipsis(...) are treated as a complete slice over the axes, thus we
      // create index tensors here accordingly.
      if (cur_dim - dim_offset >= orig_tensor_indices.size() ||
          index_put_node->input(1)
              ->node()
              ->input(cur_dim - dim_offset)
              ->node()
              ->mustBeNone()) {
        auto size = CreateSizeOfDim(orig_data, cur_dim, index_put_node);
        WithInsertPoint guard(index_put_node);
        auto index_tensor = graph->insert(
            aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
        dim_index_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cur_dim),
            std::forward_as_tuple(index_tensor, aten::slice));
      } else if (cur_dim - dim_offset < orig_tensor_indices.size()) {
        dim_index_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cur_dim),
            std::forward_as_tuple(
                orig_tensor_indices[cur_dim - dim_offset], aten::index));
      }
      cur_dim++;
    }

    AT_ASSERT(cur_dim == dim);
    if (node->kind() == aten::slice) {
      auto size = CreateSizeOfDim(orig_data, dim, index_put_node);
      auto index_tensor = ConvertSliceToIndex(node, size, index_put_node);
      dim_index_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(dim),
          std::forward_as_tuple(index_tensor, aten::slice));
    } else if (node->kind() == aten::select) {
      auto index_tensor = ConvertSelectToIndex(node->input(2), index_put_node);
      dim_index_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(dim),
          std::forward_as_tuple(index_tensor, aten::select));
      dim_offset++;
    } else {
      AT_ERROR(
          "Unexpected node kind ",
          node->kind().toDisplayString(),
          " Expected aten::slice or aten::select.");
    }

    cur_dim++;
  }

  while (cur_dim - dim_offset < orig_tensor_indices.size()) {
    dim_index_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(cur_dim),
        std::forward_as_tuple(
            orig_tensor_indices[cur_dim - dim_offset], aten::index));
    cur_dim++;
  }

  // Each dimension should have its associated index tensor.
  AT_ASSERT(dim_index_map.size() == cur_dim);
  return dim_index_map;
}

// Convert slice/select operators to tensor indices.
// Reshape the tensor indices according to their axis.
// E.g.                 x[1:3, 0, ind1, ind2] = y
//  slice index shape:   [2,   1, 1 ]
//  select index shape:  [     1, 1 ]
//  ind1 shape:          [        _ ]
//  ind2 shape:          [        _ ]
// where _ is the original size of ind1 and ind2.
// ind1 and ind2 are both 1-d tensors since currently we only supports 1-d
// tensor indices.
std::vector<Value*> ReshapeToAdvancedIndexingFormat(
    Graph* graph,
    Node* index_put_node,
    std::unordered_map<int64_t, ConvertedIndex>& dim_index_map) {
  std::vector<Value*> indices;

  size_t min_index_dim = dim_index_map.size();
  size_t max_index_dim = 0;
  size_t tensor_ind_count = 0;
  for (size_t i = 0; i < dim_index_map.size(); ++i) {
    auto index_i = dim_index_map.find(i);
    AT_ASSERT(index_i != dim_index_map.end());
    if (index_i->second.orig_node_kind == aten::index) {
      if (i < min_index_dim)
        min_index_dim = i;
      if (i > max_index_dim)
        max_index_dim = i;
      tensor_ind_count++;
    }
  }

  if (((max_index_dim - min_index_dim + 1) != tensor_ind_count) &&
      tensor_ind_count != 0) {
    AT_ERROR(
        "Only consecutive 1-d tensor indices are supported in exporting aten::index_put to ONNX.");
  }

  size_t tensor_ind_offset = tensor_ind_count == 0 ? 0 : tensor_ind_count - 1;
  WithInsertPoint guard(index_put_node);
  for (size_t i = 0; i < dim_index_map.size(); ++i) {
    size_t ind_size = 0;
    auto index_i = dim_index_map.find(i);
    AT_ASSERT(index_i != dim_index_map.end());
    Value* index = index_i->second.index;
    switch (index_i->second.orig_node_kind) {
      case aten::select:
      case aten::slice: {
        if (i < min_index_dim) {
          ind_size = dim_index_map.size() - tensor_ind_offset - i;
        } else {
          ind_size = dim_index_map.size() - i;
        }
        break;
      }

      case aten::index: {
        ind_size = dim_index_map.size() - tensor_ind_offset - min_index_dim;
        break;
      }
      default:
        AT_ERROR("Unexpected node kind ", index_i->second.orig_node_kind);
    }

    std::vector<int64_t> view_shape(ind_size, 1);
    view_shape[0] = -1;
    auto unsqueezed_index = graph->insert(aten::view, {index, view_shape});
    indices.emplace_back(unsqueezed_index);
  }

  return indices;
}

void addDummyCloneBlockOutput(Block* b, Value* orig_data) {
  auto graph = b->owningGraph();
  auto newNode = graph->create(aten::clone, /*num_outputs =*/1);
  newNode->addInput(orig_data);

  auto* noneNode = graph->create(prim::Constant);
  noneNode->output()->setType(NoneType::get());
  newNode->addInput(noneNode->output());
  newNode->output()->setType(orig_data->type());

  newNode->insertBefore(b->return_node());
  noneNode->insertBefore(newNode);
  b->registerOutput(newNode->output());
}

// Check If then/else blocks to match the number of outputs.
// If the number of block outputs do not match, insert a dummy
// constant of corresponding shape and type.
Value* MatchIfBlocksOutputForValue(
    Value* orig_data,
    Block* outer_block,
    Value* origOutput) {
  if (outer_block->owningNode()->kind() != prim::If)
    return nullptr;

  size_t output_size = outer_block->outputs().size();

  for (size_t i = 0; i < output_size - 1; i++) {
    if (outer_block->outputs().at(i)->debugNameBase() ==
        origOutput->debugNameBase()) {
      outer_block->owningNode()
          ->outputs()
          .at(output_size - 1)
          ->replaceAllUsesWith(outer_block->owningNode()->outputs().at(i));
      outer_block->owningNode()->eraseOutput(output_size - 1);
      outer_block->replaceOutput(i, outer_block->outputs().at(output_size - 1));
      outer_block->eraseOutput(output_size - 1);
      return outer_block->owningNode()->outputs().at(i);
    }
  }

  for (Block* b : outer_block->owningNode()->blocks()) {
    if (b->outputs().size() < output_size) {
      addDummyCloneBlockOutput(b, orig_data);
      b->outputs()
          .at(b->outputs().size() - 1)
          ->copyMetadata(outer_block->outputs().at(output_size - 1));
      return outer_block->owningNode()->outputs().at(output_size - 1);
    }
  }
  return outer_block->owningNode()->outputs().at(output_size - 1);
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
    Value* value,
    Value* new_inplace_node,
    Block* outer_block,
    Node* initial_node,
    const std::string& output_name) {
  if (initial_node->kind() != prim::If)
    return;

  for (auto block_output : outer_block->outputs()) {
    if (block_output->debugName() == new_inplace_node->debugName())
      return;
  }

  auto next_node = initial_node;

  new_inplace_node->setDebugName("_output_" + output_name);
  outer_block->registerOutput(new_inplace_node);
  // Block has a new output. Add the output for the prim::If node.
  if (next_node->outputs().size() < outer_block->outputs().size())
    next_node->addOutput()->copyMetadata(new_inplace_node);

  auto next_block = next_node->owningBlock();
  while (nullptr != next_block->owningNode()) {
    for (auto block_output : next_block->outputs()) {
      if (block_output->debugName() == block_output->debugName())
        return;
    }
    next_block->registerOutput(next_node->output(0));
    next_node = next_block->owningNode();
    // Block has a new output. Add the output for the prim::If node.
    if (next_node->outputs().size() < next_block->outputs().size())
      next_node->addOutput()->setType(new_inplace_node->type());
    next_block = next_node->owningBlock();
  }

  value->replaceAllUsesAfterNodeWith(
      next_node->output(0)->node(),
      next_node->outputs().at(next_node->outputs().size() - 1));
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

  next_node->addOutput()->setType(new_inplace_node->type());
  auto next_block = next_node->owningBlock();

  while (nullptr != next_block->owningNode()) {
    outer_block = next_block;
    outer_block->registerOutput(next_node->output(0));
    next_node = outer_block->owningNode();
    next_node->addOutput()->setType(new_inplace_node->type());
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
    cur_input->setType(next_data->type());
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
      next_node->output(0)->node(),
      next_node->outputs().at(next_node->outputs().size() - 1));
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
    if (block_input->debugName() ==
        orig_data->debugName()) { // TODO: enable more than one mutation.
      AT_ERROR(
          "More than one inplace mutation per object in a subblock are not supported.");
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
      orig_data,
      new_inplace_node,
      outer_block,
      next_node,
      orig_data->debugName());

  MatchIfBlocksOutputForValue(orig_data, outer_block, new_inplace_node);
}

// Trace back all the slice & select nodes associated with the index_put node,
// and convert them to associated indices.
// E.g. The IR for x[1:3, 0] = update
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
// The aten::index_put node alone does not contain any indices (%13 : Tensor?[]
// = prim::ListConstruct()).
//    ...
//    # Below constructs index from slice node.
//    %23 : Long() = aten::size(%0, %4)
//    %28 : Tensor = aten::arange(%23, %24, %25, %26, %27)
//    %33 : Tensor = aten::slice(%28, %4, %5, %6, %7)
//    %39 : int[] = prim::Constant[value=[-1, 1]]()
//    %40 : Tensor = aten::view(%33, %39)
//    ...
//    # Below constructs index from select node.
//    %36 : int = prim::Constant[value=0]()
//    %37 : Tensor = aten::unsqueeze(%10, %36)
//    %42 : int[] = prim::Constant[value=[-1]]()
//    %43 : Tensor = aten::view(%37, %42)
//    ...
//    # Adding the above two indices to index_put
//    %44 : Tensor?[] = prim::ListConstruct(%40, %43)
//    %45 : Float(2, 5) = aten::index_put(%0, %44, %14, %15)
void SquashSliceAndSelect(Node* index_put_node) {
  auto graph = index_put_node->owningGraph();

  // Find slice and select operators that are associated with this index
  // operator. E.g. x[1:3, 0] = y will generate one slice operator(1:3) and one
  // select operator(0).
  std::vector<Node*> slice_and_select_nodes =
      FetchSliceAndSelect(index_put_node);

  Node* last_node = slice_and_select_nodes.size() > 0
      ? slice_and_select_nodes.back()
      : index_put_node;
  Value* orig_data = last_node->input(0);

  // Convert fetched slice/select operators into tensor indices.
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map =
      MergeSliceAndSelectToIndices(
          graph, index_put_node, slice_and_select_nodes, orig_data);
  std::vector<Value*> indices =
      ReshapeToAdvancedIndexingFormat(graph, index_put_node, dim_index_map);

  // Create new aten::index_put operator.
  WithInsertPoint guard(index_put_node);
  const auto list_indices =
      graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
          ->output();
  auto new_index_put = graph->insert(
      aten::index_put,
      {orig_data,
       list_indices,
       index_put_node->input(2),
       index_put_node->input(3)});
  new_index_put->copyMetadata(index_put_node->output());
  index_put_node->output()->replaceAllUsesWith(new_index_put);

  auto block_node = new_index_put->node();
  auto outer_block = block_node->owningBlock();
  auto next_node = outer_block->owningNode();
  if (nullptr == next_node) {
    orig_data->replaceAllUsesAfterNodeWith(
        new_index_put->node(), new_index_put);
    return;
  }

  RegisterInplaceNodeInBlocks(
      orig_data, new_index_put, block_node, outer_block, next_node);
}

void PrepareIndexPutForONNX(Node* node) {
  if (node->kind() == aten::index_put || node->kind() == aten::index_put_) {
    SquashSliceAndSelect(node);
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
        aten::index_put,
        {node->input(0), dummy_list, expanded_value, node->input(2)});
    index_put->node()->setSourceRange(node->sourceRange());
    index_put->copyMetadata(node->output());
    node->output()->replaceAllUsesWith(index_put);

    PrepareIndexPutForONNX(index_put->node());
  }
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
    getitem_node->output()->copyMetadata(n->output());
    getitem_node->insertBefore(n);
    n->output()->replaceAllUsesWith(getitem_node->output());

    n->output()->copyMetadata(n->inputs()[0]);
    n->inputs()[0]->replaceAllUsesAfterNodeWith(n, n->output());
  }
}

static void PrepareListDeleteForONNX(Node* n) {
  if (n->kind() == aten::Delete) {
    n->addOutput();
    n->output()->setType(n->input(0)->type());
    n->input(0)->replaceAllUsesAfterNodeWith(n, n->output());
  }
}

static void PrepareListAppendAndInsertForONNX(Node* n) {
  if (n->kind() == aten::insert || n->kind() == aten::append) {
    if (n->outputs().size() == 0) {
      n->addOutput();
      n->output()->copyMetadata(n->inputs()[0]);
    }
    n->inputs()[0]->replaceAllUsesAfterNodeWith(n, n->output());
  }
}

static void PrepareInplaceOpsForONNX(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareInplaceOpsForONNX(child_block);
    }

    switch (it->kind()) {
      case aten::copy_: {
        PrepareCopyForONNX(*it);
        break;
      }
      case aten::index_put:
      case aten::index_put_: {
        PrepareIndexPutForONNX(*it);
        break;
      }
      case aten::pop: {
        PrepareListPopForONNX(*it);
        break;
      }
      case aten::insert:
      case aten::append: {
        PrepareListAppendAndInsertForONNX(*it);
        break;
      }
      case aten::Delete: {
        PrepareListDeleteForONNX(*it);
        break;
      }
    }
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

        std::cerr
            << "Warning: ONNX Preprocess - Removing mutation on block inputs. "
            << "This changes graph semantics." << std::endl;

        Node* newNode = nullptr;
        if (input->type()->kind() == TypeKind::ListType) {
          // Create an aten::list to clone the list in graph inputs
          newNode = node->owningGraph()->create(aten::list, 1);
          newNode->output()->setType(input->type());
          newNode->addInput(input);
          b->prependNode(newNode);
        } else {
          // Create an aten::clone to clone the tensor in graph inputs
          newNode = node->owningGraph()->create(aten::clone, 1);
          newNode->output()->setType(input->type());
          newNode->addInput(input);

          auto* noneNode = node->owningGraph()->create(prim::Constant);
          noneNode->output()->setType(NoneType::get());
          newNode->addInput(noneNode->output());
          b->prependNode(newNode);
          noneNode->insertBefore(newNode);
        }
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
      return moduleNames;
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
    if ((input->debugName() == name))
      return input;
  }
  throw std::runtime_error(
      "Attribute is not part of model parameters. Cannot handle SetAttr and GetAttr nodes for : " +
      name);
}

Node* insertCloneBeforeNode(
    const std::shared_ptr<Graph>& graph,
    Value* orig_data,
    Node* node) {
  Node* newNode = nullptr;
  if (orig_data->type()->kind() == TypeKind::ListType) {
    // Create an aten::list to clone the list in graph inputs
    newNode = graph->create(aten::list, /*num_outputs =*/1);
    newNode->addInput(orig_data);
    newNode->output()->setType(orig_data->type());
    newNode->insertBefore(node);
  } else if (orig_data->type()->kind() == TypeKind::TensorType) {
    auto* noneNode = graph->create(prim::Constant);
    noneNode->output()->setType(NoneType::get());
    newNode = graph->create(aten::clone, /*num_outputs =*/1);
    newNode->addInput(orig_data);

    newNode->addInput(noneNode->output());
    newNode->output()->setType(orig_data->type());
    newNode->insertBefore(node);
    noneNode->insertBefore(newNode);
  } // TODO: Handle float/int attributes
  return newNode;
}

Value* registerSetAttrInBlocks(
    const std::shared_ptr<Graph>& graph,
    Block* block,
    Value* newValue,
    Value* origValue,
    const std::string& output_name) {
  auto cloneNode = insertCloneBeforeNode(graph, newValue, block->return_node());
  auto next_node = block->owningNode();

  RegisterInplaceNodeInIfBlocks(
      origValue, cloneNode->output(), block, next_node, output_name);

  return MatchIfBlocksOutputForValue(origValue, block, cloneNode->output());
}

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

  auto moduleNames =
      findSubModuleAttr(n->inputs().at(0), name, attrModule, graph);
  if (!attrModule.hasattr(name))
    return;
  auto attr = attrModule.attr(name);
  Value* paramConst = nullptr;

  std::string fullName("");
  for (auto& name : moduleNames) {
    fullName += name + '.';
  }
  fullName += name;

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
  }

  if (n->kind() == prim::SetAttr) { // Handle SetAttr node
    if (attrModule.hasattr(name)) {
      // If inside a block, keep the output value to register in block
      // output.
      auto block_ = n->owningBlock();
      if (block_->owningNode() &&
          block_->owningNode()->kind() == prim::If) { // TODO: Add loop

        auto attrValue = (setAttrValues.find(fullName) != setAttrValues.end())
            ? setAttrValues[fullName]
            : allAttrValues[fullName];

        auto blockOutput = registerSetAttrInBlocks(
            graph, block_, n->inputs().at(1), attrValue, fullName);
        nextSetAttrValues[fullName] = blockOutput;
      }
      // SetAttr writes a value to an attr. Keep this
      // in the setAttrValues map.
      setAttrValues[fullName] = n->inputs().at(1);
    }
  } else if (n->kind() == prim::GetAttr) { // Handle GetAttr node
    if (setAttrValues.find(fullName) != setAttrValues.end()) {
      // Attr has been set earlier in the graph.
      // Read its value from setAttrValues map.
      auto set_attr_node_input = setAttrValues[fullName];
      // Clone SetAttr input
      auto cloneNode = insertCloneBeforeNode(graph, set_attr_node_input, n);
      n->output()->replaceAllUsesAfterNodeWith(n, cloneNode->output());
    } else if (allAttrValues.find(fullName) != allAttrValues.end()) {
      // Attr has not been set earlier in the graph. Replace it with the
      // graph parameter if exists.
      n->output()->replaceAllUsesWith(allAttrValues[fullName]);
      n->removeAllInputs();
    }
  }
}

std::unordered_map<std::string, Value*> registerInplaceOpAsBlockOutputs(
    MutationRemover& mr,
    Block* block,
    const std::shared_ptr<Graph>& graph,
    const Module& module_,
    std::unordered_map<std::string, Value*>& allAttrValues,
    std::unordered_map<std::string, Value*>& setAttrValues) {
  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);

  std::unordered_map<std::string, Value*> nextSetAttrValues = {};

  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    if (n->kind() == prim::GetAttr || n->kind() == prim::SetAttr) {
      trackAndRegisterAttributesInBlocks(
          n, graph, module_, allAttrValues, setAttrValues, nextSetAttrValues);
    } else if (
        mr.inplaceOpVariant(n) &&
        n->kind() != aten::copy_) { // TODO: create a list of all excluded ops
      auto orig_data = n->inputs().at(0);
      auto block_ = n->owningBlock();
      if (block_->owningNode())
        RegisterInplaceNodeInBlocks(
            orig_data, n->output(), n, block_, block_->owningNode());

    } else { // for prim::If and prim::Loop nodes with blocks.
      for (Block* sub_block : n->blocks()) {
        std::unordered_map<std::string, Value*> map_ =
            registerInplaceOpAsBlockOutputs(
                mr, sub_block, graph, module_, allAttrValues, setAttrValues);
        std::unordered_map<std::string, Value*>::iterator mapIt;
        for (mapIt = map_.begin(); mapIt != map_.end(); mapIt++) {
          setAttrValues[mapIt->first] = mapIt->second;
        }
      }
    }
  }
  return nextSetAttrValues;
}

void RegisterInplaceOpAsBlockOutputs(
    MutationRemover& mr,
    Module* module,
    const std::shared_ptr<Graph>& graph) {
  // A map of names and values of referenced attributes, to avoid duplicates.
  std::unordered_map<std::string, Value*> allAttrValues = {};
  // A map of names and values of set attributes, to track mutations.
  std::unordered_map<std::string, Value*> setAttrValues = {};

  Module moduleClone = module->clone(true);
  registerInplaceOpAsBlockOutputs(
      mr, graph->block(), graph, moduleClone, allAttrValues, setAttrValues);
  EliminateDeadCode(graph->block());
}

} // namespace

void PrepareInplaceOpsForONNX(const std::shared_ptr<Graph>& graph) {
  PrepareInplaceOpsForONNX(graph->block());
}

void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model = nullptr) {
  MutationRemover mr(graph);
  PrepareForRemoveMutations(mr, graph->block());
  RemoveTensorMutation(graph);
  RemoveListMutation(graph);
  if (model)
    RegisterInplaceOpAsBlockOutputs(mr, model, graph);
}

} // namespace jit
} // namespace torch
