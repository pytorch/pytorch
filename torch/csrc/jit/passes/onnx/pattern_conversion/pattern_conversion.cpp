#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_conversion.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>

namespace torch {
namespace jit {

namespace {

size_t CountNodesBefore(const Node* n) {
  auto graph = n->owningGraph();
  size_t i = 0;
  for (auto node : graph->nodes()) {
    if (node == n) {
      break;
    }
    i++;
  }
  return i;
}

size_t CountNodesAfterExcludingSubblock(const Node* n) {
  auto graph = n->owningGraph();
  size_t i = 0;
  auto node = n->next();
  while (node != graph->return_node()) {
    if (std::none_of(
            n->blocks().begin(), n->blocks().end(), [&](const Block* b) {
              return b == node->owningBlock();
            })) {
      i++;
    }
    node = node->next();
  }
  return i;
}

} // namespace

// Converting inplace index_put to ONNX
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
  return graph->insert(aten::unsqueeze, {index, 0});
}

Value* ConvertSliceToIndex(Node* slice, Value* size, Node* insertBefore) {
  // Create index tensor based on aten::slice node.
  auto graph = slice->owningGraph();
  WithInsertPoint guard(insertBefore);
  TORCH_INTERNAL_ASSERT((slice->inputs()).size() == 5);
  auto start = slice->inputs()[2];
  auto end = slice->inputs()[3];
  auto step = slice->inputs()[4];
  auto index =
      graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
  auto sliced_index_n = graph->create(
      aten::slice,
      {index,
       graph->insertConstant(
           scalar_to_tensor(at::Scalar(0)), c10::nullopt, slice->scope()),
       start,
       end,
       step});

  auto sliced_index = sliced_index_n->insertBefore(insertBefore)->output();
  return sliced_index;
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
    Value* orig_data,
    const std::unordered_map<Value*, Value*>& env) {
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
    // NOTE: Cannot rely on get(attr::dim), because op no longer match schema.
    int64_t dim = node->inputs().at(1)->node()->t(attr::value).item().toLong();

    if (dim < 0) {
      auto input_type = env.at(orig_data)->type()->expect<TensorType>();
      if (input_type->dim().has_value()) {
        auto rank = static_cast<int64_t>(input_type->dim().value());
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
      if (cur_dim - dim_offset >= (int64_t)orig_tensor_indices.size() ||
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
      } else if (cur_dim - dim_offset < (int64_t)orig_tensor_indices.size()) {
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

  while (cur_dim - dim_offset < (int64_t)orig_tensor_indices.size()) {
    dim_index_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(cur_dim),
        std::forward_as_tuple(
            orig_tensor_indices[cur_dim - dim_offset], aten::index));
    cur_dim++;
  }

  // Each dimension should have its associated index tensor.
  AT_ASSERT((int64_t)dim_index_map.size() == cur_dim);
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
std::vector<Value*> ConvertIndexPutToONNX(
    Block* new_block,
    Node* old_node,
    std::unordered_map<Value*, Value*>& env) {
  if (old_node->kind() != Symbol::fromQualString("onnx::Placeholder") ||
      old_node->s(attr::name) != "index_put") {
    return {};
  }

  TORCH_INTERNAL_ASSERT(old_node->blocks().size() == 1);
  auto old_graph = old_node->owningGraph();
  auto subblock = old_node->blocks()[0];
  auto index_put_node = subblock->nodes().back()->prev();

  // Open up SquashSliceAndSelect
  // 1. get all slice and select nodes, as before.
  // 2. MergeSliceAndSelectToIndices
  // 3. ReshapeToAdvancedIndexingFormat
  // 4. Create new index_put
  // 5. For nodes created in 2-4, run onnx conversion.
  // Node modifications are only allowed within the subblock.

  // 1.
  // Find slice and select operators that are associated with this index
  // operator. E.g. x[1:3, 0] = y will generate one slice operator(1:3) and one
  // select operator(0).
  std::vector<Node*> slice_and_select_nodes =
      IndexPutPatternFinder::FetchSliceAndSelect(index_put_node);
  Node* last_node = slice_and_select_nodes.size() > 0
      ? slice_and_select_nodes.back()
      : index_put_node;
  Value* orig_data = last_node->input(0);

  // 2.
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map =
      MergeSliceAndSelectToIndices(
          old_graph, index_put_node, slice_and_select_nodes, orig_data, env);

  // 3.
  std::vector<Value*> indices =
      ReshapeToAdvancedIndexingFormat(old_graph, index_put_node, dim_index_map);

  // 4.
  const auto list_indices =
      old_graph->createList(OptionalType::ofTensor(), indices)
          ->insertBefore(index_put_node)
          ->output();
  auto new_index_put_node = old_graph->create(
      aten::index_put,
      {orig_data,
       list_indices,
       index_put_node->input(2),
       index_put_node->input(3)});
  new_index_put_node->insertBefore(index_put_node);

  auto new_index_put = new_index_put_node->output();

  new_index_put->copyMetadata(index_put_node->output());
  index_put_node->output()->replaceAllUsesWith(new_index_put);

  orig_data->replaceAllUsesAfterNodeWith(new_index_put->node(), new_index_put);

  EraseNumberTypesOnBlock(subblock);
  EliminateDeadCode(
      subblock,
      true,
      DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);

  // 5. Convert all these things to onnx.
  for (auto at_n : subblock->nodes()) {
    if (at_n == subblock->param_node() || at_n == subblock->return_node()) {
      continue;
    }

    NodeToONNX(at_n, new_block, torch::onnx::OperatorExportTypes::ONNX, env);
  }

  // 6. figure out onnx outputs.
  std::vector<Value*> outs;
  for (auto o : subblock->return_node()->inputs()) {
    outs.emplace_back(env[o]);
  }
  return outs;
}

} // namespace

std::vector<Value*> ConvertPatternFromSubblock(
    Block* new_block,
    Node* old_node,
    std::unordered_map<Value*, Value*>& env) {
  // Assert that the number of nodes in original torch graph is unchanged
  // before and after converting patterns from subblock.
  size_t n_count_before = CountNodesBefore(old_node);
  size_t n_count_after = CountNodesAfterExcludingSubblock(old_node);

  std::vector<Value*> res;

  if (old_node->kind() != Symbol::fromQualString("onnx::Placeholder")) {
    return res;
  }

  if (old_node->s(attr::name) == "index_put") {
    res = ConvertIndexPutToONNX(new_block, old_node, env);
  }

  TORCH_INTERNAL_ASSERT(n_count_before == CountNodesBefore(old_node));
  TORCH_INTERNAL_ASSERT(
      n_count_after == CountNodesAfterExcludingSubblock(old_node));

  return res;
}

} // namespace jit
} // namespace torch
