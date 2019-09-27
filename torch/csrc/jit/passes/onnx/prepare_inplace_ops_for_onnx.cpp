#include <torch/csrc/jit/passes/onnx/prepare_inplace_ops_for_onnx.h>

namespace torch {
namespace jit {

namespace {

Value* CreateSizeOfDim(Value* input, int64_t dim, Node* insertBefore) {
  auto graph = input->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto size = graph->insert(aten::size, {input, dim});
  return size;
}

Value* ConvertSelectToIndex(int64_t index, Node* insertBefore) {
  // Create index tensor based on index attribute of aten::select node.
  auto graph = insertBefore->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto idx_tensor = graph->createNumToTensor(insertConstant(*graph, index));
  graph->insertNode(idx_tensor);
  return graph->insert(aten::unsqueeze, {idx_tensor->output(), 0});
}

Value* ConvertSliceToIndex(Node* slice, Value* size, Node* insertBefore) {
  // Create index tensor based on aten::slice node.
  auto graph = slice->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto start = slice->get(attr::start);
  auto end = slice->get(attr::end);
  auto step = slice->get(attr::step);
  auto index = graph->insert(aten::arange, {size});
  auto sliced_index = graph->insert(aten::slice, {index, {0}, start, end, step});
  return sliced_index;
}

Value* CreateCompleteIndexTensor(Value* size, Node* insertBefore) {
  // Create index tensor of size.
  // The result is torch.tensor([0, 1, 2, ..., size - 1])
  auto graph = size->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto index = graph->insert(aten::arange, {size});
  return index;
}

bool IsSameSource(const Node* n, const Node* m) {
  const auto& source_n = n->sourceRange().source();
  const auto& source_m = m->sourceRange().source(); 
  return ((source_n->text() == source_m->text()) &&
      (source_n->starting_line_no() == source_m->starting_line_no()));
}

std::vector<Node*> FetchSliceAndSelect(const Node* index_put_node) {
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

std::unordered_map<int64_t, Value*> ConvertSliceAndSelectToIndex(
    Graph* graph,
    Node* index_put_node,
    const std::vector<Node*>& slice_and_select_nodes,
    Node* last_node,
    Value* orig_data) {
  std::unordered_map<int64_t, Value*> dim_index_map;
  if (slice_and_select_nodes.size() == 0) {
    return dim_index_map;
  }

  // Loop over fetched slice and select nodes and convert them to index tensors.
  // keep track of which dimension the current slice/select node is applying to.
  int64_t cur_dim = 0;
  // select does not keep dims,
  // this creates offset for latter slice and select nodes.
  int64_t dim_offset = 0;
  for (auto it = slice_and_select_nodes.rbegin(); it != slice_and_select_nodes.rend(); ++it) {
    auto node = *it;
    auto dim = node->get(attr::dim)->toInt() + dim_offset;

    while (cur_dim < dim) {
      // Handle skipped dims, these are created from ..., or tensor indices
      // E.g.: x[torch.tensor([1, 0]), ..., 0] = update, where x has rank 3.
      // Both torch.tensor([1, 0]) and ... are skipped, we only observe aten::select node
      // with dim == 2.
      // Tensor indices will be handled later. Ellipsis(...) are treated as a complete slice over
      // the axes, thus we create index tensors here accordingly.
      if (cur_dim - dim_offset >= index_put_node->input(1)->node()->inputs().size() ||
          index_put_node->input(1)->node()->input(cur_dim - dim_offset)->node()->mustBeNone()) {
        auto size = CreateSizeOfDim(orig_data, cur_dim, index_put_node);
        WithInsertPoint guard(index_put_node);
        auto index_tensor = graph->insert(aten::arange, {size});
        dim_index_map[cur_dim] = index_tensor;
      }
      cur_dim++;
    }

    if (node->kind() == aten::slice) {
      auto size = CreateSizeOfDim(orig_data, dim, index_put_node);
      auto indexTensor = ConvertSliceToIndex(node, size, index_put_node);
      dim_index_map[dim] = indexTensor;
    } else if (node->kind() == aten::select) {
      // aten::select
      auto index = node->get(attr::index)->toInt();
      auto indexTensor = ConvertSelectToIndex(index, index_put_node);
      dim_index_map[dim] = indexTensor;
      dim_offset++;
    } else {
      AT_ERROR("Unexpected node kind ", node->kind().toDisplayString(),
          " Expected aten::slice or aten::select.");
    }

    cur_dim++;
  }

  return dim_index_map;
}

std::vector<Value*> ReshapeToAdvancedIndexingFormat(
    Graph* graph,
    Node* index_put_node,
    std::unordered_map<int64_t, Value*> &dim_index_map) {
  std::vector<Value*> indices;

  auto old_indices_list = index_put_node->input(1)->node()->inputs();
  size_t tensor_ind_count = 0;
  for (size_t i = 0; i < old_indices_list.size(); ++i) {
    if (!old_indices_list[i]->node()->mustBeNone()) {
      tensor_ind_count++;
    }
  }
  size_t total_ind_count = tensor_ind_count + dim_index_map.size();

  bool is_after_tensor_ind = false;
  size_t tensor_ind_offset = 0;
  WithInsertPoint guard(index_put_node);
  for (size_t i = 0; i < total_ind_count; ++i) {
    size_t ind_size = 0;
    Value *index = nullptr;
    if (i < old_indices_list.size() && !old_indices_list[i]->node()->mustBeNone()) {
      if (!is_after_tensor_ind) {
        tensor_ind_offset = i;
        is_after_tensor_ind = true;
      }
      AT_ASSERT(dim_index_map.size() + 1 > tensor_ind_offset);
      ind_size = dim_index_map.size() + 1 - tensor_ind_offset;
      index = old_indices_list[i];
    } else {
      if (is_after_tensor_ind) {
        AT_ASSERT(total_ind_count > i);
        ind_size = total_ind_count - i;
      } else {
        AT_ASSERT(dim_index_map.size() + 1 > i);
        ind_size = dim_index_map.size() + 1 - i;
      }
      AT_ASSERT(dim_index_map.find(i) != dim_index_map.end());
      index = dim_index_map[i];
    }
    std::vector<int64_t> view_shape(ind_size, 1);
    view_shape[0] = -1;
    auto unsqueezed_index = graph->insert(aten::view, {index, view_shape});
    indices.emplace_back(unsqueezed_index);
  }
  return indices;
}

void SquashSliceAndSelect(Node* index_put_node) {
  auto graph = index_put_node->owningGraph();

  std::vector<Node*> slice_and_select_nodes = FetchSliceAndSelect(index_put_node);

  Node* last_node = slice_and_select_nodes.size() > 0 ? slice_and_select_nodes.back() : index_put_node;
  Value* orig_data = last_node->input(0);

  std::unordered_map<int64_t, Value*> dim_index_map =
      ConvertSliceAndSelectToIndex(graph, index_put_node, slice_and_select_nodes, last_node, orig_data);
  std::vector<Value*> indices =
      ReshapeToAdvancedIndexingFormat(graph, index_put_node, dim_index_map);

  WithInsertPoint guard(index_put_node);
  const auto list_indices = graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
                                 ->output();
  auto new_index_put =
      graph->insert(aten::index_put, {orig_data, list_indices, index_put_node->input(2), index_put_node->input(3)});
  new_index_put->copyMetadata(index_put_node->output());
  index_put_node->output()->replaceAllUsesWith(new_index_put);

  orig_data->replaceAllUsesAfterNodeWith(new_index_put->node(), new_index_put);
}

void PrepareCopyForONNX(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      PrepareCopyForONNX(block);
    }

    if (node->kind() == aten::copy_) {
      // aten::copy_ can be viewed as a special case of index_put, where the
      // tensor indices input is empty.
      // Remove aten::copy_, and replace it with index_put.
      // 1. create an empty listConstruct node as indices input for index_put.
      // 2. create index_put node.
      WithInsertPoint guard(node);
      auto graph = node->owningGraph();
      auto dummy_list = graph->insertNode(graph->createList(
                                       OptionalType::ofTensor(), {}))
                                   ->output();
      auto index_put = graph->insert(aten::index_put, {node->input(0), dummy_list, node->input(1), node->input(2)});
      index_put->node()->setSourceRange(node->sourceRange());
      index_put->copyMetadata(node->output());
      node->output()->replaceAllUsesWith(index_put);
    }
  }
}

void PrepareIndexPutForONNX(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      PrepareIndexPutForONNX(block);
    }

    if (node->kind() == aten::index_put || node->kind() == aten::index_put_) {
      SquashSliceAndSelect(node);
    }
  }
}

} // namespace

void PrepareInplaceOpsForONNX(const std::shared_ptr<Graph>& graph) {
  PrepareCopyForONNX(graph->block());
  PrepareIndexPutForONNX(graph->block());
}

} // namespace jit
} // namespace torch
