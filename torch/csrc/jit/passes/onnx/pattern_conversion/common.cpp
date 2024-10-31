#include <torch/csrc/jit/passes/onnx/pattern_conversion/common.h>

namespace torch {
namespace jit {

bool IndexingPatternFinder::IsSameSource(const Node* n, const Node* m) {
  const auto source_n = n->sourceRange().source();
  const auto source_m = m->sourceRange().source();
  return (
      (source_n->text_str() == source_m->text_str()) &&
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
std::vector<Node*> IndexingPatternFinder::FetchSliceAndSelect(
    const Node* node) {
  std::vector<Node*> slice_and_select_node;
  auto src_node = node->input(0)->node();
  while (src_node) {
    if ((src_node->kind() == aten::slice || src_node->kind() == aten::select) &&
        IsSameSource(src_node, node)) {
      slice_and_select_node.emplace_back(src_node);
      src_node = src_node->input(0)->node();
    } else {
      src_node = nullptr;
    }
  }
  return slice_and_select_node;
}

} // namespace jit
} // namespace torch
