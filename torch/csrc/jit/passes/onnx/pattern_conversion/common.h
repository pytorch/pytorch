#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct IndexPutPatternFinder {
 public:
  static std::vector<Node*> FetchSliceAndSelect(const Node* index_put_node);

 private:
  static bool IsSameSource(const Node* n, const Node* m);
};

} // namespace jit
} // namespace torch
