#pragma once

#include <torch/csrc/jit/ir/ir.h>

// Functions used by both encapsulation and conversion.

namespace torch::jit {

struct IndexingPatternFinder {
 public:
  static std::vector<Node*> FetchSliceAndSelect(const Node* node);

 private:
  static bool IsSameSource(const Node* n, const Node* m);
};

} // namespace torch::jit
