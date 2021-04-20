#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

// Refine from IntType -> int value, or ListType -> len of list
// TODO: vector may be faster
using ListRefinement = std::unordered_map<Value*, int64_t>;

TORCH_API ListRefinement
intersectRefinements(const ListRefinement& ref1, const ListRefinement& ref2);

TORCH_API ListRefinement
unionRefinements(const ListRefinement& ref1, const ListRefinement& ref2);

struct BoolRefinements {
  BoolRefinements(ListRefinement true_refine, ListRefinement false_refine)
      : true_refine_(std::move(true_refine)),
        false_refine_(std::move(false_refine)){};
  BoolRefinements() = default; // empty

  static BoolRefinements FalseRefinements(ListRefinement false_refine) {
    return BoolRefinements({}, std::move(false_refine));
  }

  static BoolRefinements TrueRefinements(ListRefinement true_refine) {
    return BoolRefinements(std::move(true_refine), {});
  }

  BoolRefinements intersectBoolRefinements(BoolRefinements& other) {
    return BoolRefinements(
        intersectRefinements(true_refine_, other.true_refine()),
        intersectRefinements(false_refine_, other.false_refine()));
  }

  ListRefinement& true_refine() {
    return true_refine_;
  }

  ListRefinement& false_refine() {
    return false_refine_;
  }

 private:
  ListRefinement true_refine_;
  ListRefinement false_refine_;
};

TORCH_API void joinIfRefinements(
    Node* if_node,
    std::unordered_set<Block*>& throwing_blocks,
    ListRefinement& curr_block_refinements,
    ListRefinement& true_block_refinements,
    ListRefinement& false_block_refinements,
    std::unordered_map<Value*, BoolRefinements>& info);

} // namespace jit
} // namespace torch