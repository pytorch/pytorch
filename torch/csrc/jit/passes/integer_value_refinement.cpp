#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/integer_value_refinement.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>
#include <torch/csrc/utils/memory.h>
#include "jit/ir/ir.h"

namespace torch {
namespace jit {

using IntegerRefinement = std::unordered_map<Value*, int64_t>;

// see [value refinement algorithm]

struct IntegerValueRefiner {
  IntegerValueRefiner(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    if (!checkForPossibleRefinements(graph_->block())) {
      return false;
    }
    ListRefinement refinements;
    RefineListLens(graph_->block(), refinements);
    return changed_;
  }

  bool checkForPossibleRefinements(Block* b) {
    for (Node* n : b->nodes()) {
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        for (size_t const_index : {0, 1}) {
          auto non_const_index = 1 - const_index;
          if (n->inputs().at(const_index)->node()->kind() == prim::Constant &&
              n->inputs().at(non_const_index)->uses().size() > 1) {
            return true;
          }
        }
      }
      for (Block* block : n->blocks()) {
        if (checkForPossibleRefinements(block)) {
          return true;
        }
      }
    }
    return false;
  }

  ListRefinement RefineListLens(Block* b, ListRefinement block_refinements) {
    active_refinements_.push_back(&block_refinements);
    for (Node* n : b->nodes()) {
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        for (size_t const_index : {0, 1}) {
          if (auto ival = constant_as<int64_t>(n->inputs().at(const_index))) {
            IntegerRefinement refine;
            refine[n->inputs().at(1 - const_index)] = *ival;
            info_[n->output()] = n->kind() == aten::eq
                ? BoolRefinements::TrueRefinements(std::move(refine))
                : BoolRefinements::FalseRefinements(std::move(refine));
          }
        }
      }
      if (n->kind() == prim::RaiseException) {
        throwing_blocks_.insert(b);
      }

      for (size_t input = 0; input < n->inputs().size(); ++input) {
        Value* input_v = n->inputs().at(input);
        if (!input_v->type()->cast<IntType>()) {
          continue;
        }

        if (auto refine = tryFindRefinement(input_v)) {
          WithInsertPoint guard(n);
          auto refine_constant =
              graph_->insertConstant(static_cast<int64_t>(*refine));
          n->replaceInputWith(input_v, refine_constant);
          changed_ = true;
        }
      }

      if (n->kind() == prim::If) {
        IfView if_n(n);
        bool has_cond_ref = info_.count(if_n.cond()) != 0;
        ListRefinement empty;
        auto true_block_refinements = RefineListLens(
            if_n.thenBlock(),
            has_cond_ref ? info_[if_n.cond()].true_refine() : empty);
        auto false_block_refinements = RefineListLens(
            if_n.elseBlock(),
            has_cond_ref ? info_[if_n.cond()].false_refine() : empty);

        // we are looking for cases where we can replace
        // both block outputs with the same value, which opens up
        // further optimization opportunities
        // The pass will already handle if both are refined to the same
        // constant. Here, we add a case case where one block output
        // is refined to a constant in the other block, and where the existing
        // block output in the other block is the same constant
        // x = 1
        // if y == 1:
        //    return x
        // else:
        //    return y
        // can always safely be replaced with `y`
        // this is an important case for symbolic shape analysis

        for (size_t block_index : {0, 1}) {
          Block* if_block = if_n.node()->blocks().at(block_index);
          Block* other_if_block = if_n.node()->blocks().at(1 - block_index);
          for (size_t i = 0; i < n->outputs().size(); ++i) {
            Value* block_output = if_block->outputs().at(i);
            if (!block_output->type()->cast<IntType>()) {
              continue;
            }
            // Value must be in scope for both blocks
            if (!if_n.node()->isDominatedBy(block_output->node())) {
              continue;
            }
            // one constant value one not
            auto other_const_value =
                constant_as<int64_t>(other_if_block->outputs().at(i));
            if (!other_const_value ||
                block_output->node()->kind() == prim::Constant) {
              continue;
            }
            const auto& other_block_refinements = block_index == 0
                ? false_block_refinements
                : true_block_refinements;
            c10::optional<int64_t> maybe_refine =
                tryFindRefinement(block_output);
            if (!maybe_refine && other_block_refinements.count(block_output)) {
              maybe_refine = other_block_refinements.at(block_output);
            }
            if (maybe_refine && *maybe_refine == *other_const_value) {
              if_n.outputs().at(i)->replaceAllUsesWith(block_output);
              changed_ = true;
            }
          }
        }

        joinIfRefinements(
            n,
            throwing_blocks_,
            block_refinements,
            true_block_refinements,
            false_block_refinements,
            info_);
      }
    }

    // this is useful for things like if block outputs,
    // where the output value node may not be defined in this block
    // but we have refined its use as a block output
    for (size_t i = 0; i < b->outputs().size(); ++i) {
      Value* input_v = b->outputs().at(i);
      if (!input_v->type()->cast<IntType>()) {
        continue;
      }

      if (auto refine = tryFindRefinement(input_v)) {
        WithInsertPoint guard(b);
        auto refine_constant =
            graph_->insertConstant(static_cast<int64_t>(*refine));
        b->replaceOutput(i, refine_constant);
        changed_ = true;
      }
    }

    active_refinements_.pop_back();
    return block_refinements;
  };

  c10::optional<int64_t> tryFindRefinement(Value* v) {
    for (const auto& ref : active_refinements_) {
      auto maybe_refinement = ref->find(v);
      if (maybe_refinement != ref->end()) {
        return maybe_refinement->second;
      }
    }
    return c10::nullopt;
  }

  std::shared_ptr<Graph> graph_;
  // A stack of active refinements, one for each block
  std::vector<ListRefinement*> active_refinements_;
  // A map from Boolean Value * -> associated refinements
  std::unordered_map<Value*, BoolRefinements> info_;
  std::unordered_set<Block*> throwing_blocks_;
  bool changed_ = false;
};

bool RefineIntegerValues(const std::shared_ptr<Graph>& graph) {
  return IntegerValueRefiner(graph).run();
}

} // namespace jit
} // namespace torch