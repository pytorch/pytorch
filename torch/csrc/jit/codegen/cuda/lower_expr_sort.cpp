#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace {

//! Returns an output tensor of an expression if found.
TensorView* findOutputTensor(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      expr->outputs().size() <= 1, "Unexpected number of outputs");
  if (expr->outputs().size() != 1) {
    return nullptr;
  }
  auto out = expr->output(0);
  if (out->getValType() != ValType::TensorView) {
    return nullptr;
  }
  return out->as<TensorView>();
}

struct TargetInfo {
  TensorView* target = nullptr;
  unsigned score = 0;
};

//! Finds the tensor that governs the loop-nest where an Expr should
//! be placed. Also, gives a score to the expression for the ordering
//! among the expressions in the same loop-nest.
TargetInfo findTargetTensor(Expr* expr) {
  TORCH_INTERNAL_ASSERT(expr->outputs().size() <= 1);

  TargetInfo info;

  TensorView* out_tv = findOutputTensor(expr);
  if (out_tv == nullptr) {
    return info;
  }

  if (!out_tv->hasComputeAt()) {
    info.target = out_tv;
    // No computeAt, so this should come last.
    info.score = std::numeric_limits<unsigned>::max();
    return info;
  }

  // Note this returns the computeAt position
  int pos = (int)out_tv->getRelativeComputeAtAxis();
  info.target = out_tv->getComputeAtView();
  while (info.target->hasComputeAt()) {
    if ((int)info.target->getThisComputeAtAxis() < pos) {
      break;
    }
    // getComputeAtRelPos accepts an axis index.
    pos = pos == 0 ? 0 : info.target->getComputeAtRelPos(pos - 1) + 1;
    info.target = info.target->getComputeAtView();
  }

  info.score = pos;
  return info;
}

// Type definitions for brevity
using ExprList = std::vector<Expr*>;
using TargetGroupMap = std::unordered_map<TensorView*, ExprList>;
using ExprTargetMap = std::unordered_map<Expr*, TensorView*>;
using Score = unsigned;
using ExprScoreMap = std::unordered_map<const Expr*, Score>;

void sanityCheck(
    const ExprList& exprs,
    const ExprList& reordered_exprs,
    const ExprScoreMap& scores,
    const ExprTargetMap& target_map,
    const TargetGroupMap& computed_at_exprs) {
  const auto num_exprs = exprs.size();
  TORCH_INTERNAL_ASSERT(scores.size() == num_exprs);
  TORCH_INTERNAL_ASSERT(
      reordered_exprs.size() + target_map.size() == num_exprs);
  int num_computed_exprs = std::accumulate(
      computed_at_exprs.begin(),
      computed_at_exprs.end(),
      0,
      [](int acc, const std::pair<TensorView*, ExprList>& p) {
        return acc + p.second.size();
      });
  TORCH_INTERNAL_ASSERT(num_computed_exprs == (int)target_map.size());
}

// Arrange exprs into loop-nest groups. Loop-nest groups are
// disjoint grouping of expressions based on the expression
// where each expression is computed at.
void groupExpressions(
    Expr* expr,
    ExprList& reordered_exprs,
    ExprTargetMap& target_map,
    TargetGroupMap& computed_at_exprs,
    ExprScoreMap& scores) {
  const auto info = findTargetTensor(expr);
  scores.emplace(expr, info.score);
  if (info.target == nullptr) {
    reordered_exprs.push_back(expr);
  } else {
    target_map.emplace(expr, info.target);
    if (computed_at_exprs.find(info.target) == computed_at_exprs.end()) {
      computed_at_exprs.emplace(info.target, TargetGroupMap::mapped_type());
    }
    auto& exprs = computed_at_exprs[info.target];
    exprs.push_back(expr);
  }
}

// Sort each loop-nest group based on axis (i.e., score)
void sortGroup(ExprList& exprs, ExprScoreMap& scores) {
  std::stable_sort(
      exprs.begin(),
      exprs.end(),
      [&scores](const Expr* expr1, const Expr* expr2) {
        return scores[expr1] < scores[expr2];
      });
}

// If an expression is missing from expr_status, search for all ancestors
// that are necessary for the expression
void mapMissingInputsToAncestors(
    const TensorView* tv,
    const std::unordered_map<const Expr*, bool>& expr_status,
    std::vector<const TensorView*>& ancestors) {
  const Expr* expr = tv->definition();
  const auto& expr_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
  for (auto input : expr_inputs) {
    const Expr* input_definition = input->definition();
    if (input_definition != nullptr) {
      if (expr_status.find(input_definition) == expr_status.end()) {
        mapMissingInputsToAncestors(input, expr_status, ancestors);
      } else {
        ancestors.push_back(input);
      }
    }
  }
}

// For each expression, find all TensorView inputs.
// If an input TensorView is missing from expr_status,
// find that input's ancestors that are present in expr_status.
std::unordered_map<const Expr*, std::vector<const TensorView*>> findExprTvInputs(
    const std::unordered_map<const Expr*, bool>& expr_status) {
  std::unordered_map<const Expr*, std::vector<const TensorView*>>
      map_expr_to_tv_inputs;

  // Iterate over all exprs and filter missing expr
  for (auto item : expr_status) {
    const auto expr = item.first;
    const auto& expr_inputs =
        ir_utils::filterByType<TensorView>(expr->inputs());

    map_expr_to_tv_inputs.insert({expr, std::vector<const TensorView*>()});
    auto& tv_inputs = map_expr_to_tv_inputs[expr];

    for (auto input : expr_inputs) {
      const Expr* input_definition = input->definition();
      bool missing_input = input_definition != nullptr &&
          expr_status.find(input_definition) == expr_status.end();

      if (missing_input) {
        // Map missing input to ancestor that is present in exprs_status
        std::vector<const TensorView*> ancestors;
        mapMissingInputsToAncestors(input, expr_status, ancestors);
        tv_inputs.insert(tv_inputs.begin(), ancestors.begin(), ancestors.end());
      } else {
        tv_inputs.push_back(input);
      }
    }
  }
  return map_expr_to_tv_inputs;
}

// Reorder expressions that are computed at the same position in a
// breadth-first order.
void reorderSegmentBreadthFirst(
    ExprList::iterator seg_begin,
    ExprList::const_iterator seg_end) {
  // mapping of each expression to a bool flag indicating if it's
  // already been visited
  std::unordered_map<const Expr*, bool> expr_status;
  for (auto it = seg_begin; it != seg_end; ++it) {
    expr_status.insert({*it, false});
  }

  // Holds all input TVs necessary for every expression.
  const auto map_expr_to_tv_inputs = findExprTvInputs(expr_status);

  while (seg_begin != seg_end) {
    std::vector<const Expr*> visited_exprs;
    for (auto it = seg_begin; it != seg_end; ++it) {
      const auto expr = *it;
      const auto& expr_inputs = map_expr_to_tv_inputs.at(expr);

      // if all input expressions are visited
      // then expr can be visited
      const bool ready_to_visit = std::all_of(
          expr_inputs.begin(),
          expr_inputs.end(),
          [&expr_status](const TensorView* input) {
            const Expr* input_definition = input->definition();
            return input_definition == nullptr ||
                (expr_status.find(input_definition) != expr_status.end() &&
                 expr_status.at(input_definition));
          });
      if (ready_to_visit) {
        std::iter_swap(seg_begin, it);
        TORCH_INTERNAL_ASSERT(*seg_begin == expr);
        ++seg_begin;
        visited_exprs.push_back(expr);
      }
    }
    for (const auto& visited_expr : visited_exprs) {
      expr_status.at(visited_expr) = true;
    }
  }
}

// Reorder expressions in a group in a breadth-first order. Reordering
// is done within a subset of expressions that have the same score
// (i.e., computeAt position). For each subset,
// reorderSegmentBreadthFirst is called.
void reorderGroupBreadthFirst(ExprList& exprs, const ExprScoreMap& scores) {
  auto seg_begin = exprs.begin();
  auto seg_end = exprs.begin();
  Score seg_score = scores.at(*seg_begin);
  while (seg_end != exprs.end()) {
    const auto expr = *seg_end;
    const auto cur_score = scores.at(expr);
    if (seg_score == cur_score) {
      // advance further
      ++seg_end;
      continue;
    } else if (seg_score < cur_score) {
      // segment ended
      reorderSegmentBreadthFirst(seg_begin, seg_end);
      seg_begin = seg_end;
      seg_score = cur_score;
    } else {
      // exprs list is assumed to be sorted in the order of scores, so
      // this should never be reachable
      TORCH_INTERNAL_ASSERT(
          false, "Unexpected expression: ", expr, ", score: ", cur_score);
    }
  }
  reorderSegmentBreadthFirst(seg_begin, seg_end);
}

void mergeNonRootGroupsIntoRootGroups(
    TargetGroupMap& computed_at_exprs,
    ExprTargetMap& target_map) {
  for (auto it = computed_at_exprs.begin(); it != computed_at_exprs.end();) {
    TensorView* target = it->first;
    if (target->hasComputeAt()) {
      Expr* target_expr = target->definition();
      TensorView* target_of_target = target_map.at(target_expr);
      auto& target_group = computed_at_exprs.at(target_of_target);
      auto pos =
          std::find(target_group.begin(), target_group.end(), target_expr);
      TORCH_INTERNAL_ASSERT(pos != target_group.end());
      target_group.insert(pos, it->second.begin(), it->second.end());
      // Update the target map
      for (auto& inserted_expr : it->second) {
        TORCH_INTERNAL_ASSERT(target_map.at(inserted_expr) == target);
        target_map.at(inserted_expr) = target_of_target;
      }
      it = computed_at_exprs.erase(it);
    } else {
      ++it;
    }
  }
}

// Merge root loop-nests into reordered_exprs
void mergeGroupsIntoSortedList(
    TargetGroupMap& computed_at_exprs,
    ExprList& reordered_exprs) {
  while (computed_at_exprs.size() > 0) {
    // Find the root loop-nest that has no dependency with the other
    // loop-nests
    TensorView* cur_target = computed_at_exprs.begin()->first;
    for (auto& group : computed_at_exprs) {
      auto target = group.first;
      if (cur_target == target)
        continue;
      if (DependencyCheck::isDependencyOf(target, cur_target)) {
        cur_target = target;
      }
    }
    // cur_target can be visited
    reordered_exprs.insert(
        reordered_exprs.end(),
        computed_at_exprs.at(cur_target).begin(),
        computed_at_exprs.at(cur_target).end());
    computed_at_exprs.erase(cur_target);
  }
}

} // namespace

// Reorder exprs so that LoopNestGenerator::handle(Expr*) can generate
// correct loop nests. Vector exprs is assumed to be topologically
// sorted, but that is not sufficient as tensors computed at
// outer loops need to be located earlier.
std::vector<Expr*> reorderExprsForComputeAt(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("reorderExprsForComputeAt");
  ExprList reordered_exprs;

  // expr -> target
  ExprTargetMap target_map;

  // target -> [computed at expressions]
  TargetGroupMap computed_at_exprs;

  // score of each expression that is calculated based on the
  // computeAt axis. A lower score of an expression means it should be
  // placed earlier in the expression list. This is a requirement for
  // the loop-nest generation of this class to work.
  ExprScoreMap scores;

  // 1. Group expressions by target tensors. Non-grouped expressions
  // are copied into reordered_exprs.
  for (auto& expr : exprs) {
    groupExpressions(
        expr, reordered_exprs, target_map, computed_at_exprs, scores);
  }

  sanityCheck(exprs, reordered_exprs, scores, target_map, computed_at_exprs);

  // If no computeAt found, no need to reorder.
  if (computed_at_exprs.size() == 0) {
    return exprs;
  }

  // 2. Sort each loop-nest group based on axis (i.e., score)
  for (auto& group : computed_at_exprs) {
    sortGroup(group.second, scores);

    // Reorder expressions in a breadth-first order
    reorderGroupBreadthFirst(group.second, scores);
  }

  // 3. Merge non-root loop-nests into root loop-nests
  mergeNonRootGroupsIntoRootGroups(computed_at_exprs, target_map);

  // At this point, only root loop-nests (i.e., no computeAt'ed)
  // should exist.
  for (auto& group : computed_at_exprs) {
    // Guarantee only root loop-nests exist.
    TensorView* target = group.first;
    TORCH_INTERNAL_ASSERT(!target->hasComputeAt());
  }

  sanityCheck(exprs, reordered_exprs, scores, target_map, computed_at_exprs);

  mergeGroupsIntoSortedList(computed_at_exprs, reordered_exprs);

  // Reordering completed. Reordered exprs exist in reordered_exprs.

  TORCH_INTERNAL_ASSERT(exprs.size() == reordered_exprs.size());
  return reordered_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
