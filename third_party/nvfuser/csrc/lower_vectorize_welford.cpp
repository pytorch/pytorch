#include <lower_vectorize_welford.h>

#include <dispatch.h>
#include <instrumentation.h>
#include <ir_utils.h>
#include <kernel_ir_dispatch.h>
#include <lower2device.h>
#include <lower_utils.h>
#include <ops/arith.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Vectorize serial WelfordOp, in other words hoists loop-invariant
// expressions out of a loop that is exactly mapped with a vectorized
// IterDomain.
//
// This pass currently requires tensor indexing has been completed. As
// we hoist the count Val of a WelfordOp out of its innermost loop, if
// indexing weren't done, it would get confused as it would miss the
// innermost loop. The override map of indexing might help to work
// around it.
class WelfordVectorizer : public kir::ExprMutator {
 public:
  static std::vector<Expr*> vectorize(const std::vector<Expr*>& exprs) {
    WelfordVectorizer inserter(exprs);
    return inserter.exprs_;
  }

 private:
  WelfordVectorizer(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  void handle(WelfordOp* wop) final {
    // Indexing is assumed to be completed
    if (!wop->out()->isA<kir::TensorIndex>()) {
      return;
    }

    if (isVectorizableWelford(wop)) {
      vectorize(wop);
    }
  }

  // Check if a WelfordOp can be vectorized.
  //
  // Look for a pattern as:
  //
  // for (int i = 0; i < N; ++i) {
  //   if (pred) {
  //     WelfordCombine();
  //   }
  // }
  bool isVectorizableWelford(WelfordOp* wop) const {
    const auto out_tv = ir_utils::getTvOutput(wop);
    const auto out_domain = out_tv->domain();

    // Only consider the innermost leaf ID to vectorize. Should be
    // possible to consider non-innermost IDs as well
    auto innermost_leaf_id = out_tv->axis(-1);

    if (innermost_leaf_id->isReduction() || innermost_leaf_id->isBroadcast()) {
      return false;
    }

    // Check if the innermost loop can be vectorized
    TORCH_INTERNAL_ASSERT(!for_loops_.empty());
    auto innermost_loop = for_loops_.back();

    if (innermost_loop->isTrivial()) {
      return false;
    }

    // Allreduce not supported
    if (wop->isAllreduce()) {
      return false;
    }

    if (!GpuLower::current()->caMap()->areMapped(
            innermost_loop->iter_domain(),
            innermost_leaf_id,
            IdMappingMode::EXACT)) {
      return false;
    }

    // Check if the loop is exactly mapped with a vectorized
    // ID. Technically, predicate hoisting is legal as long as this
    // loop is produced only with divisible splits, but for now only
    // enable when it's mapped with a vectorized ID.
    const auto& exact_set = GpuLower::current()
                                ->caMap()
                                ->getIdSets(IdMappingMode::EXACT)
                                .getDisjointSetOf(innermost_leaf_id);
    // If none of IterDomains is vectorized, don't vectorize the WelfordOp
    if (std::none_of(exact_set.begin(), exact_set.end(), [&](IterDomain* id) {
          return id->getParallelType() == ParallelType::Vectorize;
        })) {
      return false;
    }

    // This optimization should be safe for the initial sequential
    // welford, where the var and N arguments are zero and one,
    // respectively. It should also be mostly safe otherwise, but not
    // guaranteed.
    if (out_domain->hasBlockReduction() || out_domain->hasGridReduction() ||
        !wop->inVar()->isZeroInt() || !wop->inN()->isOneInt()) {
      return false;
    }

    // The WelfordOp structure should look like:
    //
    // if (lowered_pred) {
    //   WelfordOp(...);
    // }
    //
    // Bail out if the structure is not detected.

    TORCH_INTERNAL_ASSERT(!scope_exprs_.empty());
    kir::IfThenElse* wop_ite =
        dynamic_cast<kir::IfThenElse*>(scope_exprs_.back());
    if (wop_ite == nullptr) {
      // Unexpected code structure
      return false;
    }

    // The predicate should be either Manual or Inline
    if (wop_ite->predicate()->predicate_type() != PredicateType::Manual &&
        wop_ite->predicate()->predicate_type() != PredicateType::Inline) {
      return false;
    }

    TORCH_INTERNAL_ASSERT(
        wop_ite->predicate()->hasValue(),
        "All predicates should have been lowered at this point: ",
        wop_ite->toString());

    if (!(wop_ite->thenBody().size() == 1 && wop_ite->thenBody().at(0) == wop &&
          wop_ite->elseBody().empty())) {
      return false;
    }

    return true;
  }

  // Transform a serial WelfordOp.
  void vectorize(WelfordOp* wop) {
    TORCH_INTERNAL_ASSERT(!scope_exprs_.empty());
    kir::IfThenElse* wop_ite =
        dynamic_cast<kir::IfThenElse*>(scope_exprs_.back());
    TORCH_INTERNAL_ASSERT(
        wop_ite != nullptr,
        "Predicate IfThenElse not found for ",
        wop->toString());

    TORCH_INTERNAL_ASSERT(!for_loops_.empty());
    innermost_loop_ = for_loops_.back();

    scope_of_innermost_loop_ = nullptr;
    for (int i = (int)scope_.size() - 1; i >= 0; --i) {
      if (&(innermost_loop_->body()) == scope_.at(i)) {
        scope_of_innermost_loop_ = scope_.at(i - 1);
      }
    }
    TORCH_INTERNAL_ASSERT(scope_of_innermost_loop_ != nullptr);

    // If the expr is predicated, hoist the predicate as the innermost
    // loop should not have any dependency with the predicate (which
    // is guaranteed by the fact that the loop is exactly mapped with
    // a vectorized IterDomain)
    bool is_predicated = true;

    auto pred = wop_ite->predicate();
    if (pred->isConst()) {
      is_predicated = false;
      // If the conditional is const, it isn't really
      // predicated. If it's true, this IfThenElse can just be
      // ignored. If not, the whole block of IfThenElse should be ignored.
      if (!pred->value()->value()) {
        // Can this happen? This should be just ignored, assuming
        // there's no else path.
        TORCH_INTERNAL_ASSERT(
            wop_ite->elseBody().empty(),
            "Unexpected IfThenElse: ",
            wop_ite->toString());
      }
    }

    // The transformation pattern slightly varies depending on how the
    // Welford expr is predicated. See the comment of each of the
    // transformation functions.

    if (is_predicated) {
      // When predicated, check if the whole loop body has the same
      // predicate. If so, just wrap the loop with the predicate
      if (canPredicateWholeLoop(wop, wop_ite)) {
        vectorizeWithLoopPredicate(wop);
      } else {
        vectorizeWithInlinePredicate(wop);
      }
    } else {
      vectorizeWithoutPredicate(wop);
    }
  }

  // Non-predicated case
  //
  // Before:
  // for () {
  //   if (true) {
  //     welfordCombine(...);
  //   }
  // }
  //
  // After:
  // nvfuser_index_t new_count = outN()[0] + 1;
  // float reciprocal = 1 / new_count;
  // for () {
  //   welfordVectorized(..., new_count, reciprocal);
  // }
  void vectorizeWithoutPredicate(WelfordOp* wop) {
    auto vectorized_wop = applyVectorizeTransformation(wop, nullptr);

    registerReplace(wop, vectorized_wop);
  }

  // Predicated case
  //
  // Before:
  // for (i) {
  //   if (pred) {
  //     welfordCombine(...);
  //   }
  // }
  //
  // After:
  // bool p = pred;
  // nvfuser_index_t new_count = outN()[0] + p;
  // float reciprocal = 0;
  // if (p) {
  //   reciprocal = 1 / new_count;
  // }
  // for (i) {
  //   welfordVectorized(..., new_count, reciprocal, p);
  // }
  void vectorizeWithInlinePredicate(WelfordOp* wop) {
    kir::IfThenElse* wop_ite = scope_exprs_.back()->as<kir::IfThenElse>();
    auto conditional = wop_ite->predicate()->value();

    // Allocate a boolean scalar for cond
    auto pred_var = defineScalar(DataType::Bool)->as<Bool>();

    registerInsertBeforeInnerMostLoop(
        IrBuilder::create<UnaryOp>(UnaryOpType::Set, pred_var, conditional));

    auto vectorized_wop = applyVectorizeTransformation(wop, pred_var);

    // Replace the predicated welfordOp with welfordVectorized. The
    // if-then-else predicate is no longer needed.
    registerReplace(wop_ite, vectorized_wop, *(scope_.rbegin() + 1));
  }

  // Predicated case when the innermost loop has no externaly
  // visible effect except for the welford outputs
  //
  // Before:
  // for (i) {
  //   if (pred) {
  //     welfordCombine(...);
  //   }
  // }
  //
  // After:
  // if (pred) {
  //   nvfuser_index_t new_count = outN()[0] + 1;
  //   float reciprocal = 1 / new_count;
  //   for (i) {
  //     welfordVectorized(..., new_count, reciprocal);
  //   }
  // }
  void vectorizeWithLoopPredicate(WelfordOp* wop) {
    kir::IfThenElse* wop_ite = scope_exprs_.back()->as<kir::IfThenElse>();
    auto conditional = wop_ite->predicate()->value();

    auto loop_ite = IrBuilder::create<kir::IfThenElse>(
        IrBuilder::create<kir::Predicate>(conditional));

    // Insert the IfThenElse and move the innermost loop inside it
    registerInsertBeforeInnerMostLoop(loop_ite);
    registerRemove(innermost_loop_, scope_of_innermost_loop_);
    registerInsertBefore(nullptr, innermost_loop_, &loop_ite->thenBody());

    // Change the scope of the innermost loop to the body of loop_ite
    scope_of_innermost_loop_ = &loop_ite->thenBody();

    auto vectorized_wop = applyVectorizeTransformation(wop, nullptr);

    registerReplace(wop_ite, vectorized_wop, &innermost_loop_->body());
  }

  kir::VectorizedWelfordOp* applyVectorizeTransformation(
      WelfordOp* wop,
      Bool* pred) {
    DataType data_type = wop->outAvg()->getDataType().value();
    DataType index_type = wop->outN()->getDataType().value();

    bool is_predicated = pred != nullptr;
    if (!is_predicated) {
      pred = GpuLower::current()->kernel()->trueVal();
    }

    // nvfuser_index_t new_count;
    // new_count = hoisted_count + count_increment

    auto new_count = defineScalar(index_type);

    auto hoisted_count = hoistCount(wop->outN()->as<kir::TensorIndex>());

    Int* count_increment = nullptr;
    if (!is_predicated) {
      count_increment = GpuLower::current()->kernel()->oneVal();
    } else {
      // count_increment = (int)pred;
      count_increment = defineScalar(index_type)->as<Int>();
      registerInsertBeforeInnerMostLoop(
          IrBuilder::create<UnaryOp>(UnaryOpType::Cast, count_increment, pred));
    }

    registerInsertBeforeInnerMostLoop(IrBuilder::create<BinaryOp>(
        BinaryOpType::Add, new_count, hoisted_count, count_increment));

    // float new_count_float;
    auto new_count_float = defineScalar(data_type);

    // new_count_float = (float)new_count;
    registerInsertBeforeInnerMostLoop(IrBuilder::create<UnaryOp>(
        UnaryOpType::Cast, new_count_float, new_count));

    // float reciprocal;
    auto reciprocal = defineScalar(data_type);

    auto reciprocal_expr = IrBuilder::create<BinaryOp>(
        BinaryOpType::Div,
        reciprocal,
        GpuLower::current()->kernel()->oneVal(),
        new_count_float);

    // If not predicated, just set the reciprocal variable
    // with the reciprocl expr. Otherwise, guard it with an if
    // statement.
    if (!is_predicated) {
      registerInsertBeforeInnerMostLoop(reciprocal_expr);
    } else {
      // Initialize reciprocal as 0;
      registerInsertBeforeInnerMostLoop(IrBuilder::create<UnaryOp>(
          UnaryOpType::Set,
          reciprocal,
          GpuLower::current()->kernel()->zeroVal()));

      // if (pred) reciprocal = 1 / new_count_float;
      auto reciprocal_ite = IrBuilder::create<kir::IfThenElse>(
          IrBuilder::create<kir::Predicate>(pred));
      registerInsertBeforeInnerMostLoop(reciprocal_ite);
      registerInsertBefore(
          nullptr, reciprocal_expr, &(reciprocal_ite->thenBody()));
    }

    auto vectorized_wop = IrBuilder::create<kir::VectorizedWelfordOp>(
        wop->outputTriplet(),
        wop->inputTriplet(),
        wop->initTriplet(),
        new_count,
        reciprocal,
        pred);

    return vectorized_wop;
  }

  // Declare a scalar variable of type dt and insert its allocation
  Val* defineScalar(DataType dt) {
    Val* val = IrBuilder::newScalar(dt);

    auto alloc = IrBuilder::create<kir::Allocate>(
        val, MemoryType::Local, GpuLower::current()->kernel()->oneVal());

    registerInsertBeforeInnerMostLoop(alloc);

    return val;
  }

  // Hoist the count TensorIndex out of the innermost loop, assuming
  // it is invariant within the loop. The loop index can be replaced
  // with what ever value within the loop range since it is
  // independent of the loop index.
  kir::TensorIndex* hoistCount(kir::TensorIndex* out_N) {
    TORCH_INTERNAL_ASSERT(!for_loops_.empty());
    auto innermost_loop = for_loops_.back();
    const auto& original_index = out_N->index();
    std::unordered_map<Val*, Val*> index_replacement_map;
    index_replacement_map.emplace(
        innermost_loop->index(), GpuLower::current()->kernel()->zeroVal());

    Val* indices_zero =
        ir_utils::replaceValInIndexVal(original_index, index_replacement_map);

    auto hoisted_count =
        IrBuilder::create<kir::TensorIndex>(out_N->view(), indices_zero);

    return hoisted_count;
  }

  // Check if all exprs in the same scope as a WelfordOp have the same
  // predicate, thus allowing the predicates of all of the exprs can
  // be hoisted collectively. The motivation of this analysis is to
  // transform exprs from:
  //
  // for () {
  //   if (pred1) {
  //     expr1;
  //   }
  //   if (pred2) {
  //     welfordOp;
  //   }
  // }
  //
  // to below when pred1 == pred2.
  //
  // if (pred1) {
  //   for () {
  //     expr1;
  //     welfordOp;
  //   }
  // }
  //
  // Rather than:
  //
  // bool pred2 = ...;
  // for () {
  //   if (pred1) {
  //     expr1;
  //   }
  //   welfordOp(pred2, ...);
  // }
  //
  // The pattern matching here only works with a simple sequence of
  // expressions but should be sufficient as it is likely the most
  // common situation.
  bool canPredicateWholeLoop(WelfordOp* wop, kir::IfThenElse* wop_ite) {
    // The thread predicate may be different from the other
    // expressions in the same innermost loop. Only allows if it's not
    // used or known to be true. We could check if all the other
    // expressions have the same thread predicate, though not sure if
    // it could have any meaningful benfit
    auto thread_pred = wop_ite->predicate()->thread_pred();
    if (thread_pred != nullptr &&
        (!thread_pred->isConst() || !thread_pred->value())) {
      return false;
    }

    TORCH_INTERNAL_ASSERT(!for_loops_.empty());
    auto innermost_loop = for_loops_.back();

    // Check all the exprs in the same scope
    for (auto expr : innermost_loop->body().exprs()) {
      // Bail out if a loop is found
      if (expr->isA<kir::ForLoop>()) {
        return false;
      }

      // If expr is not an IfThenElse and not predicated, nothing to
      // worry about
      if (!expr->isA<kir::IfThenElse>() && expr->predicate() == nullptr) {
        continue;
      }

      // Consider only when this is a predicated expr, i.e, expr is an
      // IfThenElse containing just one expr. When it's not an
      // IfThenElse, it should be an expr with predicates embedded in
      // it as predicate_, e.g., block/grid reductions. It is highly
      // unlikely such expressions show up in the same ForLoop as a
      // sequential WelfordOp, so not handling them shouldn't be a
      // problem.
      auto expr_ite = dynamic_cast<kir::IfThenElse*>(expr);
      if (expr_ite == nullptr ||
          !(expr_ite->thenBody().size() == 1 &&
            expr_ite->elseBody().size() == 0)) {
        return false;
      }

      expr = expr_ite->thenBody().at(0);

      // Check only other expressions than the WelfordOp itself
      if (expr == wop) {
        continue;
      }

      // Only TV ops should appear
      if (!ir_utils::isTvOp(expr)) {
        return false;
      }

      TensorView* tv = ir_utils::getTvOutput(expr);

      // tv should never be nullptr as non-TvOp is filtered out, but
      // if that happens, just bails out
      if (tv == nullptr) {
        return false;
      }

      // Make sure this expr doesn't have a thread predicate
      auto expr_thread_pred = expr_ite->predicate()->thread_pred();
      if (expr_thread_pred != nullptr &&
          (!expr_thread_pred->isConst() || !expr_thread_pred->value())) {
        return false;
      }

      // Check if the predicate of the expr is known to be true
      if (expr_ite->predicate()->isConst() &&
          expr_ite->predicate()->value()->value()) {
        continue;
      }

      // If tv is fully loop-mapped with the welford output TV, it is
      // guaranteed that its predicate can be safely replaced with the
      // predicate of the WelfordOp. If not, that doesn't necessarily
      // mean the expr predicate is different, but likely not
      // worthwhile to consider.
      auto wop_out = ir_utils::getTvOutput(wop);
      for (auto tv_leaf_id : tv->domain()->domain()) {
        if (std::none_of(
                wop_out->domain()->domain().begin(),
                wop_out->domain()->domain().end(),
                [&](auto wop_leaf_id) {
                  return GpuLower::current()->caMap()->areMapped(
                      tv_leaf_id, wop_leaf_id, IdMappingMode::LOOP);
                })) {
          return false;
        }
      }
    }

    return true;
  }

  void registerInsertBeforeInnerMostLoop(Expr* expr) {
    registerInsertBefore(innermost_loop_, expr, scope_of_innermost_loop_);
  }

 private:
  kir::ForLoop* innermost_loop_ = nullptr;
  kir::Scope* scope_of_innermost_loop_ = nullptr;
};

} // namespace

std::vector<Expr*> vectorizeWelford(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::vectorizeWelford");
  return WelfordVectorizer::vectorize(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
