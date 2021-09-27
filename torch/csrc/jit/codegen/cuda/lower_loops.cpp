#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <algorithm>
#include <deque>
#include <numeric>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<kir::Expr*> LoopNestGenerator::loweredExprs(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::LoopNestGenerator::loweredExprs");
  TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  LoopNestGenerator generator(exprs);
  return generator.lowered_exprs_;
}

LoopNestGenerator::LoopNestGenerator(const std::vector<Expr*>& exprs) {
  generate(exprs);
}

namespace {

kir::ForLoop* openForHelper(kir::ForLoop* scope, IterDomain* id) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());
  const auto kir_id = gpu_lower->lowerValue(id)->as<kir::IterDomain>();
  auto extent_with_halo = gpu_lower->haloInfo().getExtent(kir_id);
  kir::ForLoop* new_scope = nullptr;
  if (extent_with_halo) {
    // When an axis is extended with halo, unrolling and vectorization
    // are assumed to not be used for now.
    TORCH_INTERNAL_ASSERT(
        id->getParallelType() != ParallelType::Unroll &&
        !isParallelTypeVectorize(id->getParallelType()));
    // Use the extent that's extended by halo
    new_scope = ir_builder.create<kir::ForLoop>(
        kir_id,
        id->isBroadcast() ? ir_builder.zeroVal()
                          : ir_builder.create<kir::Int>(c10::nullopt),
        nullptr,
        extent_with_halo,
        nullptr,
        false,
        nullptr);
  } else {
    new_scope = ir_builder.create<kir::ForLoop>(kir_id);
  }
  if (scope != nullptr) {
    scope->body().insert(0, new_scope);
  }
  return new_scope;
}

} // namespace

void LoopNestGenerator::openFor(IterDomain* iter_domain) {
  if (for_loops_.size() > 0) {
    const auto new_scope = openForHelper(for_loops_.back(), iter_domain);
    // for_loop_allocations_.insert({new_scope, 0});
    for_loops_.push_back(new_scope);
  } else {
    for_loops_.push_back(openForHelper(nullptr, iter_domain));
    lowered_exprs_.insert(lowered_exprs_.begin(), for_loops_.back());
  }
}

void LoopNestGenerator::closeFor() {
  TORCH_INTERNAL_ASSERT(!for_loops_.empty());
  for_loops_.pop_back();
}

void LoopNestGenerator::pushFront(kir::Expr* expr) {
  if (for_loops_.size() == 0) {
    lowered_exprs_.insert(lowered_exprs_.begin(), expr);
  } else {
    for_loops_.back()->body().insert(0, expr);
  }
}

void LoopNestGenerator::handle(Expr* expr) {
  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Check if it's a tensor view expression we need to place in the loop nest
  // structure
  if (!ir_utils::isTVOp(expr)) {
    // Close all the loops, scalar operations cannot be inside for loops based
    // on expr sorting.
    while (!for_loops_.empty()) {
      closeFor();
    }
    pushFront(gpu_lower->lowerExpr(expr));

    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(
          out->getValType().value() == ValType::Scalar,
          "Unrecognized output type found in expr ",
          expr,
          " cannot lower ",
          out->getValType().value());

      pushFront(ir_builder.create<kir::Allocate>(
          gpu_lower->lowerValue(out),
          MemoryType::Local,
          ir_builder.create<kir::Int>(1)));
    }
    return;
  }

  TensorView* out_tv = expr->output(0)->as<TensorView>();

  // Figure out what the entire loop structure should look like.
  std::deque<IterDomain*> loop_structure;

  // Fill the entire loop structure by Looking at each axis
  // individually in out's domain
  for (size_t out_i = 0; out_i < out_tv->nDims(); out_i++) {
    // Note: It is not safe to skip trivial reduction axes as they could be
    // inlined with other tensor views. This happens in
    // NVFuserTest.FusionBNRepro_CUDA as of this commit on norm_hack_2_rebased
    // branch

    // Look up the concrete ID in the parallel map, not in the loop
    // map, which also maps non-CA axes.
    auto concrete_id =
        gpu_lower->caParallelMap().getConcreteMappedID(out_tv->axis(out_i));
    loop_structure.push_back(concrete_id);
  }

  auto loop_structure_it = loop_structure.begin();
  auto for_loop_it = for_loops_.begin();
  auto last_for_loop_matched = for_loops_.begin();

  // Match the loop structure with the current for-loops. Reuse
  // matching loops and close unmatched ones.
  while (loop_structure_it != loop_structure.end() &&
         for_loop_it != for_loops_.end()) {
    auto lowered_out_id =
        gpu_lower->lowerValue(*loop_structure_it)->as<kir::IterDomain>();
    // Similar to the above, the parallel map is used rather than the
    // loop map. Again, non-CA axes should not share loops, so the
    // parallel map should be used.
    if (gpu_lower->caParallelMap().areMapped(
            lowered_out_id, (*for_loop_it)->iter_domain())) {
      loop_structure_it++;
      last_for_loop_matched = ++for_loop_it;
    } else {
      ++for_loop_it;
    }
  }

  auto n_loops_to_close =
      std::distance(last_for_loop_matched, for_loops_.end());

  TORCH_INTERNAL_ASSERT(
      n_loops_to_close >= 0 &&
          n_loops_to_close <= (std::ptrdiff_t)for_loops_.size(),
      "Tried to close an invalid number of loops: ",
      n_loops_to_close);

  if (max_close < n_loops_to_close && max_close > 0) {
    // Figure out where the last for loop matches from out_tv, go until the
    // max_close loop marked from previous tv's producer domain. Make sure
    // none of these domains are actually present in current out_tv. If these
    // loops map to current out_tv, it should be responsible for deciding if
    // they stay or go, this could result from an invalid compute at topology
    // on the DAG or bad expression sorting.
    auto for_loops_it = for_loops_.end() - n_loops_to_close;
    auto for_loops_it_end = for_loops_.end() - max_close;

    for (; for_loops_it != for_loops_it_end; for_loops_it++) {
      TORCH_INTERNAL_ASSERT(
          std::none_of(
              loop_structure_it,
              loop_structure.end(),
              [&gpu_lower, &for_loops_it](IterDomain* loop_structure_id) {
                // Check loop structure doesn't map for_loops in for loop map
                auto id0 = (*for_loops_it)->iter_domain();
                auto id1 = gpu_lower->lowerValue(loop_structure_id)
                               ->as<kir::IterDomain>();
                return gpu_lower->caLoopMap().areMapped(id0, id1);
              }),
          "Invalid loop found to close.");
    }

    n_loops_to_close = std::min(n_loops_to_close, max_close);
  }

  for (int64_t i_loop_close = 0; i_loop_close < n_loops_to_close;
       i_loop_close++) {
    closeFor();
  }

  // Open the remaining needed loops
  for (; loop_structure_it != loop_structure.end(); ++loop_structure_it) {
    openFor(*loop_structure_it);
  }

  if (out_tv->getMaxProducerPosition() == 0) {
    max_close = -1;
  } else {
    auto produce_at_id = loop_structure[out_tv->getMaxProducerPosition() - 1];
    auto max_close_loop = std::find_if(
        for_loops_.begin(),
        for_loops_.end(),
        [&produce_at_id, &gpu_lower](kir::ForLoop* fl) {
          auto produce_at_lowered_it =
              gpu_lower->lowerValue(produce_at_id)->as<kir::IterDomain>();
          return gpu_lower->caParallelMap().areMapped(
              produce_at_lowered_it, fl->iter_domain());
        });

    max_close = std::distance(max_close_loop, for_loops_.end());
    max_close = max_close > 0 ? max_close - 1 : max_close;
  }
  pushFront(gpu_lower->lowerExpr(expr));
}

// Generate the loop nest structure and place it in lowered_exprs_
void LoopNestGenerator::generate(const std::vector<Expr*>& exprs) {
  TORCH_INTERNAL_ASSERT(lowered_exprs_.empty());

  // Process the carefully ordered expressions
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    handle(*it);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
