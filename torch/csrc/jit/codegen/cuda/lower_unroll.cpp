#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Provide a new for loop matching the one provided
kir::ForLoop* cloneLoopNest(const kir::ForLoop* for_loop, bool unroll = false) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  const auto new_loop = ir_builder.create<kir::ForLoop>(
      for_loop->index(), for_loop->iter_domain(), for_loop->extent(), unroll);
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      expr = cloneLoopNest(nested_for_loop, unroll);
    }
    new_loop->body().push_back(expr);
  }
  return new_loop;
}

// Create a new vectorize For-Loop
// Add For-Loop to If-Then-Else parent scope
// for (index = start; index < extent; index += offset)
// vectorize flag - Do not generate for-loop
// shift value - Add shift to global indices generated within For-Loop
void cloneVectorizeLoopNests(
    kir::IfThenElse* parent_ite,
    const std::vector<kir::ForLoop*>& for_loops,
    kir::Val* extent,
    bool vectorize,
    kir::Val* shift) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  for (auto fl : for_loops) {
    auto first_expr = fl->body().exprs().front();
    bool has_vectorize_op =
        (first_expr->isA<kir::UnaryOp>() &&
         first_expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set &&
         first_expr->as<kir::UnaryOp>()->out()->isA<kir::TensorView>() &&
         first_expr->as<kir::UnaryOp>()
             ->out()
             ->as<kir::TensorView>()
             ->domain()
             ->hasVectorize());

    TORCH_INTERNAL_ASSERT(!has_vectorize_op || fl->body().exprs().size() == 1);

    const auto new_loop = ir_builder.create<kir::ForLoop>(
        fl->index(),
        fl->iter_domain(),
        vectorize && has_vectorize_op,
        extent,
        false,
        shift);

    for (auto expr : fl->body().exprs()) {
      new_loop->body().push_back(expr);
    }

    parent_ite->thenBody().push_back(new_loop);
  }
}

// Find any child For-Loops
// Add remaining expressions to new parent For-Loop
std::vector<kir::ForLoop*> parseVectorizedForLoop(
    const kir::ForLoop* for_loop,
    kir::ForLoop* new_loop) {
  std::vector<kir::ForLoop*> loops;
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      loops.push_back(nested_for_loop);
    } else {
      new_loop->body().push_back(expr);
    }
  }
  return loops;
}

// Find the first vectorize set - either read or write
// Add child For-Loop to loop_structure
// Enable vectorize flag in child For-Loop
kir::Expr* findVectorizedSet(
    std::vector<kir::ForLoop*>& loop_structure,
    const std::vector<kir::ForLoop*>& for_loops) {
  for (auto fl : for_loops) {
    auto first_expr = fl->body().exprs().front();
    bool has_vectorize_op =
        (first_expr->isA<kir::UnaryOp>() &&
         first_expr->as<kir::UnaryOp>()->operation() == UnaryOpType::Set &&
         fl->iter_domain()->parallelType() ==
             ParallelType::MisalignedVectorize);
    if (has_vectorize_op) {
      loop_structure.push_back(fl);
      return first_expr;
    }
  }
  return nullptr;
}

// Get full extent for the inner-most, merged root domain
kir::Val* getVectorizeExtent(
    kir::TensorView* tv,
    const std::vector<kir::Val*>& indices) {
  auto domain = tv->domain()->hasRFactor() ? tv->domain()->rfactorDomain()
                                           : tv->domain()->rootDomain();

  TORCH_INTERNAL_ASSERT(domain.size() == indices.size());

  bool is_contiguous = true;
  for (auto status : tv->domain()->contiguity()) {
    is_contiguous &= status;
  }

  // If the tensorview is not contiguous, return inner-most root domain extent
  if (!is_contiguous) {
    return domain.back()->extent();
  }

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // Calculate extent of merged root domains
  kir::Val* extent = nullptr;
  for (int i = int(domain.size()) - 1; i >= 0; --i) {
    auto root_id = domain.at(i);
    if (root_id->isBroadcast() || root_id->isReduction() ||
        gpu_lower->trivialReductionInfo().isDerived(root_id)) {
      continue;
    } else if (extent == nullptr) {
      extent = root_id->extent();
    } else if (extent != nullptr && indices.at(i)->isZeroInt()) {
      // This root id must be merged and contiguous. Expand the
      // vectorization partition.
      extent = ir_builder.mulExpr(extent, root_id->extent());
    } else {
      break;
    }
  }

  TORCH_INTERNAL_ASSERT(extent != nullptr);

  return extent;
}

kir::Val* setupNamedScalar(
    kir::Scope& body,
    kir::Val* val,
    const std::string& name,
    bool address = false) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  auto namedScalar = (address) ? ir_builder.namedAddressExpr(name, val)
                               : ir_builder.namedSetExpr(name, val);
  auto alloc = ir_builder.create<kir::Allocate>(
      namedScalar, MemoryType::Local, ir_builder.one());
  body.push_back(alloc);
  body.push_back(namedScalar->definition());
  return namedScalar;
}

kir::ForLoop* handleMisalignedVectorization(
    std::vector<kir::ForLoop*> loop_structure,
    const kir::ForLoop* for_loop) {
  // for_loop body contains allocate, read, compute, write operations
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  // create new base For-Loop
  const auto new_loop = ir_builder.create<kir::ForLoop>(
      for_loop->index(), for_loop->iter_domain(), for_loop->extent());

  // Find child For-Loops and add remaining expressions to base For-Loop
  auto child_loops = parseVectorizedForLoop(for_loop, new_loop);

  // Find the first vectorize set - either read or write
  auto vec_expr = findVectorizedSet(loop_structure, child_loops);
  TORCH_INTERNAL_ASSERT(vec_expr != nullptr);
  TORCH_INTERNAL_ASSERT(vec_expr->outputs().front()->isA<kir::TensorView>());
  TORCH_INTERNAL_ASSERT(vec_expr->inputs().front()->isA<kir::TensorView>());

  auto out_tv = vec_expr->outputs().front()->as<kir::TensorView>();
  auto in_tv = vec_expr->inputs().front()->as<kir::TensorView>();

  // It is assumed that either of input and output is on global memory
  TORCH_INTERNAL_ASSERT(
      out_tv->memoryType() == MemoryType::Global ||
          in_tv->memoryType() == MemoryType::Global,
      "Either input or output tensor must be on global memory.");
  // However, not both of them
  TORCH_INTERNAL_ASSERT(
      !(out_tv->memoryType() == MemoryType::Global &&
        in_tv->memoryType() == MemoryType::Global),
      "Both input and output tensors are on global memory.");
  // Must be either global or local
  TORCH_INTERNAL_ASSERT(
      (out_tv->memoryType() == MemoryType::Global ||
       out_tv->memoryType() == MemoryType::Local),
      "Invalid memory type of output tensor");
  TORCH_INTERNAL_ASSERT(
      (in_tv->memoryType() == MemoryType::Global ||
       in_tv->memoryType() == MemoryType::Local),
      "Invalid memory type of input tensor");

  // TensorView on global memory. This is the tensor that may have
  // a non-aligned base address.
  auto global_tv =
      (out_tv->memoryType() == MemoryType::Global) ? out_tv : in_tv;

  // TensorView with the misaligned vec iterDomain. It is the consumer
  // of vectorized load or the producer of vectorized store. It is
  // assumed that when the output TV is not on global memory, this
  // expression is a vectorized load, so the output TV is vec_tv.
  // TODO: Check vec_tv has indeed MisalignedVectorize parallel type.
  auto vec_tv = (out_tv->memoryType() != MemoryType::Global) ? out_tv : in_tv;

  auto pred = PredicateCompute::getInlinePredicate(
      vec_expr, loop_structure, nullptr, false, true);
  if (pred == nullptr) {
    pred = ir_builder.create<kir::Bool>(true);
  }

  kir::IfThenElse* pred_ite = ir_builder.create<kir::IfThenElse>(pred);
  new_loop->body().push_back(pred_ite);

  // Generate vectorize index
  auto indices = (out_tv->memoryType() == MemoryType::Global)
      ? Index::getConsumerStridedIndices(out_tv->fuserTv(), loop_structure)
      : Index::getProducerStridedIndices(
            in_tv->fuserTv(), out_tv->fuserTv(), loop_structure);

  //  Get full extent for merged root domains
  auto extent = getVectorizeExtent(vec_tv, indices);

  auto vector_size =
      vec_tv->domain()->domain().back()->extent()->as<kir::Int>();

  auto index =
      ir_builder.create<kir::TensorIndex>(global_tv->fuserTv(), indices);
  auto base_address =
      setupNamedScalar(pred_ite->thenBody(), index, "base_address", true);

  kir::Int* data_size =
      ir_builder.create<kir::Int>(dataTypeSize(vec_tv->dtype()));
  auto vector_data_size = ir_builder.mulExpr(vector_size, data_size);
  auto a = ir_builder.modExpr(base_address, vector_data_size);
  auto b = ir_builder.divExpr(a, data_size);
  auto c = ir_builder.subExpr(vector_size, b);
  auto shift_init = setupNamedScalar(pred_ite->thenBody(), c, "shift_val");

  auto shift_pred = ir_builder.eqExpr(shift_init, vector_size);
  auto shift_val =
      ir_builder.whereExpr(shift_pred, ir_builder.zero(), shift_init);
  auto shift = setupNamedScalar(pred_ite->thenBody(), shift_val, "shift");

  auto remaining_extent = ir_builder.subExpr(extent, shift);
  auto remainder_val = ir_builder.modExpr(remaining_extent, vector_size);
  auto remainder =
      setupNamedScalar(pred_ite->thenBody(), remainder_val, "remainder");

  auto last_index = ir_builder.subExpr(extent, vector_size);
  auto threshold_val = ir_builder.subExpr(last_index, shift);
  auto threshold =
      setupNamedScalar(pred_ite->thenBody(), threshold_val, "threshold");

  auto last_root_dim_index = setupNamedScalar(
      pred_ite->thenBody(), indices.back(), "last_root_dim_index");
  auto last_root_dim_index_shift =
      ir_builder.addExpr(last_root_dim_index, shift);

  // Part A - Vectorize
  kir::Val* vectorize_pred = ir_builder.leExpr(last_root_dim_index, threshold);
  kir::IfThenElse* vectorize_ite =
      ir_builder.create<kir::IfThenElse>(vectorize_pred->as<kir::Bool>());
  cloneVectorizeLoopNests(vectorize_ite, child_loops, vector_size, true, shift);
  pred_ite->thenBody().push_back(vectorize_ite);

  // Part B - Pre
  kir::Val* lshift_pred =
      ir_builder.eqExpr(last_root_dim_index, ir_builder.zero());
  kir::IfThenElse* pre_ite =
      ir_builder.create<kir::IfThenElse>(lshift_pred->as<kir::Bool>());
  cloneVectorizeLoopNests(pre_ite, child_loops, shift, false, nullptr);
  pred_ite->thenBody().push_back(pre_ite);

  // Part C - Post
  kir::Val* lower_bound = ir_builder.gtExpr(last_root_dim_index, threshold);
  kir::Val* upper_bound = ir_builder.ltExpr(last_root_dim_index_shift, extent);
  kir::Val* rshift_pred = ir_builder.andExpr(lower_bound, upper_bound);
  kir::IfThenElse* post_ite =
      ir_builder.create<kir::IfThenElse>(rshift_pred->as<kir::Bool>());
  cloneVectorizeLoopNests(post_ite, child_loops, remainder, false, shift);
  pred_ite->thenBody().push_back(post_ite);

  return new_loop;
}

// Returns true if expr is an expression that initializes a reduction
// buffer.
bool isReductionInitExpr(const kir::Expr* expr) {
  // False if its output isn't a TensorView
  if (!ir_utils::isTVOp(expr)) {
    return false;
  }
  // False if it doesn't have any reduction axis
  const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
  if (!out_tv->domain()->hasReduction()) {
    return false;
  }
  // False if it has have TensorView inputs as initialization should
  // never use TensorViews
  const auto tv_filter_inp_view =
      ir_utils::filterByType<kir::TensorView>(expr->inputs());
  if (tv_filter_inp_view.begin() != tv_filter_inp_view.end()) {
    return false;
  }
  return true;
}

bool containsMisalignedVectorization(const kir::ForLoop* fl) {
  for (auto expr : fl->body().exprs()) {
    if (expr->isA<kir::ForLoop>()) {
      auto child_fl = expr->as<kir::ForLoop>();
      if (child_fl->iter_domain()->parallelType() ==
          ParallelType::MisalignedVectorize) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

kir::Bool* UnrollPass::getThreadPredicate(const kir::TensorView* tv) {
  // No thread predicate is needed predicate when tv is output of a
  // parallel broadcast expression.
  if (auto bop = dynamic_cast<kir::BroadcastOp*>(tv->definition())) {
    TORCH_INTERNAL_ASSERT(bop->out()->isA<kir::TensorView>());
    const auto out = bop->out()->as<kir::TensorView>()->fuserTv();
    if (ir_utils::getParallelBroadcastDomains(out, thread_predicates_).any()) {
      return kir::IrBuilder(GpuLower::current()->kernel())
          .create<kir::Bool>(true);
    }
  }
  return thread_predicates_.getExpr(tv->fuserTv());
}

void UnrollPass::handle(kir::Expr* expr) {
  if (ir_utils::isTVOp(expr)) {
    // If tv op, predicate it
    const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
    const bool should_predicate = !for_loops_.empty() ||
        out_tv->memoryType() == MemoryType::Global ||
        out_tv->memoryType() == MemoryType::Shared;
    if (!should_predicate) {
      return;
    }
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    const auto thread_pred = isReductionInitExpr(expr)
        ? ir_builder.create<kir::Bool>(true)
        : getThreadPredicate(out_tv);

    // Vectorized expressions should never use inline predicates
    kir::Bool* vectorized_pred = nullptr;
    if (std::any_of(
            for_loops_.begin(), for_loops_.end(), [](const kir::ForLoop* fl) {
              return fl->iter_domain()->parallelType() ==
                  ParallelType::Vectorize;
            })) {
      std::vector<kir::ForLoop*> outer_loops;
      kir::ForLoop* vectorized_loop = nullptr;
      for (auto loop : for_loops_) {
        if (loop->iter_domain()->parallelType() == ParallelType::Vectorize) {
          vectorized_loop = loop;
          break;
        } else {
          outer_loops.emplace_back(loop);
        }
      }
      TORCH_INTERNAL_ASSERT(
          vectorized_loop != nullptr, "Should be unreachable.");
      vectorized_pred =
          UnswitchPredicate::get(outer_loops, vectorized_loop, p2c_root_map_);
    }

    const auto pred = vectorized_pred == nullptr
        ? PredicateCompute::getInlinePredicate(expr, for_loops_, thread_pred)
        : vectorized_pred;

    TORCH_INTERNAL_ASSERT(pred != nullptr);

    // If we need a predicate, put expr inside an if then else
    if (!pred->isConst() || !(pred->isConst() && pred->value().value())) {
      non_trivial_pred_found_ = true;
      kir::IfThenElse* inline_ite = ir_builder.create<kir::IfThenElse>(pred);
      if (for_loops_.empty()) {
        // Special handling for top level output expressions that still
        // need predicates. One motivating example is a reduction op that
        // reduces to a scalar (issue #491)
        loop_replacement_map_.insert({expr, inline_ite});
      } else {
        for_loops_.back()->body().insert_before(expr, inline_ite);
        for_loops_.back()->body().erase(expr);
      }
      inline_ite->thenBody().push_back(expr);
    }
  } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
    handle(for_loop);
  }
}

// We should factor our actual predicate generation from unrolling but insering
// IR nodes "unroll_pred" or "inline_pred", then generate those later.
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  const bool is_unroll =
      fl->iter_domain()->parallelType() == ParallelType::Unroll ||
      fl->iter_domain()->parallelType() == ParallelType::Unswitch ||
      fl->iter_domain()->parallelType() == ParallelType::Vectorize;

  // If we're not looking for an unroll loop, or didn't find one, process as
  // normal.
  if (!is_unroll || !look_for_unroll_) {
    for_loops_.push_back(fl);

    // Make copy of exprs because we replace them inplace in fl
    const auto exprs_copy = fl->body().exprs();

    if (containsMisalignedVectorization(fl)) {
      auto new_fl = handleMisalignedVectorization(for_loops_, fl);
      loop_replacement_map_.insert({fl, new_fl});
      return;
    } else {
      for (auto expr : exprs_copy) {
        handle(expr);
      }
    }

    for_loops_.pop_back();
    return;
  }

  auto unroll_pred = UnswitchPredicate::get(for_loops_, fl, p2c_root_map_);

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  kir::IfThenElse* unroll_ite = ir_builder.create<kir::IfThenElse>(unroll_pred);

  // Get the loop nest for the unrolled path
  kir::ForLoop* unrolled_loop_nest = cloneLoopNest(fl, true);

  unroll_ite->thenBody().push_back(unrolled_loop_nest);
  if (fl->iter_domain()->parallelType() == ParallelType::Vectorize) {
    loop_replacement_map_.insert({fl, unroll_ite});
    return;
  }

  // Loop nest for inlined path
  kir::ForLoop* inlined_loop = cloneLoopNest(fl);

  // Add inline predicates for inlined loop nest
  look_for_unroll_ = false;
  non_trivial_pred_found_ = false;
  handle(inlined_loop);
  look_for_unroll_ = true;
  if (!non_trivial_pred_found_) {
    loop_replacement_map_.insert({fl, inlined_loop});
  } else {
    if (!canOmitElseClause(fl)) {
      unroll_ite->elseBody().push_back(inlined_loop);
    }
    loop_replacement_map_.insert({fl, unroll_ite});
  }
}

bool UnrollPass::canOmitElseClause(kir::ForLoop* fl) const {
  kir::ExpressionEvaluator eval;
  std::vector<kir::ForLoop*> loops({fl});
  while (loops.size() > 0) {
    auto loop = loops.back();
    loops.pop_back();
    auto id = loop->iter_domain();
    if (id->isThread() || id->parallelType() == ParallelType::Vectorize) {
      continue;
    }
    const auto result = eval.evaluate(id->rawExtent());
    if (!(result.has_value() && result.value() == 1)) {
      return false;
    }
    for (auto nested_loop :
         ir_utils::filterByType<kir::ForLoop>(loop->body().exprs())) {
      loops.push_back(nested_loop);
    }
  }
  return true;
}

// Generate the loop nest structure and place it in lowered_exprs
void UnrollPass::computeMap(const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("UnrollPass::computeMap");

  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    handle(expr);
  }
}

// TODO(kir): incorporate this into a new Scope interface
kir::Expr* UnrollPass::applyReplacements(kir::Expr* expr) const {
  auto handle_scope = [this](kir::Scope& scope) {
    for (size_t i = 0; i < scope.size(); ++i) {
      scope[i] = applyReplacements(scope[i]);
    }
  };

  const auto it = loop_replacement_map_.find(expr);
  if (it != loop_replacement_map_.end()) {
    return it->second;
  } else {
    if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle_scope(for_loop->body());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle_scope(ite->thenBody());
      handle_scope(ite->elseBody());
    }
    return expr;
  }
}

std::vector<kir::Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs,
    const ThreadPredicateMap& thread_predicates) {
  FUSER_PERF_SCOPE("UnrollPass::runPass");

  UnrollPass unroll_pass(fusion, thread_predicates);
  unroll_pass.computeMap(exprs);

  std::vector<kir::Expr*> mutated_exprs;
  mutated_exprs.reserve(exprs.size());
  for (auto expr : exprs) {
    mutated_exprs.push_back(unroll_pass.applyReplacements(expr));
  }

  return mutated_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
