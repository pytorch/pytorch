#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ConditionalFromPredicateModifier : public kir::IrVisitor {
 public:
  ConditionalFromPredicateModifier() = delete;

  static std::vector<Expr*> fillPredicates(const std::vector<Expr*>& exprs) {
    ConditionalFromPredicateModifier cfpm(exprs);
    return cfpm.exprs_;
  }

 private:
  ConditionalFromPredicateModifier(const std::vector<Expr*>& exprs) {
    FUSER_PERF_SCOPE(
        "GpuLower::Lower::ConditionalFromPredicateModifier::process");
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    if (expr != nullptr && expr->predicate() != nullptr) {
      // Replace expr predicate with bool conditional
      auto conditional = generateConditional(expr->predicate());
      TORCH_INTERNAL_ASSERT(conditional != nullptr);
      expr->predicate()->setValue(conditional);
      TORCH_INTERNAL_ASSERT(expr->predicate()->value() != nullptr);
      setWritePredicate(expr, conditional);
    }

    kir::IrVisitor::handle(expr);
  }

  void setWritePredicate(Expr* expr, Bool* read_cond) {
    if (expr->writePredicate() != nullptr) {
      auto write_cond = generateConditional(expr->writePredicate());
      if (write_cond) {
        expr->writePredicate()->setValue(write_cond);
      } else {
        // If generateConditional returns null, it means no specific
        // predicate needs to be used.
        expr->setWritePredicate(nullptr);
      }
    }
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(ite->predicate() != nullptr);

    // If ite already has Bool conditional, handle internal expressions
    // Otherwise, generate conditional and update predicate
    if (!ite->predicate()->hasValue()) {
      auto conditional = generateConditional(ite->predicate());
      TORCH_INTERNAL_ASSERT(conditional != nullptr);
      TORCH_INTERNAL_ASSERT(conditional->isA<Bool>());

      // Update bool conditional in-place
      ite->predicate()->setValue(conditional);
      TORCH_INTERNAL_ASSERT(ite->predicate()->value() != nullptr);
    }
    kir::IrVisitor::handle(ite);
  }

  // Generate conditional according to PredicateType
  Bool* generateConditional(kir::Predicate* pred) {
    switch (pred->predicate_type()) {
      case PredicateType::Inline:
      case PredicateType::ReductionWrite:
      case PredicateType::Misaligned:
      case PredicateType::Shift:
      case PredicateType::Padding: {
        return PredicateCompute::getInlinePredicate(
            pred->expr(),
            for_loops_,
            pred->thread_pred(),
            pred->predicate_type());
      }
      case PredicateType::Vectorize: {
        std::vector<kir::ForLoop*> outer_loops;
        kir::ForLoop* vectorized_loop = nullptr;
        for (auto loop : for_loops_) {
          if (loop->iter_domain()->getParallelType() ==
              ParallelType::Vectorize) {
            vectorized_loop = loop;
            break;
          } else {
            outer_loops.emplace_back(loop);
          }
        }
        TORCH_INTERNAL_ASSERT(
            vectorized_loop != nullptr, "Should be unreachable.");
        return UnswitchPredicate::get(outer_loops, vectorized_loop);
      }
      case PredicateType::Unswitch: {
        return UnswitchPredicate::get(for_loops_, pred->unrolled_loop());
      }
      case PredicateType::Manual: {
        return pred->value();
      }
      default:
        break;
    }
    return nullptr;
  }
};

void assertOnWarpOps(const Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      !expr->isA<MmaOp>(),
      "Mma op: cannot eliminate predicate for mma op, tiling not valid");
}

} // namespace

std::vector<Expr*> generateConditionalFromPredicate(
    const std::vector<Expr*>& exprs) {
  return ConditionalFromPredicateModifier::fillPredicates(exprs);
}

namespace {

class PredicateAnalyzer : public OptOutDispatch {
 public:
  //! Checks if a predicate is needed to avoid out-of-bound accesses.
  //!
  //! Due to the way we allocate local-memory tensors, there should
  //! never be out-of-bound accesses with consumer tensors when allocated on
  //! local memory. However, accessing producer tensors still may
  //! result in out-of-bound as they are replayed as consumers.
  static bool needsPredicate(TensorView* producer, TensorView* consumer) {
    // Both tensors must be on local memory. Global tensors must be
    // predicated as allocation is done based on root domains. Smem
    // and local tensors are allocated based on leaf domains, however,
    // smem tensors are parallelized, which is highly likely, the size
    // of the parallelized axis is the actual size of the axis, not
    // the number of threads. Since the number of threads can be
    // larger than the axis size, it's not safe to skip predication

    // Check that parallel dimension will not generate out of bound index
    if (!(producer->getMemoryType() == MemoryType::Local &&
          consumer->getMemoryType() == MemoryType::Local)) {
      return true;
    }

    auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
    auto c2p =
        BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_map)
            .getReplay();

    PredicateAnalyzer analyzer(c2p);

    for (auto id : consumer->domain()->domain()) {
      if (analyzer.needsPredicate(id)) {
        return true;
      }
    }

    return false;
  }

 private:
  PredicateAnalyzer(const std::unordered_map<IterDomain*, IterDomain*>& c2p_map)
      : c2p_map_(c2p_map) {}

  // Returns true if no out-of-bound accesses could occur with a
  // producer
  bool needsPredicate(IterDomain* consumer_id) {
    needs_predicate_ = false;
    handle(consumer_id);
    return needs_predicate_;
  }

  void handle(IterDomain* consumer_id) override {
    // The traversal should have ended if needs_predicate_ was true
    TORCH_INTERNAL_ASSERT(!needs_predicate_);

    // If consumer_id is not going to be materialized as a loop (e.g.,
    // broadcast), no need to predicate
    if (consumer_id->isBroadcast() ||
        GpuLower::current()->trivialReductionInfo().isDerived(consumer_id)) {
      return;
    }

    // If the producer has a matching domain, it should not cause
    // out-of-bound accesses
    if (c2p_map_.find(consumer_id) != c2p_map_.end()) {
      return;
    }

    // If no definition exists, stop traversing
    if (consumer_id->definition() == nullptr) {
      return;
    }

    OptOutDispatch::handle(consumer_id->definition());
  }

  // If it splits the input axis evenly, proceeds to check the input
  // axis. Otherwise, we can't skip predication as it might cause
  // out-bound accesses with the producer tensor
  void handle(Split* split) override {
    auto factor = split->factor()->getInt();
    if (!factor.has_value()) {
      needs_predicate_ = true;
      return;
    }

    ExpressionEvaluator ee(split->fusion());
    const auto in_extent = ee.evaluate(split->in()->extent());

    if (!in_extent.has_value() || ((in_extent.value() % factor.value()) != 0)) {
      needs_predicate_ = true;
      return;
    }

    handle(split->in());
  }

  void handle(Merge* merge) override {
    handle(merge->inner());
    if (needs_predicate_) {
      return;
    }
    handle(merge->outer());
  }

 private:
  //! BestEffort map from consumer IDs to producer IDs
  const std::unordered_map<IterDomain*, IterDomain*>& c2p_map_;
  bool needs_predicate_ = false;
};

} // namespace

bool PredicateElimination::needsPredicate(Expr* expr) const {
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }

  std::vector<std::function<bool(Expr*)>> filters;

  // Always predicate integer division and related ops as we don't
  // know what values are in the out-of-bound region and they may
  // cause exceptions
  filters.emplace_back([](Expr* expr) {
    auto dt = expr->outputs()[0]->getDataType().value();
    return (
        (dt == DataType::Int || dt == DataType::Int32) &&
        expr->isA<BinaryOp>() &&
        (expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Div ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Mod ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Remainder ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::CeilDiv));
  });

  // Skip if MisalignedVectorize is involved for now. This could be
  // relaxed.
  filters.emplace_back([](Expr* expr) {
    std::vector<const std::vector<Val*>*> inputs_and_outputs = {
        &(expr->inputs()), &(expr->outputs())};
    for (const auto& inputs_or_outputs : inputs_and_outputs) {
      for (auto tv : ir_utils::filterByType<TensorView>(*inputs_or_outputs)) {
        if (std::any_of(
                tv->domain()->domain().begin(),
                tv->domain()->domain().end(),
                [](IterDomain* axis) {
                  return axis->getParallelType() ==
                      ParallelType::MisalignedVectorize;
                })) {
          return true;
        }
      }
    }
    return false;
  });

  // Shift is not supported yet.
  filters.emplace_back([](Expr* expr) {
    auto& halo_info = GpuLower::current()->haloInfo();
    auto input_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
    return halo_info.needsShiftPredicate(expr) ||
        std::any_of(input_tvs.begin(), input_tvs.end(), [&](auto input_tv) {
             return input_tv->definition() != nullptr &&
                 halo_info.needsShiftPredicate(input_tv->definition());
           });
  });

  // Predicates the expression if any producer-consumer pair of the
  // expression needs to be predicated
  filters.emplace_back([](Expr* expr) {
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (PredicateAnalyzer::needsPredicate(input, output)) {
          return true;
        }
      }
    }
    return false;
  });

  // Predicates Welford ops
  filters.emplace_back([](Expr* expr) { return expr->isA<WelfordOp>(); });

  // If this is a reduction, and if we omit the predicate for the
  // input, the input may have a garbabe value, which must not be used
  // for this reduction. However, if the input is also an output of
  // another reduction with the same binary op, which is a common
  // pattern with rfactor, the input should be safe to use with no
  // predication.
  filters.emplace_back([this](Expr* expr) {
    if (expr->isA<ReductionOp>()) {
      auto input = expr->inputs()[0]->as<TensorView>();
      auto input_def = input->definition();
      // When input_def is null, input must be an input to the fusion,
      // so that must be allocated on global memory. Since we don't omit
      // predication for expressions involving global memory, this
      // should never occur.
      TORCH_INTERNAL_ASSERT(
          input_def != nullptr, "Inconsistent input found: ", input);

      if (non_predicated_exprs_.find(input_def) !=
              non_predicated_exprs_.end() &&
          !(input_def->isA<ReductionOp>() &&
            (expr->as<ReductionOp>()->getReductionOpType() ==
             input_def->as<ReductionOp>()->getReductionOpType()))) {
        return true;
      }
    }
    return false;
  });

  // If any of the filters returns true, predicate must be used.
  return std::any_of(filters.begin(), filters.end(), [expr](auto filter) {
    return filter(expr);
  });
}

void PredicateElimination::handle(Expr* expr) {
  if (!ir_utils::isTvOp(expr)) {
    return;
  }

  if (needsPredicate(expr)) {
    // Warp primitives are currently limited to un-predicated usage,
    //   predicating these ops will require extra steps to ensure that
    //   the whole warp will get the same value.
    assertOnWarpOps(expr);
    return;
  }

  non_predicated_exprs_.insert(expr);

  // Ensure all inputs have some values set at the out-of-bound
  // regions
  for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    auto input_def = input->definition();
    // When input_def is null, input must be an input to the fusion,
    // so that must be allocated on global memory. Since we don't omit
    // predication for expressions involving global memory, this
    // should never occur.
    std::stringstream ss;
    ss << input;
    TORCH_INTERNAL_ASSERT(
        input_def != nullptr, "Inconsistent input found: ", ss.str());

    // If input is an output of reduction, it should be fully
    // initialied as it's allocated on local memory.
    if (input_def->isA<ReductionOp>() || input_def->isA<WelfordOp>()) {
      continue;
    }

    // If this expr is reduction, always initilize the input with the
    // default value. NOTE: This can be done more
    // intelligently. A garbage value can only cause a problem when
    // it's reduced with non-garbage values, so if the non-reduction
    // axes do not have any garbage, it should be just fine without
    // explicit initialization. However, initialization cost should be
    // cheap, so that further optimization should not make a large
    // difference.
    if (expr->isA<ReductionOp>()) {
      setReductionInitValue(input, expr->as<ReductionOp>()->init());
      continue;
    }

    if (expr->isA<MmaOp>()) {
      setReductionInitValue(input, expr->as<MmaOp>()->init());
      continue;
    }

    // If an input does not need a predicate either, then it should
    // have some value, so no need to set a default value
    if (non_predicated_exprs_.find(input_def) != non_predicated_exprs_.end()) {
      continue;
    }

    // Make sure input is initialized
    setDefaultInitValue(input);
  }
}

bool PredicateElimination::setDefaultInitValue(TensorView* tv) {
  auto it = init_value_map_.find(tv);
  // If there's already a mapping for tv, it should be mapped to a
  // zero val or a reduction init. Either case, no need to modify
  // the existing mapping.
  if (it == init_value_map_.end()) {
    init_value_map_.insert({tv, nullptr});
  }
  return true;
}

bool PredicateElimination::setReductionInitValue(
    TensorView* tv,
    Val* reduction_init) {
  auto it = init_value_map_.find(tv);
  if (it == init_value_map_.end()) {
    init_value_map_.insert({tv, reduction_init});
    return true;
  }

  auto existing_val = it->second;
  if (existing_val == nullptr) {
    // If the existing mapping returns nullptr, it means that a
    // default init was set before. Overwrite with the reduction
    // init val.
    init_value_map_[tv] = reduction_init;
    return true;
  } else if (existing_val->sameAs(reduction_init)) {
    return true;
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Incosistent setting of initialization value for t",
        tv->name(),
        ". Prev: ",
        existing_val,
        ", New: ",
        reduction_init);
    return false;
  }
}

bool PredicateElimination::canOmitPredicate(const Expr* expr) const {
  TORCH_INTERNAL_ASSERT(expr != nullptr);
  const auto out_tv = ir_utils::getTvOutput(expr);
  TORCH_INTERNAL_ASSERT(out_tv != nullptr, "Not a tensor expression");
  // No need to predicate local tensors to which a scalar is assigned
  if (out_tv->getMemoryType() == MemoryType::Local) {
    if (auto uop = dynamic_cast<const UnaryOp*>(expr)) {
      if (uop->getUnaryOpType() == UnaryOpType::Set && uop->in()->isScalar()) {
        return true;
      }
    }
  }
  if (non_predicated_exprs_.find(expr) != non_predicated_exprs_.end()) {
    return true;
  }

  return false;
}

Val* PredicateElimination::getInitValue(TensorView* tv) const {
  auto it = init_value_map_.find(tv);
  if (it == init_value_map_.end()) {
    return nullptr;
  }
  auto init_val = it->second;
  if (init_val == nullptr) {
    // No reduction restriction. Just use zero
    return GpuLower::current()->kernel()->zeroVal();
  } else {
    return init_val;
  }
}

void PredicateElimination::build(Fusion* fusion) {
  traverseFrom(fusion, fusion->outputs());
}

std::string PredicateElimination::toString() const {
  std::stringstream ss;
  ss << "Tensors that do not need predication:";
  for (auto expr : non_predicated_exprs_) {
    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(out->isA<TensorView>());
      ss << " T" << out->name();
    }
  }
  ss << "\n";
  ss << "Init values:";
  for (auto kv : init_value_map_) {
    ss << " T" << kv.first->name() << "->";
    if (kv.second == nullptr) {
      ss << "<default(0)>";
    } else {
      ss << kv.second;
    }
  }
  ss << "\n";
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
