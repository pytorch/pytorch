#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ConditionalFromPredicateModifier {
 public:
  ConditionalFromPredicateModifier(Fusion* fusion) {
    p2c_root_map_ = loop_utils::p2cRootMap(fusion->exprs());
  }

  void process(const std::vector<kir::Expr*>& exprs) {
    FUSER_PERF_SCOPE("ConditionalFromPredicateModifier::process");
    for (auto* expr : exprs) {
      handle(expr);
    }
  }

  const std::unordered_map<kir::Expr*, kir::Expr*>& replacementMap() const {
    return expr_replacement_map_;
  }

 private:
  void handle(kir::Expr* expr) {
    if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (expr != nullptr && expr->predicate() != nullptr) {
      // Replace expr predicate with bool conditional
      auto conditional = generateConditional(expr->predicate());
      expr->predicate()->setValue(conditional);
      TORCH_INTERNAL_ASSERT(expr->predicate()->value() != nullptr);
    }
  }

  void handle(kir::ForLoop* fl) {
    for_loops_structure_.push_back(fl);

    const auto exprs_copy = fl->body().exprs();
    for (auto expr : exprs_copy) {
      handle(expr);
    }

    for_loops_structure_.pop_back();
  }

  void handle(kir::IfThenElse* ite) {
    TORCH_INTERNAL_ASSERT(ite->predicate() != nullptr);

    // If ite already has Bool conditional, handle internal expressions
    // Otherwise, generate conditional and update predicate
    if (ite->predicate()->hasValue()) {
      const auto then_exprs_copy = ite->thenBody().exprs();
      for (auto expr : then_exprs_copy) {
        handle(expr);
      }

      const auto else_exprs_copy = ite->elseBody().exprs();
      for (auto expr : else_exprs_copy) {
        handle(expr);
      }
    } else {
      auto conditional = generateConditional(ite->predicate());
      TORCH_INTERNAL_ASSERT(conditional != nullptr);
      TORCH_INTERNAL_ASSERT(conditional->isA<kir::Bool>());

      // Update bool conditional in-place
      ite->predicate()->setValue(conditional);
      handle(ite);
      TORCH_INTERNAL_ASSERT(ite->predicate()->value() != nullptr);
    }
  }

  // Generate conditional according to PredicateType
  kir::Bool* generateConditional(kir::Predicate* pred) {
    switch (pred->predicate_type()) {
      case PredicateType::Inline:
      case PredicateType::Misaligned: {
        return PredicateCompute::getInlinePredicate(
            pred->expr(),
            for_loops_structure_,
            pred->thread_pred(),
            pred->predicate_type());
      }
      case PredicateType::Vectorize: {
        std::vector<kir::ForLoop*> outer_loops;
        kir::ForLoop* vectorized_loop = nullptr;
        for (auto loop : for_loops_structure_) {
          if (loop->iter_domain()->parallelType() == ParallelType::Vectorize) {
            vectorized_loop = loop;
            break;
          } else {
            outer_loops.emplace_back(loop);
          }
        }
        TORCH_INTERNAL_ASSERT(
            vectorized_loop != nullptr, "Should be unreachable.");
        return UnswitchPredicate::get(
            outer_loops, vectorized_loop, p2c_root_map_);
      }
      case PredicateType::Unswitch: {
        return UnswitchPredicate::get(
            for_loops_structure_, pred->unrolled_loop(), p2c_root_map_);
      }
      case PredicateType::Shift: {
        kir::TensorView* out_tv = ir_utils::getTVOutput(pred->expr());
        TORCH_INTERNAL_ASSERT(
            out_tv != nullptr, "Missing kir::TensorView output");
        return ShiftPredicateInserter::getPredicate(
            pred->expr(),
            for_loops_structure_,
            out_tv,
            pred->thread_pred(),
            true);
      }
      case PredicateType::Padding: {
        kir::TensorView* out_tv = ir_utils::getTVOutput(pred->expr());
        TORCH_INTERNAL_ASSERT(
            out_tv != nullptr, "Missing kir::TensorView output");
        return ShiftPredicateInserter::getPredicate(
            pred->expr(),
            for_loops_structure_,
            out_tv,
            pred->thread_pred(),
            false);
      }
      case PredicateType::Manual: {
        TORCH_INTERNAL_ASSERT(
            false,
            "Predicate generation is not required for PredicateType::Manual");
      }
      default:
        break;
    }
    return nullptr;
  }

 private:
  // We will track which loops in the incoming IR will be replaced and by what
  std::unordered_map<kir::Expr*, kir::Expr*> expr_replacement_map_;

  // A depth-first ordering of nested for loops
  // It is used for indexing and predicate generation
  std::vector<kir::ForLoop*> for_loops_structure_;

  IterDomainMap p2c_root_map_;
};

} // namespace

std::vector<kir::Expr*> generateConditionalFromPredicate(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("generateConditionalFromPredicate");

  ConditionalFromPredicateModifier p2cm(fusion);
  p2cm.process(exprs);

  std::vector<kir::Expr*> mutated_exprs;
  mutated_exprs.reserve(exprs.size());
  for (auto expr : exprs) {
    mutated_exprs.push_back(
        ir_utils::applyReplacements(p2cm.replacementMap(), expr));
  }

  return mutated_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
