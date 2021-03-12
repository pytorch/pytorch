#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <algorithm>

// TODO: refactor this file (one per namespace)

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace scope_utils {

std::vector<kir::ForLoop*> getLoops(kir::Expr* scope) {
  std::vector<kir::ForLoop*> loops;
  while (scope != nullptr) {
    if (auto loop = dynamic_cast<kir::ForLoop*>(scope)) {
      loops.push_back(loop);
    }
    scope = scope->parentScope();
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

void insertBefore(kir::Expr* scope, kir::Expr* ref, kir::Expr* expr) {
  if (auto ite = dynamic_cast<kir::IfThenElse*>(scope)) {
    ite->thenBody().insert_before(ref, expr);
  } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(scope)) {
    for_loop->body().insert_before(ref, expr);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unexpected scope expression");
  }
}

} // namespace scope_utils

namespace ir_utils {

TVDomainGuard::TVDomainGuard(TensorView* _tv, TensorDomain* td)
    : tv_(_tv), prev_domain(tv_->domain()) {
  tv_->setDomain(td);
}

TVDomainGuard::~TVDomainGuard() {
  tv_->setDomain(prev_domain);
}

std::vector<IterDomain*> iterDomainInputsOf(
    const std::vector<IterDomain*>& input_ids) {
  auto inputs = IterVisitor::getInputsTo({input_ids.begin(), input_ids.end()});
  std::vector<IterDomain*> id_inputs(
      ir_utils::filterByType<IterDomain>(inputs).begin(),
      ir_utils::filterByType<IterDomain>(inputs).end());
  return id_inputs;
}

std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order) {
  auto inputs_vec = iterDomainInputsOf(of);

  std::unordered_set<IterDomain*> inputs_set(
      inputs_vec.begin(), inputs_vec.end());

  std::vector<IterDomain*> ordered_inputs;
  std::copy_if(
      order.begin(),
      order.end(),
      std::back_inserter(ordered_inputs),
      [&inputs_set](const auto& id) {
        return inputs_set.find(id) != inputs_set.end();
      });

  return ordered_inputs;
}

bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* expr) {
  if (expr->outputs().size() == 1 && isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::ReductionOp ||
       expr->getExprType().value() == ExprType::BroadcastOp ||
       expr->getExprType().value() == ExprType::TransposeOp)) {
    return true;
  }
  if (expr->getExprType().value() == ExprType::WelfordOp) {
    return true;
  }
  return false;
}

bool isTVOp(const kir::Expr* expr) {
  const auto& outputs = expr->outputs();
  return outputs.size() >= 1 && outputs[0]->isA<kir::TensorView>();
}

// TODO: why do we assume there's a single TV output?
TensorView* getTVOutput(const Expr* expr) {
  for (auto out : expr->outputs()) {
    if (out->getValType().value() == ValType::TensorView) {
      return out->as<TensorView>();
    }
  }
  return nullptr;
}

bool isScalarOp(const Expr* expr) {
  for (auto out : expr->outputs())
    if (!out->isScalar())
      return false;
  return true;
}

Expr* asExpr(Statement* stmt) {
  TORCH_INTERNAL_ASSERT(stmt->isExpr());
  return stmt->as<Expr>();
}

TensorView* asTV(Val* val) {
  TORCH_INTERNAL_ASSERT(isTV(val));
  return val->as<TensorView>();
}

ParallelTypeBitmap getParallelBroadcastDomains(
    const TensorView* tv,
    const ThreadPredicateMap& preds) {
  // If no pred is found for tv, no predicate is necessary
  if (preds.find(tv) == preds.end()) {
    return ParallelTypeBitmap();
  }

  ParallelTypeBitmap parallel_broadcast;

  const auto& iter_domains = tv->domain()->domain();

  // If the output is on shared memory, assume that all subsequent
  // reads from all threads in its CTA can be done with no parallel
  // broadcast. Only one thread will write to shared memory followed
  // by a proper _syncthreads.
  const bool output_smem = tv->getMemoryType() == MemoryType::Shared;

  for (auto id : iter_domains) {
    if (!id->isBroadcast()) {
      continue;
    }
    if (id->isBlockDim() || (!output_smem && id->isThreadDim())) {
      parallel_broadcast.set(id->getParallelType(), true);
    }
  }

  return parallel_broadcast & preds.at(tv).pred;
}

} // namespace ir_utils

namespace loop_utils {

// TODO: Clean this up, Naoya added a mechanism we should be able to reuse.
std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    bool use_id_map) {
  const auto gpu_lower = GpuLower::current();

  // If in global memory, it can be all the way outside the loops.
  if (tv->getMemoryType() == MemoryType::Global) {
    return {nullptr, 0};
  }

  // Figure out where we want to place alloc/reduction initialization. We want
  // outside an unroll loop, or inside our computeAt point.
  kir::ForLoop* alloc_loop = nullptr;

  auto loops_it = loops.begin();
  // Look at each axis individually in out's domain
  for (int64_t tv_i = 0; tv_i < (int64_t)tv->getComputeAtPosition(); tv_i++) {
    // Grab the axis ID

    auto local_id = tv->axis(tv_i);
    if (use_id_map) {
      auto id_it = id_map.find(local_id);
      if (id_it != id_map.end()) {
        local_id = id_it->second;
      }
    }

    if (gpu_lower->trivialReductionInfo().isDerivedFromRoot(local_id)) {
      continue;
    }

    auto lowered_local_id =
        gpu_lower->lowerValue(local_id)->as<kir::IterDomain>();
    loops_it = std::find_if(
        loops_it, loops.end(), [&lowered_local_id](const auto& loop) {
          return GpuLower::current()->caLoopMap().areMapped(
                     lowered_local_id, loop->iter_domain()) ||
              loop->iter_domain()->parallelType() == ParallelType::Unroll;
        });

    TORCH_INTERNAL_ASSERT(
        loops_it != loops.end(),
        "Could not find all required axes for indexing when trying to index into ",
        tv);
    if ((*loops_it)->iter_domain()->parallelType() == ParallelType::Unroll) {
      return {alloc_loop, tv_i};
    }

    alloc_loop = *loops_it;
    ++loops_it;
  }

  return {alloc_loop, (int64_t)tv->getComputeAtPosition()};
}

std::pair<kir::ForLoop*, int64_t> getAllocPoint(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops) {
  return getAllocPoint(tv, loops, {}, false);
}

IterDomainMap p2cRootMap(const std::vector<Expr*>& exprs) {
  IterDomainMap p2c_root_map;

  const auto gpu_lower = GpuLower::current();

  for (auto expr : exprs) {
    auto out_tv = ir_utils::getTVOutput(expr);
    for (auto in_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      const auto root_p2c =
          PairwiseRootDomainMap(in_tv, out_tv)
              .mapProducerToConsumer(in_tv->domain(), out_tv->domain());
      for (auto entry : root_p2c) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        // Careful we don't allow circular references
        if (p_id != c_id) {
          const auto kir_p_id =
              gpu_lower->lowerValue(p_id)->as<kir::IterDomain>();
          const auto kir_c_id =
              gpu_lower->lowerValue(c_id)->as<kir::IterDomain>();
          p2c_root_map[kir_p_id] = kir_c_id;
        }
      }
    }
  }

  return p2c_root_map;
}

} // namespace loop_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
