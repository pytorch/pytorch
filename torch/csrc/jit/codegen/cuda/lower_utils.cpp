#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
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

//! Create an **empty** Forloop and copy the metadata.
kir::ForLoop* cloneForLoop(kir::ForLoop* for_loop) {
  return IrBuilder::create<kir::ForLoop>(for_loop);
}

//! Create an **empty** IfThenElse and copy the metadata.
kir::IfThenElse* cloneIfThenElse(kir::IfThenElse* ite) {
  return IrBuilder::create<kir::IfThenElse>(ite->predicate());
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
    const std::vector<IterDomain*>& input_ids,
    const std::vector<IterDomain*>& all_inputs) {
  auto inputs = IterVisitor::getInputsTo(
      {input_ids.begin(), input_ids.end()},
      {all_inputs.begin(), all_inputs.end()});
  std::vector<IterDomain*> id_inputs(
      ir_utils::filterByType<IterDomain>(inputs).begin(),
      ir_utils::filterByType<IterDomain>(inputs).end());
  return id_inputs;
}

std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order) {
  auto inputs_vec = iterDomainInputsOf(of, order);

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
  return val->getValType().value() == ValType::TensorView ||
      val->getValType().value() == ValType::TensorIndex;
}

// Check if we're a TensorView op that we can generate code for.
bool isTvOp(const Expr* expr) {
  if (std::any_of(
          expr->outputs().begin(),
          expr->outputs().end(),
          [](Val* v) { return isTV(v); }) &&
      (expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp ||
       expr->getExprType().value() == ExprType::ReductionOp ||
       expr->getExprType().value() == ExprType::WelfordOp ||
       expr->getExprType().value() == ExprType::MmaOp ||
       expr->getExprType().value() == ExprType::BroadcastOp ||
       expr->getExprType().value() == ExprType::TransposeOp ||
       expr->getExprType().value() == ExprType::ShiftOp ||
       expr->getExprType().value() == ExprType::GatherOp ||
       expr->getExprType().value() == ExprType::ViewDtypeOp ||
       expr->getExprType().value() == ExprType::ViewOp ||
       expr->getExprType().value() == ExprType::GridReduction ||
       expr->getExprType().value() == ExprType::GridBroadcast ||
       expr->getExprType().value() == ExprType::GridWelford)) {
    return true;
  }
  return false;
}

TensorView* getTv(Val* val) {
  if (val->isA<TensorView>()) {
    return val->as<TensorView>();
  } else if (val->isA<kir::TensorIndex>()) {
    return val->as<kir::TensorIndex>()->view();
  }
  return nullptr;
}

std::vector<TensorView*> getTvs(const std::vector<Val*>& vals) {
  std::vector<TensorView*> tvs;
  for (auto val : vals) {
    auto tv = ir_utils::getTv(val);
    if (tv) {
      tvs.emplace_back(tv);
    }
  }
  return tvs;
}

TensorView* getTvOutput(const Expr* expr) {
  for (auto out : expr->outputs()) {
    if (auto tv = getTv(out)) {
      return tv;
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

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map) {
  if (!isTvOp(expr)) {
    return false;
  }

  if (!(expr->isA<ReductionOp>() || expr->isA<BroadcastOp>() ||
        expr->isA<WelfordOp>() || expr->isA<kir::GridReduction>() ||
        expr->isA<kir::GridBroadcast>() || expr->isA<kir::GridWelford>())) {
    return false;
  }

  auto tv = getTvOutput(expr);

  if (tv->hasBlockReduction() || tv->hasGridReduction()) {
    return true;
  } else if (expr->isA<BroadcastOp>()) {
    const ParallelTypeBitmap pt_map =
        GpuLower::current()->threadPredMap().getParallelBroadcastDomains(tv);
    return pt_map.any();
  }

  return false;
}

c10::optional<IterDomain*> getMaybeWarpReductionDim(const ReductionOp* node) {
  auto tv_out = getTv(node->out());
  if (tv_out == nullptr) {
    return c10::nullopt;
  }

  auto tv_in = getTv(node->in());

  // only support reducing to registers for now.
  if (tv_in->getMemoryType() != MemoryType::Local ||
      tv_out->getMemoryType() != MemoryType::Local) {
    return c10::nullopt;
  }

  IterDomain* reduction_on_xdim = nullptr;
  for (auto id : tv_out->domain()->domain()) {
    // Currently warp reduction only allows
    //  serial and block.x parallel reductions
    if (id->isReduction() && id->isParallelized()) {
      if (id->getParallelType() == ParallelType::TIDx) {
        reduction_on_xdim = id;
      } else if (id->isThread()) {
        return c10::nullopt;
      }
    }
  }
  if (!reduction_on_xdim) {
    return c10::nullopt;
  }

  if (!reduction_on_xdim->start()->isZeroInt()) {
    return c10::nullopt;
  }

  if (reduction_on_xdim->hasPaddingToMultipleOfWarp()) {
    return c10::optional<IterDomain*>(reduction_on_xdim);
  }

  if (reduction_on_xdim->extent()->isConst()) {
    auto extent_value = reduction_on_xdim->extent()->getInt().value();
    if (extent_value % at::cuda::warp_size() == 0) {
      return c10::optional<IterDomain*>(reduction_on_xdim);
    }
  }

  return c10::nullopt;
}

bool derivedFromRootCAAxes(const TensorView* tv, IterDomain* axis) {
  std::vector<IterDomain*> ca_axes(
      tv->domain()->domain().begin(),
      tv->domain()->domain().begin() + tv->getComputeAtPosition());

  auto ca_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(ca_axes.begin(), ca_axes.end()));

  auto root_vals = IterVisitor::getInputsTo({axis});

  return std::any_of(
      root_vals.begin(), root_vals.end(), [&ca_root_vals](auto root) {
        return std::find(ca_root_vals.begin(), ca_root_vals.end(), root) !=
            ca_root_vals.end();
      });
}

std::unordered_map<ParallelType, IterDomain*, TypeHash> getParallelDomains(
    Val* val) {
  TensorView* tv = nullptr;
  if (val->isA<TensorView>()) {
    tv = val->as<TensorView>();
  } else if (val->isA<kir::TensorIndex>()) {
    tv = val->as<kir::TensorIndex>()->view();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Provided val is not TensorIndex or TensorView.");
  }

  std::unordered_map<ParallelType, IterDomain*, TypeHash> parallel_domains;
  for (auto d : tv->domain()->domain()) {
    if (d->isThread()) {
      parallel_domains.insert(std::make_pair(d->getParallelType(), d));
    }
  }
  return parallel_domains;
}

} // namespace ir_utils

namespace loop_utils {

BasicAllocInfo getAllocInformation(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& for_loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    bool use_id_map) {
  BasicAllocInfo info;
  auto gpu_lower = GpuLower::current();
  const auto& loop_map = gpu_lower->caLoopMap();

  bool outer_alloc_found = false;

  for (auto fl : for_loops) {
    if (info.alloc_pos == tv->getComputeAtPosition()) {
      break;
    }

    if (tv->axis(info.alloc_pos)->isReduction()) {
      const auto outputs = FusionGuard::getCurFusion()->getTerminatingOutputs();
      TORCH_INTERNAL_ASSERT(
          std::find(outputs.begin(), outputs.end(), tv) != outputs.end(),
          "Invalid computeAt of T",
          tv->name(),
          ". A reducation axis is detected outside computeAt point even though it is not an output tensor.");
      break;
    }

    auto fl_id = fl->iter_domain();

    if (fl_id->getParallelType() == ParallelType::Unroll) {
      break;
    }

    // Shared memory must be allocated outside of unswitched
    // domains. See issue #1133.
    if (fl_id->getParallelType() == ParallelType::Unswitch &&
        tv->getMemoryType() == MemoryType::Shared) {
      outer_alloc_found = true;
    }

    // Assume global memory is allocated at outer most scope.
    if (tv->getMemoryType() == MemoryType::Global) {
      outer_alloc_found = true;
    }

    // Allocation of a double buffered tensor is placed outside its
    // double buffer axis.
    if (tv->isDoubleBuffered() &&
        tv->axis(info.alloc_pos) ==
            gpu_lower->doubleBufferInfo().getDoubleBufferAxis(tv)) {
      outer_alloc_found = true;
    }

    auto local_id = tv->axis(info.alloc_pos);

    if (use_id_map) {
      auto id_it = id_map.find(local_id);
      if (id_it != id_map.end()) {
        local_id = id_it->second;
      }
    }

    if (loop_map.areMapped(local_id, fl_id)) {
      info.alloc_pos++;
    }

    info.init_for_loop = fl;

    if (!outer_alloc_found) {
      info.alloc_for_loop = fl;
    }
  }

  return info;
}

} // namespace loop_utils

namespace {

class ReplaceExprInput : private kir::ExprMutator {
 public:
  static std::vector<Expr*> replace(
      const std::vector<Expr*>& exprs,
      const std::unordered_map<Val*, Val*>& replacement_map) {
    ReplaceExprInput replacer(replacement_map);
    replacer.traverseAndInsert(exprs);
    return replacer.exprs_;
  }

 private:
  ReplaceExprInput(const std::unordered_map<Val*, Val*>& replacement_map)
      : replacement_map_(replacement_map) {}

  using kir::ExprMutator::handle;

  c10::optional<std::unordered_map<Val*, Val*>> getMaybeInputReplacementMap(
      Expr* expr) {
    bool need_replacement = false;

    std::unordered_map<Val*, Val*> replaced_val;
    for (auto in : expr->inputs()) {
      auto replace_it = replacement_map_.find(in);
      if (replace_it != replacement_map_.end()) {
        need_replacement = true;
        replaced_val[in] = replace_it->second;
      } else {
        replaced_val[in] = in;
      }
    }
    if (need_replacement) {
      return c10::optional<std::unordered_map<Val*, Val*>>(replaced_val);
    } else {
      return c10::nullopt;
    }
  }

  // Copy predicates and register expression replacement
  void registerReplaceWithPredicate(Expr* old_expr, Expr* new_expr) {
    new_expr->setPredicate(old_expr->predicate());
    new_expr->setWritePredicate(old_expr->writePredicate());
    registerReplace(old_expr, new_expr);
  }

  void handle(UnaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<UnaryOp>(
          node->getUnaryOpType(),
          node->out(),
          replaced_inputs.value().at(node->in()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(BinaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<BinaryOp>(
          node->getBinaryOpType(),
          node->out(),
          replaced_inputs.value().at(node->lhs()),
          replaced_inputs.value().at(node->rhs()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(TernaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<TernaryOp>(
          node->getTernaryOpType(),
          node->out(),
          replaced_inputs.value().at(node->in1()),
          replaced_inputs.value().at(node->in2()),
          replaced_inputs.value().at(node->in3()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(ReductionOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<ReductionOp>(
          node->getReductionOpType(),
          node->init(),
          node->out(),
          replaced_inputs.value().at(node->in()),
          node->isFused());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(BroadcastOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<BroadcastOp>(
          node->out(),
          replaced_inputs.value().at(node->in()),
          node->getBroadcastDimFlags());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(WelfordOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<WelfordOp>(
          node->outAvg(),
          node->outVar(),
          node->outN(),
          node->initAvg(),
          node->initVar(),
          node->initN(),
          replaced_inputs.value().at(node->inAvg()),
          replaced_inputs.value().at(node->inVar()),
          replaced_inputs.value().at(node->inN()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(MmaOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<MmaOp>(
          node->out(),
          replaced_inputs.value().at(node->inA()),
          replaced_inputs.value().at(node->inB()),
          node->init(),
          node->options());
      registerReplaceWithPredicate(node, replacement);
    }
  }

 private:
  const std::unordered_map<Val*, Val*>& replacement_map_;
};

} // namespace

std::vector<Expr*> replaceInputsInExpr(
    const std::vector<Expr*>& exprs,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  return ReplaceExprInput::replace(exprs, replacement_map);
}

bool isTrivialIterDomain(IterDomain* id) {
  auto pt = id->getParallelType();
  return id->isReduction() || id->isBroadcast() || id->isStride() ||
      (id->extent()->isOneInt() && id->start()->isZeroInt()) ||
      pt == ParallelType::Vectorize ||
      (isParallelTypeThread(pt) &&
       !GpuLower::current()->haloInfo().hasHaloWidth(id));
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
