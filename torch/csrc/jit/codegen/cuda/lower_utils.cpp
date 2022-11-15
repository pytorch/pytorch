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

TVDomainGuard::TVDomainGuard(TensorView* tv, TensorDomain* td)
    : tv_(tv), prev_domain_(tv_->domain()) {
  tv_->setDomain(td);
}

TVDomainGuard::TVDomainGuard(TVDomainGuard&& guard)
    : tv_(nullptr), prev_domain_(guard.prev_domain_) {
  std::swap(tv_, guard.tv_);
}

TVDomainGuard::~TVDomainGuard() {
  if (tv_ != nullptr) {
    tv_->setDomain(prev_domain_);
  }
}

ir_utils::TVDomainGuard overrideContiguityGuard(
    TensorView* tv,
    bool contiguity) {
  // Use domain guard to ignore the contiguity of
  //  consumer tv.
  TensorDomain* domain_with_specified_contiguity = nullptr;
  std::vector<bool> contiguity_vector(
      tv->getMaybeRFactorDomain().size(), contiguity);
  if (tv->hasRFactor()) {
    domain_with_specified_contiguity = IrBuilder::create<TensorDomain>(
        tv->getRootDomain(),
        tv->getRFactorDomain(),
        tv->domain()->domain(),
        contiguity_vector);
  } else {
    domain_with_specified_contiguity = IrBuilder::create<TensorDomain>(
        tv->getRootDomain(), tv->domain()->domain(), contiguity_vector);
  }

  return ir_utils::TVDomainGuard(tv, domain_with_specified_contiguity);
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
       expr->getExprType().value() == ExprType::SelectOp ||
       expr->getExprType().value() == ExprType::RNGOp ||
       expr->getExprType().value() == ExprType::FullOp ||
       expr->getExprType().value() == ExprType::ARangeOp ||
       expr->getExprType().value() == ExprType::EyeOp ||
       expr->getExprType().value() == ExprType::ReductionOp ||
       expr->getExprType().value() == ExprType::GroupedReductionOp ||
       expr->getExprType().value() == ExprType::WelfordOp ||
       expr->getExprType().value() == ExprType::GroupedWelfordOp ||
       expr->getExprType().value() == ExprType::LoadStoreOp ||
       expr->getExprType().value() == ExprType::MmaOp ||
       expr->getExprType().value() == ExprType::BroadcastOp ||
       expr->getExprType().value() == ExprType::SqueezeOp ||
       expr->getExprType().value() == ExprType::TransposeOp ||
       expr->getExprType().value() == ExprType::ExpandOp ||
       expr->getExprType().value() == ExprType::ShiftOp ||
       expr->getExprType().value() == ExprType::GatherOp ||
       expr->getExprType().value() == ExprType::ViewAsScalar ||
       expr->getExprType().value() == ExprType::ViewOp ||
       expr->getExprType().value() == ExprType::GridReduction ||
       expr->getExprType().value() == ExprType::GroupedGridReduction ||
       expr->getExprType().value() == ExprType::GridBroadcast ||
       expr->getExprType().value() == ExprType::GridWelford ||
       expr->getExprType().value() == ExprType::GroupedGridWelford)) {
    return true;
  }
  return false;
}

bool isLdMatrixOp(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    return ldst->opType() == LoadStoreOpType::LdMatrix ||
        ldst->opType() == LoadStoreOpType::LdMatrixTranspose;
  }
  return false;
}

bool isCpAsyncOp(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    return ldst->opType() == LoadStoreOpType::CpAsync;
  }
  return false;
}

bool isTensorScalarFillOp(const Expr* expr) {
  // Check that the input is a single scalar.
  if (expr->inputs().size() == 1 && expr->input(0)->isScalar()) {
    // All load store op with a single scalar input
    //  should be a scalar filling op. Semantically
    //  it literally means `Store`'ing a scalar
    //  into a tensor.
    if (expr->isA<LoadStoreOp>()) {
      return true;
    }
    // Unary copy op is also a scalar filling op.
    if (auto uop = dynamic_cast<const UnaryOp*>(expr)) {
      return uop->getUnaryOpType() == UnaryOpType::Set;
    }
  }
  // Ideally any scalar expression that outputs
  //  to a tensor should be considered in this function
  //  but since we currently only limit scope to
  //  initialization patterns so other scalar expr's
  //  are low priority and are excluded here to avoid confusion.
  return false;
}

TensorView* getTv(Val* val) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<TensorView*>(getTv(const_cast<const Val*>(val)));
}

const TensorView* getTv(const Val* val) {
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

TensorView* getTvInput(const Expr* expr) {
  for (auto inp : expr->inputs()) {
    if (auto tv = getTv(inp)) {
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

c10::optional<IterDomain*> getMaybeWarpReductionDim(
    const Val* output,
    const Val* input) {
  auto tv_out = getTv(output);
  if (tv_out == nullptr) {
    return c10::nullopt;
  }

  auto tv_in = getTv(input);
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

  if (reduction_on_xdim->extent()->isConstInt()) {
    auto extent_value = reduction_on_xdim->extent()->evaluateInt();
    if (extent_value % at::cuda::warp_size() == 0) {
      return c10::optional<IterDomain*>(reduction_on_xdim);
    }
  }

  return c10::nullopt;
}

std::unordered_map<ParallelType, IterDomain*, TypeHash> getParallelDomains(
    const Val* val) {
  const TensorView* tv = nullptr;
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

bool isCpAsyncInit(const Expr* expr) {
  return isTensorScalarFillOp(expr) &&
      // FIXME:
      //  We'd need to add a flag to all the init
      //   exprs so we could robustly detect initialization
      //   in all cases.
      isCpAsyncOp(getTvOutput(expr)->definition());
}

c10::optional<Expr*> getMaybePredicatedSingleton(Expr* expr) {
  if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
    if (ite->elseBody().empty()) {
      if (ite->thenBody().size() == 1) {
        return ite->thenBody().exprs()[0];
      }
    }
  }
  return c10::nullopt;
}

//! Short-cut for checking if the expression loads from global memory.
bool isGlobalLoad(const Expr* expr) {
  if (expr->isA<LoadStoreOp>() ||
      (expr->isA<UnaryOp>() &&
       expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Set)) {
    if (auto in_tv = getTv(expr->input(0))) {
      return in_tv->getMemoryType() == MemoryType::Global;
    }
  }
  return false;
}

//! Short-cut for checking if the given expression initializes buffers
//!  for global memory load.
bool isGlobalLoadInit(const Expr* expr) {
  if (auto uop = dynamic_cast<const UnaryOp*>(expr)) {
    if (uop->in()->isScalar()) {
      // FIXME:
      //  We'd need to add a flag to all the init
      //   exprs so we could robustly detect initialization
      //   in all cases.
      if (isGlobalLoad(getTvOutput(uop)->definition())) {
        return true;
      }
    }
  }
  return false;
}

namespace {

class ExprFlattener : private kir::IrVisitor {
 private:
  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::handle(expr);
    } else {
      flat_exprs_.push_back(expr);
    }
  }

 private:
  std::vector<Expr*> flat_exprs_;

 public:
  //! Flattens scopes extracting out a single ordered list of exprs.
  static std::vector<Expr*> flatten(const std::vector<Expr*>& loop_nests) {
    ExprFlattener flattener;
    for (auto expr : loop_nests) {
      flattener.handle(expr);
    }
    return flattener.flat_exprs_;
  }
};

} // namespace

std::vector<Expr*> flattenScopedExprs(const std::vector<Expr*>& loop_nests) {
  return ExprFlattener::flatten(loop_nests);
}

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
    new_expr = new_expr->withPredicate(old_expr->predicate())
                   ->withWritePredicate(old_expr->writePredicate());
    registerReplace(old_expr, new_expr);
  }

  void handle(UnaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<UnaryOp>(
          node->getUnaryOpType(), node->out(), replaced_inputs->at(node->in()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(BinaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<BinaryOp>(
          node->getBinaryOpType(),
          node->out(),
          replaced_inputs->at(node->lhs()),
          replaced_inputs->at(node->rhs()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(TernaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<TernaryOp>(
          node->getTernaryOpType(),
          node->out(),
          replaced_inputs->at(node->in1()),
          replaced_inputs->at(node->in2()),
          replaced_inputs->at(node->in3()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(RNGOp* node) final {
    // RNGOp has no input
    return;
  }

  void handle(ReductionOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<ReductionOp>(
          node->getReductionOpType(),
          node->init(),
          node->out(),
          replaced_inputs->at(node->in()),
          node->isAllreduce());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(GroupedReductionOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      const auto& map = replaced_inputs.value();
      auto inputs = node->inputs();
      for (auto& input : inputs) {
        auto it = map.find(input);
        if (it != map.end()) {
          input = it->second;
        }
      }
      auto replacement = IrBuilder::create<GroupedReductionOp>(
          node->getReductionOpTypes(),
          node->initVals(),
          node->outputs(),
          inputs,
          node->isAllreduce());
      registerReplaceWithPredicate(node, replacement);
    }
  }
  void handle(BroadcastOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<BroadcastOp>(
          node->out(),
          replaced_inputs->at(node->in()),
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
          replaced_inputs->at(node->inAvg()),
          replaced_inputs->at(node->inVar()),
          replaced_inputs->at(node->inN()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(MmaOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<MmaOp>(
          node->out(),
          replaced_inputs->at(node->inA()),
          replaced_inputs->at(node->inB()),
          node->init(),
          node->options());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(LoadStoreOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<LoadStoreOp>(
          node->opType(), node->out(), node->in());
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

std::vector<Expr*> getAllSwizzlesBetween(
    std::vector<IterDomain*> from,
    std::vector<IterDomain*> to) {
  auto all_expr = DependencyCheck::getAllExprsBetween(
      {from.begin(), from.end()}, {to.begin(), to.end()});

  std::vector<Expr*> all_swizzles;

  std::copy_if(
      all_expr.begin(),
      all_expr.end(),
      std::back_inserter(all_swizzles),
      [](Expr* expr) {
        return expr->getExprType().has_value() &&
            (expr->etype() == ExprType::Swizzle2D);
      });

  return all_swizzles;
}

} // namespace ir_utils

namespace lower_utils {

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map) {
  if (expr->isA<kir::BlockSync>()) {
    return true;
  }

  if (!ir_utils::isTvOp(expr)) {
    return false;
  }

  if (!(ir_utils::isReductionOp(expr) || expr->isA<BroadcastOp>() ||
        expr->isA<kir::GridBroadcast>())) {
    return false;
  }

  // GroupedReductionOp can have multiple output TVs, but they must be
  // parallelized in the same way, so just checking one of them is enough.
  auto tv = ir_utils::getTvOutput(expr);

  if (tv->hasBlockReduction() || tv->hasGridReduction()) {
    return true;
  } else if (expr->isA<BroadcastOp>()) {
    const ParallelTypeBitmap pt_map =
        GpuLower::current()->threadPredMap().getParallelBroadcastDomains(tv);
    return pt_map.any();
  }

  return false;
}

kir::Allocate* allocGlobalBufferForGridComm(
    Val* buffer_size,
    DataType dtype,
    bool zero_init) {
  const std::vector<IterDomain*> new_buffer_ids = {
      IrBuilder::create<IterDomain>(IterDomainBuilder(
          GpuLower::current()->kernel()->zeroVal(), buffer_size))};
  const auto buffer_domain = IrBuilder::create<TensorDomain>(new_buffer_ids);
  const auto buffer_tv =
      IrBuilder::create<TensorView>(buffer_domain, dtype, MemoryType::Global);
  return IrBuilder::create<kir::Allocate>(
      buffer_tv, buffer_tv->getMemoryType(), nullptr, zero_init);
}

BasicAllocInfo getAllocInformation(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& for_loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    bool use_id_map) {
  BasicAllocInfo info;
  auto gpu_lower = GpuLower::current();

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
    if ((tv->isDoubleBuffered() || tv->isCircularBuffered()) &&
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

    if (GpuLower::current()->caMap()->areMapped(
            local_id, fl_id, IdMappingMode::PERMISSIVE)) {
      info.alloc_pos++;
    }

    info.init_for_loop = fl;

    if (!outer_alloc_found) {
      info.alloc_for_loop = fl;
    }
  }

  return info;
}

//! Implementing this in here to avoid including too many headers
//!  in type.cpp. Conceptually this should be a generic definition
//!  rather than a util.
bool supportInlinePredicate(Expr* expr) {
  if (ir_utils::isCpAsyncOp(expr)) {
    return true;
  }
  // TODO: build out support.
  return false;
}

} // namespace lower_utils

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
