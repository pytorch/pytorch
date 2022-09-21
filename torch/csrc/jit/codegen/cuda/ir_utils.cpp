#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace ir_utils {

std::vector<int64_t> normalizeNew2Old(
    const std::vector<int64_t>& new2old_in,
    size_t ndims) {
  TORCH_CHECK(
      new2old_in.size() == ndims,
      "There must be a transpose mapping for each dimension in domain");

  // Canonicalize dimensions by wrapping each dim for the given ndims
  std::vector<int64_t> new2old;
  std::transform(
      new2old_in.begin(),
      new2old_in.end(),
      std::inserter(new2old, new2old.begin()),
      [ndims](int64_t entry) { return entry < 0 ? entry + ndims : entry; });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid
  TORCH_CHECK(
      std::none_of(
          new2old.begin(),
          new2old.end(),
          [ndims](int64_t entry) {
            return entry < 0 || (unsigned int)entry >= ndims;
          }),
      "New2Old axes are not within the number of dimensions of the provided domain.\t",
      new2old);

  // Going to use sets, to see if any duplicate values are in the map.
  std::set<int64_t> old_pos_set;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](int64_t entry) { return entry; });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      new2old.size() == ndims && old_pos_set.size() == new2old.size(),
      "Duplicate entries in transformation map.");

  // END VALIDATION CHECKS
  return new2old;
}

std::vector<int> normalizeOld2New(
    const std::unordered_map<int, int>& old2new_in,
    size_t ndims) {
  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> old2new;
  std::transform(
      old2new_in.begin(),
      old2new_in.end(),
      std::inserter(old2new, old2new.begin()),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return std::unordered_map<int, int>::value_type({
            entry.first < 0 ? entry.first + ndims : entry.first,
            entry.second < 0 ? entry.second + ndims : entry.second,
        });
      });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid

  TORCH_CHECK(
      std::none_of(
          old2new.begin(),
          old2new.end(),
          [ndims](std::unordered_map<int, int>::value_type entry) {
            return entry.first < 0 || (unsigned int)entry.first >= ndims ||
                entry.second < 0 || (unsigned int)entry.second >= ndims;
          }),
      "Reorder axes are not within the number of dimensions of the provided domain.");

  // Going to use sets, to see if any duplicate values are in the map.

  std::set<int> old_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  std::set<int> new_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(new_pos_set, new_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.second;
      });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      old_pos_set.size() == old2new.size() &&
          new_pos_set.size() == old2new.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // END VALIDATION CHECKS

  std::vector<int> new2old(ndims, -1);

  // Go through each old and new position, make sure they're within [0, ndims)
  for (std::pair<int, int> elem : old2new) {
    int old_pos = elem.first;
    int new_pos = elem.second;
    new2old[new_pos] = old_pos;
  }

  // old_positions that already have a new position
  std::set<int> old_positions(new2old.begin(), new2old.end());
  old_positions.erase(-1);

  // All available new positions
  std::set<int> all_positions;
  for (decltype(ndims) i{0}; i < ndims; i++)
    all_positions.insert(i);

  // Check what positions haven't been specified.
  std::set<int> positions_left;
  std::set_difference(
      all_positions.begin(),
      all_positions.end(),
      old_positions.begin(),
      old_positions.end(),
      std::inserter(positions_left, positions_left.end()));

  // Fill in positions that weren't specified, in relative order,
  // in empty spots in the set of new positions.
  // new2old[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  std::transform(
      new2old.begin(), new2old.end(), new2old.begin(), [&it](int i) -> int {
        return i == -1 ? *it++ : i;
      });

  return new2old;
}

namespace ValReplacement {
// Create New Expr given producer - [an input for the expression]
// Creates a new Expr substituting current with producer
struct SubstituteInExpr : public OptInDispatch {
 public:
  static Expr* subsitute(Expr* expr, Val* reference, Val* substitute) {
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && reference != nullptr && substitute != nullptr,
        "Nullptr arg found.");
    SubstituteInExpr sie(reference, substitute);
    sie.handle(expr);
    TORCH_INTERNAL_ASSERT(
        sie.expr_ != nullptr,
        "Substitution failed of ",
        reference,
        " with ",
        substitute);
    return sie.expr_;
  }

 private:
  explicit SubstituteInExpr(Val* reference, Val* substitute)
      : reference_(reference), substitute_(substitute) {}

  void handle(Expr* expr) final {
    OptInDispatch::handle(expr);
  }

  void handle(ARangeOp* arange_expr) final {
    auto start = reference_->sameAs(arange_expr->start())
        ? substitute_
        : arange_expr->start();
    auto end = reference_->sameAs(arange_expr->end()) ? substitute_
                                                      : arange_expr->end();
    auto step = reference_->sameAs(arange_expr->step()) ? substitute_
                                                        : arange_expr->step();
    auto out = reference_->sameAs(arange_expr->output(0))
        ? substitute_
        : arange_expr->output(0);
    expr_ = IrBuilder::create<ARangeOp>(
        arange_expr->container(),
        out,
        start,
        end,
        step,
        arange_expr->getLinearIndex());
  }

  void handle(UnaryOp* unary_expr) final {
    auto in =
        reference_->sameAs(unary_expr->in()) ? substitute_ : unary_expr->in();
    auto out =
        reference_->sameAs(unary_expr->out()) ? substitute_ : unary_expr->out();
    expr_ = IrBuilder::create<UnaryOp>(
        unary_expr->container(), unary_expr->getUnaryOpType(), out, in);
  }

  void handle(BinaryOp* binary_expr) final {
    auto lhs = reference_->sameAs(binary_expr->lhs()) ? substitute_
                                                      : binary_expr->lhs();
    auto rhs = reference_->sameAs(binary_expr->rhs()) ? substitute_
                                                      : binary_expr->rhs();
    auto out = reference_->sameAs(binary_expr->out()) ? substitute_
                                                      : binary_expr->out();

    expr_ = IrBuilder::create<BinaryOp>(
        binary_expr->container(),
        binary_expr->getBinaryOpType(),
        out,
        lhs,
        rhs);
  }

  void handle(TernaryOp* ternary_expr) final {
    auto in1 = reference_->sameAs(ternary_expr->in1()) ? substitute_
                                                       : ternary_expr->in1();
    auto in2 = reference_->sameAs(ternary_expr->in2()) ? substitute_
                                                       : ternary_expr->in2();
    auto in3 = reference_->sameAs(ternary_expr->in3()) ? substitute_
                                                       : ternary_expr->in3();
    auto out = reference_->sameAs(ternary_expr->out()) ? substitute_
                                                       : ternary_expr->out();
    expr_ = IrBuilder::create<TernaryOp>(
        ternary_expr->container(),
        ternary_expr->getTernaryOpType(),
        out,
        in1,
        in2,
        in3);
  }

  void handle(RNGOp* rng_expr) final {
    auto out = reference_->sameAs(rng_expr->output(0)) ? substitute_
                                                       : rng_expr->output(0);
    expr_ = IrBuilder::create<RNGOp>(
        rng_expr->container(),
        rng_expr->getRNGOpType(),
        out,
        rng_expr->getRNGOffset(),
        rng_expr->getPhiloxIndex());
  }

  void handle(ReductionOp* reduction_expr) final {
    auto init = reference_->sameAs(reduction_expr->init())
        ? substitute_
        : reduction_expr->init();
    auto out = reference_->sameAs(reduction_expr->out())
        ? substitute_
        : reduction_expr->out();
    auto in = reference_->sameAs(reduction_expr->in()) ? substitute_
                                                       : reduction_expr->in();

    expr_ = IrBuilder::create<ReductionOp>(
        reduction_expr->container(),
        reduction_expr->getReductionOpType(),
        init,
        out,
        in);
  }

  void handle(GroupedReductionOp* grouped_reduction_expr) final {
    std::vector<Val*> outputs;
    std::transform(
        grouped_reduction_expr->outputs().begin(),
        grouped_reduction_expr->outputs().end(),
        std::back_inserter(outputs),
        [&](Val* val) { return reference_->sameAs(val) ? substitute_ : val; });

    std::vector<Val*> inputs;
    std::transform(
        grouped_reduction_expr->inputs().begin(),
        grouped_reduction_expr->inputs().end(),
        std::back_inserter(inputs),
        [&](Val* val) { return reference_->sameAs(val) ? substitute_ : val; });

    std::vector<Val*> init_vals;
    std::transform(
        grouped_reduction_expr->initVals().begin(),
        grouped_reduction_expr->initVals().end(),
        std::back_inserter(init_vals),
        [&](Val* val) { return reference_->sameAs(val) ? substitute_ : val; });

    expr_ = IrBuilder::create<GroupedReductionOp>(
        grouped_reduction_expr->container(),
        grouped_reduction_expr->getReductionOpTypes(),
        init_vals,
        outputs,
        inputs);
  }

  void handle(BroadcastOp* broadcast_expr) final {
    auto out = reference_->sameAs(broadcast_expr->out())
        ? substitute_
        : broadcast_expr->out();
    auto in = reference_->sameAs(broadcast_expr->in()) ? substitute_
                                                       : broadcast_expr->in();

    expr_ = IrBuilder::create<BroadcastOp>(
        broadcast_expr->container(),
        out,
        in,
        broadcast_expr->getBroadcastDimFlags());
  }

  void handle(TransposeOp* transpose_expr) final {
    TORCH_INTERNAL_ASSERT(
        substitute_->isA<TensorView>(),
        "All args to transpose must be tensor view, but received a non-TensorView for replacement: ",
        substitute_);
    auto out = reference_->sameAs(transpose_expr->out())
        ? substitute_->as<TensorView>()
        : transpose_expr->out();
    auto in = reference_->sameAs(transpose_expr->in())
        ? substitute_->as<TensorView>()
        : transpose_expr->in();
    expr_ = IrBuilder::create<TransposeOp>(
        transpose_expr->container(), out, in, transpose_expr->new2old());
  }

  void handle(ExpandOp* expand_expr) final {
    auto out = reference_->sameAs(expand_expr->out())
        ? substitute_->as<TensorView>()
        : expand_expr->out();
    auto in = reference_->sameAs(expand_expr->in())
        ? substitute_->as<TensorView>()
        : expand_expr->in();

    auto expanded_extents = expand_expr->expanded_extents();
    if (substitute_->isA<Int>()) {
      for (auto i : c10::irange(expanded_extents.size())) {
        if (!expanded_extents[i]->sameAs(substitute_)) {
          expanded_extents[i] = substitute_;
        }
      }
    }
    expr_ = IrBuilder::create<ExpandOp>(
        expand_expr->container(), out, in, expanded_extents);
  }

  void handle(ShiftOp* shift_expr) final {
    auto out =
        reference_->sameAs(shift_expr->out()) ? substitute_ : shift_expr->out();
    auto in =
        reference_->sameAs(shift_expr->in()) ? substitute_ : shift_expr->in();

    expr_ = IrBuilder::create<ShiftOp>(
        shift_expr->container(),
        out,
        in,
        shift_expr->offsets(),
        shift_expr->padWidth());
  }

  void handle(GatherOp* gather_expr) final {
    auto out = reference_->sameAs(gather_expr->out()) ? substitute_
                                                      : gather_expr->out();
    auto in =
        reference_->sameAs(gather_expr->in()) ? substitute_ : gather_expr->in();

    expr_ = IrBuilder::create<GatherOp>(
        gather_expr->container(),
        out,
        in,
        gather_expr->windowShape(),
        gather_expr->padWidth());
  }

  void handle(ViewAsScalar* expr) final {
    TORCH_INTERNAL_ASSERT(
        substitute_->isA<TensorView>(),
        "All args to view must be TensorView, but received a non-TensorView for replacement: ",
        substitute_);
    auto in = reference_->sameAs(expr->in()) ? substitute_->as<TensorView>()
                                             : expr->in();
    auto out = reference_->sameAs(expr->out()) ? substitute_->as<TensorView>()
                                               : expr->out();
    expr_ = IrBuilder::create<ViewAsScalar>(
        expr->container(), out, in, expr->vector_id(), expr->index());
  }

  void handle(ViewOp* view_expr) final {
    TORCH_INTERNAL_ASSERT(
        substitute_->isA<TensorView>(),
        "All args to view must be TensorView, but received a non-TensorView for replacement: ",
        substitute_);
    auto in = reference_->sameAs(view_expr->in())
        ? substitute_->as<TensorView>()
        : view_expr->in();
    auto out = reference_->sameAs(view_expr->out())
        ? substitute_->as<TensorView>()
        : view_expr->out();
    expr_ = IrBuilder::create<ViewOp>(view_expr->container(), out, in);
  }

  void handle(WelfordOp* welford_expr) final {
    auto out_avg = reference_->sameAs(welford_expr->outAvg())
        ? substitute_->as<TensorView>()
        : welford_expr->outAvg();
    auto out_var = reference_->sameAs(welford_expr->outVar())
        ? substitute_->as<TensorView>()
        : welford_expr->outVar();
    auto out_N = reference_->sameAs(welford_expr->outN())
        ? substitute_->as<TensorView>()
        : welford_expr->outN();
    auto in_avg = reference_->sameAs(welford_expr->inAvg())
        ? substitute_->as<TensorView>()
        : welford_expr->inAvg();
    auto in_var =
        welford_expr->inVar() && reference_->sameAs(welford_expr->inVar())
        ? substitute_->as<TensorView>()
        : welford_expr->inVar();
    auto in_N = reference_->sameAs(welford_expr->inN()) ? substitute_
                                                        : welford_expr->inN();
    auto init_avg =
        welford_expr->initAvg() && reference_->sameAs(welford_expr->initAvg())
        ? substitute_->as<TensorView>()
        : welford_expr->initAvg();
    auto init_var =
        welford_expr->initVar() && reference_->sameAs(welford_expr->initVar())
        ? substitute_->as<TensorView>()
        : welford_expr->initVar();
    auto init_N =
        welford_expr->initN() && reference_->sameAs(welford_expr->initN())
        ? substitute_
        : welford_expr->initN();
    expr_ = IrBuilder::create<WelfordOp>(
        welford_expr->container(),
        out_avg,
        out_var,
        out_N,
        in_avg,
        in_var,
        in_N,
        init_avg,
        init_var,
        init_N,
        welford_expr->isAllreduce());
  }

  void handle(LoadStoreOp* ldst_expr) final {
    TORCH_INTERNAL_ASSERT(
        substitute_->isA<TensorView>(),
        "All args to view must be TensorView, but received a non-TensorView for replacement: ",
        substitute_);
    auto in = reference_->sameAs(ldst_expr->in())
        ? substitute_->as<TensorView>()
        : ldst_expr->in();
    auto out = reference_->sameAs(ldst_expr->out())
        ? substitute_->as<TensorView>()
        : ldst_expr->out();
    expr_ = IrBuilder::create<LoadStoreOp>(
        ldst_expr->container(), ldst_expr->opType(), out, in);
  }

  void handle(MmaOp* mma_expr) final {
    TORCH_INTERNAL_ASSERT(
        substitute_->isA<TensorView>(),
        "All args to MmaOp must be TensorView, but received a non-TensorView for replacement: ",
        substitute_);
    auto in_a = reference_->sameAs(mma_expr->inA())
        ? substitute_->as<TensorView>()
        : mma_expr->inA();
    auto in_b = reference_->sameAs(mma_expr->inB())
        ? substitute_->as<TensorView>()
        : mma_expr->inB();
    auto out = reference_->sameAs(mma_expr->out())
        ? substitute_->as<TensorView>()
        : mma_expr->out();
    auto init = reference_->sameAs(mma_expr->init())
        ? substitute_->as<TensorView>()
        : mma_expr->init();
    expr_ = IrBuilder::create<MmaOp>(
        mma_expr->container(), out, in_a, in_b, init, mma_expr->options());
  }

 private:
  Val* reference_ = nullptr;
  Val* substitute_ = nullptr;
  Expr* expr_ = nullptr;
};

} // namespace ValReplacement

Expr* replaceValInExpr(Expr* expr, Val* reference, Val* substitute) {
  FusionGuard fg(expr->fusion());
  return ValReplacement::SubstituteInExpr::subsitute(
      expr, reference, substitute);
}

TensorView* rfactorHelper(
    TensorView* reduction_tv,
    const std::vector<int>& axes) {
  TORCH_INTERNAL_ASSERT(reduction_tv->definition() != nullptr);
  const bool has_multiple_tvs = reduction_tv->definition()->inputs().size() > 1;
  if (!has_multiple_tvs) {
    return reduction_tv->rFactor(axes);
  }

  std::vector<TensorView*> out_tvs;
  std::transform(
      reduction_tv->definition()->outputs().begin(),
      reduction_tv->definition()->outputs().end(),
      std::back_inserter(out_tvs),
      [](Val* val) { return val->as<TensorView>(); });

  auto rf_tvs = reduction_tv->rFactor(axes, out_tvs);

  return rf_tvs.at(std::distance(
      out_tvs.begin(),
      std::find(out_tvs.begin(), out_tvs.end(), reduction_tv)));
}

namespace {

template <typename T>
std::vector<T*> uniqueEntries(const std::vector<T*>& tv_deuqe) {
  std::vector<T*> unique_entries;
  std::unordered_set<T*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.emplace(tv_entry).second) {
      unique_entries.emplace_back(tv_entry);
    }
  }
  return unique_entries;
}

} // namespace

// Return immediate producers of val
TORCH_CUDA_CU_API std::vector<Val*> producerValsOf(Val* val) {
  if (val->definition() == nullptr) {
    return {};
  }
  auto producer_vals = val->definition()->inputs();
  return uniqueEntries<Val>({producer_vals.begin(), producer_vals.end()});
}

// Return immediate consumers of val
TORCH_CUDA_CU_API std::vector<Val*> consumerValsOf(Val* val) {
  std::vector<Val*> consumer_vals;
  for (auto use_expr : val->uses()) {
    auto outputs = use_expr->outputs();
    consumer_vals.insert(consumer_vals.end(), outputs.begin(), outputs.end());
  }
  return uniqueEntries<Val>(consumer_vals);
}

// Return immediate siblings of val
TORCH_CUDA_CU_API std::vector<Val*> siblingValsOf(Val* val) {
  std::vector<Val*> sibling_vals;
  auto def = val->definition();
  if (def != nullptr) {
    auto outs = def->outputs();
    for (auto sibling_val : outs) {
      if (sibling_val == val) {
        continue;
      }
      sibling_vals.emplace_back(sibling_val);
    }
  }
  return sibling_vals;
}

// Return immediate producers of val
TORCH_CUDA_CU_API std::vector<Val*> producerValsOf(
    const std::vector<Val*>& vals) {
  std::vector<Val*> all_producer_vals;
  for (auto val : vals) {
    auto producer_vals = producerValsOf(val);
    all_producer_vals.insert(
        all_producer_vals.end(), producer_vals.begin(), producer_vals.end());
  }

  return uniqueEntries<Val>(all_producer_vals);
}

// Return immediate consumers of val
TORCH_CUDA_CU_API std::vector<Val*> consumerValsOf(
    const std::vector<Val*>& vals) {
  std::vector<Val*> all_consumer_vals;
  for (auto val : vals) {
    auto consumer_vals = consumerValsOf(val);
    all_consumer_vals.insert(
        all_consumer_vals.end(), consumer_vals.begin(), consumer_vals.end());
  }

  return uniqueEntries<Val>(all_consumer_vals);
}

std::vector<TensorView*> producerTvsOf(TensorView* tv) {
  auto producer_vals = producerValsOf(tv);
  auto producer_tvs = ir_utils::filterByType<TensorView>(producer_vals);
  return {producer_tvs.begin(), producer_tvs.end()};
}

std::vector<TensorView*> consumerTvsOf(TensorView* tv) {
  auto consumer_vals = consumerValsOf(tv);
  auto consumer_tvs = ir_utils::filterByType<TensorView>(consumer_vals);
  return {consumer_tvs.begin(), consumer_tvs.end()};
}

TORCH_CUDA_CU_API std::vector<TensorView*> siblingTvsOf(TensorView* tv) {
  auto sibling_vals = siblingValsOf(tv);
  auto sibling_tvs = ir_utils::filterByType<TensorView>(sibling_vals);
  return {sibling_tvs.begin(), sibling_tvs.end()};
}

std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_producer_tvs;
  for (auto tv : tvs) {
    auto producer_tvs = producerTvsOf(tv);
    all_producer_tvs.insert(
        all_producer_tvs.end(), producer_tvs.begin(), producer_tvs.end());
  }

  return uniqueEntries<TensorView>(all_producer_tvs);
}

std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_consumer_tvs;
  for (auto tv : tvs) {
    auto consumer_tvs = consumerTvsOf(tv);
    all_consumer_tvs.insert(
        all_consumer_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
  }

  return uniqueEntries<TensorView>(all_consumer_tvs);
}

std::vector<TensorView*> inputTvsOf(TensorView* tv) {
  return inputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> outputTvsOf(TensorView* tv) {
  return outputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs) {
  auto inp_vals = IterVisitor::getInputsTo({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(inp_vals);
  std::vector<TensorView*> inp_tvs(filtered.begin(), filtered.end());
  return uniqueEntries<TensorView>(inp_tvs);
}

std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs) {
  auto out_vals = DependencyCheck::getAllOutputsOf({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(out_vals);
  std::vector<TensorView*> out_tvs(filtered.begin(), filtered.end());
  return uniqueEntries<TensorView>(out_tvs);
}

std::vector<TensorView*> allTvs(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);

  // This shouldn't be necessary but FusionSegmentIoAlias_CUDA due to aliasing
  // is having an input disconnected from outputs, and these iter domains are
  // being checked in compute at maps in scheduling logic. This shouldn't hurt
  // AFAICT.
  auto tv_inputs = ir_utils::filterByType<TensorView>(fusion->inputs());

  std::vector<TensorView*> all_tvs({used_tvs.begin(), used_tvs.end()});
  // Sometimes inputs are not connected to outputs, however, we still include
  // them when returning allTvs because they are registered as an input.
  all_tvs.insert(all_tvs.end(), tv_inputs.begin(), tv_inputs.end());

  // all_tvs has duplicates, to deduplicate it and return
  return uniqueEntries<TensorView>(all_tvs);
}

std::vector<TensorView*> allTvsExcept(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& except) {
  auto all_tvs = allTvs(fusion);
  std::vector<TensorView*> result;
  for (auto tv : all_tvs) {
    if (except.count(tv) == 0) {
      result.emplace_back(tv);
    }
  }
  return result;
}

std::vector<Expr*> getReductionOps(Fusion* fusion, bool ignore_trivial) {
  std::vector<Expr*> red_ops;

  auto isReduction = [&ignore_trivial](Val* out_val) {
    if (out_val == nullptr || !out_val->isA<TensorView>()) {
      return false;
    }
    auto out_tv = out_val->as<TensorView>();
    return std::any_of(
        out_tv->getRootDomain().begin(),
        out_tv->getRootDomain().end(),
        [&ignore_trivial](IterDomain* id) {
          return id->isReduction() &&
              !(ignore_trivial && id->isTrivialReduction());
        });
  };

  for (auto expr : fusion->exprs()) {
    bool is_reduction = false;
    if (expr->isA<ReductionOp>()) {
      is_reduction = isReduction(expr->as<ReductionOp>()->out());
    } else if (expr->isA<GroupedReductionOp>()) {
      is_reduction = std::any_of(
          expr->as<GroupedReductionOp>()->outputs().begin(),
          expr->as<GroupedReductionOp>()->outputs().end(),
          isReduction);
    } else if (expr->isA<WelfordOp>()) {
      is_reduction = isReduction(expr->as<WelfordOp>()->outAvg());
    }
    if (is_reduction) {
      red_ops.push_back(expr);
    }
  }

  return red_ops;
}

namespace {

class ValReplacementMutator : private OptOutMutator {
 public:
  ValReplacementMutator(
      Fusion* fusion,
      const std::unordered_map<Val*, Val*>& replacement_map)
      : replacement_map_(replacement_map) {
    FusionGuard fg(fusion);

    // Welford makes this a little annoying since it holds a count which is
    // typically not used by anything else. If we don't grab that count, then it
    // would be a tensorview that doesn't get updated extents. Therefore, first
    // grab all leaves towards outputs and grab stmts from there.
    auto stmts = StmtSort::getStmts(fusion, allLeafOuts(fusion), true);

    // Some fusions, such as standalone randlike, can have disconnected DAG, so
    // we need some mechanism to make sure our replacement set is as complete as
    // possible
    // TODO: I think we need a more general mechanism to support disconnected
    // DAG
    std::vector<Val*> more;
    for (auto v : fusion->inputs()) {
      if (std::find(stmts.begin(), stmts.end(), v) == stmts.end()) {
        more.emplace_back(v);
      }
    }
    auto more_stmts = StmtSort::getStmts(fusion, more, true);
    more_stmts.insert(more_stmts.end(), stmts.begin(), stmts.end());

    for (auto stmt : more_stmts) {
      mutate(stmt);
    }
  }

 private:
  using OptOutMutator::mutate;

  void mutate(Val* val) final {
    if (replacement_map_.find(val) == replacement_map_.end()) {
      return OptOutMutator::mutate(val);
    }
    auto replaced_val = replacement_map_.at(val);
    registerMutation(val, replaced_val);
  }

  std::vector<Val*> allLeafOuts(Fusion* fusion) {
    auto exprs = StmtSort::getExprs(fusion, true);
    std::unordered_set<Val*> inputs;
    std::unordered_set<Val*> outputs;
    std::vector<Val*> ordered_outputs;
    for (auto expr : exprs) {
      inputs.insert(expr->inputs().begin(), expr->inputs().end());
      outputs.insert(expr->outputs().begin(), expr->outputs().end());
      ordered_outputs.insert(
          ordered_outputs.end(),
          expr->outputs().begin(),
          expr->outputs().end());
    }
    for (auto input : inputs) {
      outputs.erase(input);
    }

    std::vector<Val*> ordered_leaf_outs;
    for (auto out : ordered_outputs) {
      if (outputs.find(out) != outputs.end()) {
        ordered_leaf_outs.push_back(out);
      }
    }
    return ordered_leaf_outs;
  }

  const std::unordered_map<Val*, Val*>& replacement_map_;
};

} // namespace

void replaceValue(
    Fusion* fusion,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  ValReplacementMutator(fusion, replacement_map);
}

Val* getReductionInitValOf(TensorView* tv) {
  auto def = tv->definition();
  if (def == nullptr) {
    return nullptr;
  }

  Val* init = nullptr;
  if (auto rop = dynamic_cast<ReductionOp*>(def)) {
    init = rop->init();
  } else if (auto grop = dynamic_cast<GroupedReductionOp*>(def)) {
    int output_idx = grop->getExprIndexOfOutput(tv);
    init = grop->initVal(output_idx);
  } else if (auto wop = dynamic_cast<WelfordOp*>(def)) {
    return wop->getInitValOfOutput(tv);
  } else if (auto gwop = dynamic_cast<GroupedWelfordOp*>(def)) {
    init = gwop->getInitValOfOutput(tv);
  } else if (auto mma = dynamic_cast<MmaOp*>(def)) {
    init = mma->init();
  }

  return init;
}

// TODO: Should mma be in here? Should we return true if it's a trivial
// reduction?
bool isReductionOp(const Expr* expr) {
  // Note that GridReduction inherits ReductionOp
  return expr->isA<ReductionOp>() || expr->isA<GroupedReductionOp>() ||
      expr->isA<WelfordOp>() || expr->isA<GroupedWelfordOp>() ||
      expr->isA<kir::GridWelford>() || expr->isA<kir::GroupedGridWelford>();
}

bool isReductionTvOp(const Expr* expr) {
  return ir_utils::isTvOp(expr) && isReductionOp(expr);
}

namespace {

struct ReplaceValInIndexVal : public OptInDispatch {
 public:
  //! Apply replacements to index as specified in
  //! replacement_map. index is assumed to consist only from Int and
  //! NamedScalar
  static Val* replace(
      Val* index,
      const std::unordered_map<Val*, Val*>& replacement_map) {
    ReplaceValInIndexVal replace_index_val(replacement_map);
    replace_index_val.handle(index);
    // Return the original index if not replaced
    if (replace_index_val.is_replaced_) {
      return replace_index_val.last_visited_val_;
    } else {
      return index;
    }
  }

 private:
  ReplaceValInIndexVal(const std::unordered_map<Val*, Val*>& replacement_map)
      : replacement_map_(replacement_map) {}

  using OptOutDispatch::handle;

  void handle(Val* val) override {
    TORCH_INTERNAL_ASSERT(
        val->isA<Int>() || val->isA<NamedScalar>() || val->isA<kir::IntPair>(),
        "Invalid Val type: ",
        val->toString());

    // if val appears in the replacement map, stop traversing and set
    // the current val with the replacement
    auto it = replacement_map_.find(val);
    if (it != replacement_map_.end()) {
      last_visited_val_ = it->second;
      is_replaced_ = true;
      return;
    }

    // Recursively traverse its defining expr
    auto def = val->definition();
    if (def != nullptr) {
      switch (def->etype()) {
        case ExprType::UnaryOp:
        case ExprType::BinaryOp:
        case ExprType::Swizzle2DInt:
        case ExprType::PairSelect:
          handle(val->definition());
          break;
        default:
          TORCH_INTERNAL_ASSERT(
              false, "Unexpected definition: ", def->toString())
      }
      // last_visited_val_ is set in the expr handlers
    } else {
      last_visited_val_ = val;
    }
  }

  // Clone expression after recurisvely replacing inputs
  void handle(UnaryOp* uop) override {
    handle(uop->in());
    auto inp = last_visited_val_;
    TORCH_INTERNAL_ASSERT(uop->out()->isA<Int>());
    auto out = IrBuilder::create<Int>(c10::nullopt);
    IrBuilder::create<UnaryOp>(uop->getUnaryOpType(), out, inp);
    last_visited_val_ = out;
  }

  // Clone expression after recurisvely replacing inputs
  void handle(BinaryOp* bop) override {
    handle(bop->lhs());
    auto lhs = last_visited_val_;
    handle(bop->rhs());
    auto rhs = last_visited_val_;
    TORCH_INTERNAL_ASSERT(bop->out()->isA<Int>());
    auto out = IrBuilder::create<Int>(c10::nullopt);
    IrBuilder::create<BinaryOp>(bop->getBinaryOpType(), out, lhs, rhs);
    last_visited_val_ = out;
  }

  // Clone expression after recurisvely replacing inputs
  void handle(kir::Swizzle2DInt* swizzle_2d) override {
    handle(swizzle_2d->inX());
    auto in_x = last_visited_val_;
    handle(swizzle_2d->inY());
    auto in_y = last_visited_val_;
    auto out = IrBuilder::create<kir::IntPair>();

    // Extents are assumed constant in swizzle so no need to
    //  duplicate their graphs.
    IrBuilder::create<kir::Swizzle2DInt>(
        out,
        in_x,
        in_y,
        swizzle_2d->extentX(),
        swizzle_2d->extentY(),
        swizzle_2d->swizzleType());
    last_visited_val_ = out;
  }

  void handle(kir::PairSelect* pair_select) override {
    handle(pair_select->in()->asVal());
    auto in = last_visited_val_;
    TORCH_INTERNAL_ASSERT(pair_select->out()->isA<Int>());
    auto out = IrBuilder::create<Int>(c10::nullopt);
    IrBuilder::create<kir::PairSelect>(
        out, in->as<kir::IntPair>(), pair_select->selection());
    last_visited_val_ = out;
  }

 private:
  const std::unordered_map<Val*, Val*>& replacement_map_;
  Val* last_visited_val_ = nullptr;
  bool is_replaced_ = false;
};

} // namespace

Val* replaceValInIndexVal(
    Val* index,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  return ReplaceValInIndexVal::replace(index, replacement_map);
}

} // namespace ir_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
