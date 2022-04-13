#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

#include <set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace ir_utils {

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

  void handle(ViewDtypeOp* view_expr) final {
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
    expr_ = IrBuilder::create<ViewDtypeOp>(
        view_expr->container(), out, in, view_expr->dtype());
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
        init_avg,
        init_var,
        init_N,
        in_avg,
        in_var,
        in_N,
        welford_expr->isFused());
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
  const bool is_welford = reduction_tv->definition()->isA<WelfordOp>();
  if (!is_welford) {
    return reduction_tv->rFactor(axes);
  }
  auto welford = reduction_tv->definition()->as<WelfordOp>();
  auto w_avg = welford->outAvg()->as<TensorView>();
  auto w_var = welford->outVar()->as<TensorView>();
  auto w_n = welford->outN()->as<TensorView>();

  WelfordResult rtvs = reduction_tv->rFactor(axes, w_avg, w_var, w_n);

  if (reduction_tv == w_n) {
    return rtvs.n;
  } else if (reduction_tv == w_var) {
    return rtvs.var_sum;
  } else {
    return rtvs.avg;
  }
}

namespace {

std::vector<TensorView*> uniqueEntries(
    const std::vector<TensorView*>& tv_deuqe) {
  std::vector<TensorView*> unique_entries;
  std::unordered_set<TensorView*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.emplace(tv_entry).second) {
      unique_entries.emplace_back(tv_entry);
    }
  }
  return unique_entries;
}

} // namespace

std::vector<TensorView*> producerTvsOf(TensorView* tv) {
  if (tv->definition() == nullptr) {
    return {};
  }
  auto producer_vals =
      ir_utils::filterByType<TensorView>(tv->definition()->inputs());
  return uniqueEntries({producer_vals.begin(), producer_vals.end()});
}

std::vector<TensorView*> consumerTvsOf(TensorView* tv) {
  std::vector<TensorView*> consumer_tvs;
  for (auto use_expr : tv->uses()) {
    auto outputs = ir_utils::filterByType<TensorView>(use_expr->outputs());
    consumer_tvs.insert(consumer_tvs.end(), outputs.begin(), outputs.end());
  }
  return uniqueEntries(consumer_tvs);
}

std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_producer_tvs;
  for (auto tv : tvs) {
    auto producer_tvs = producerTvsOf(tv);
    all_producer_tvs.insert(
        all_producer_tvs.end(), producer_tvs.begin(), producer_tvs.end());
  }

  return uniqueEntries(all_producer_tvs);
}

std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_consumer_tvs;
  for (auto tv : tvs) {
    auto consumer_tvs = consumerTvsOf(tv);
    all_consumer_tvs.insert(
        all_consumer_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
  }

  return uniqueEntries(all_consumer_tvs);
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
  return uniqueEntries(inp_tvs);
}

std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs) {
  auto out_vals = DependencyCheck::getAllOutputsOf({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(out_vals);
  std::vector<TensorView*> out_tvs(filtered.begin(), filtered.end());
  return uniqueEntries(out_tvs);
}

std::vector<TensorView*> allTvs(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  return uniqueEntries({used_tvs.begin(), used_tvs.end()});
}

std::vector<Expr*> getReductionOps(Fusion* fusion, bool ignore_trivial) {
  std::vector<Expr*> red_ops;
  for (auto expr : fusion->exprs()) {
    const Val* out_val = nullptr;
    if (expr->isA<ReductionOp>()) {
      out_val = expr->as<ReductionOp>()->out();
    } else if (expr->isA<WelfordOp>()) {
      out_val = expr->as<WelfordOp>()->outAvg();
    } else {
      continue;
    }
    if (out_val == nullptr || !out_val->isA<TensorView>()) {
      continue;
    }
    auto out_tv = out_val->as<TensorView>();
    if (std::any_of(
            out_tv->getRootDomain().begin(),
            out_tv->getRootDomain().end(),
            [&ignore_trivial](IterDomain* id) {
              return id->isReduction() &&
                  !(ignore_trivial && id->isTrivialReduction());
            })) {
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
    for (auto stmt : stmts) {
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

} // namespace ir_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
