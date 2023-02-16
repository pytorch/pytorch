#include <fusion.h>
#include <ir_builder.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <lower_utils.h>
#include <ops/arith.h>

#include <set>

namespace nvfuser {
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
struct SubstituteInExpr : public OptOutMutator {
 public:
  static Expr* subsitute(Expr* expr, Val* reference, Val* substitute) {
    TORCH_INTERNAL_ASSERT(
        expr != nullptr && reference != nullptr && substitute != nullptr,
        "Nullptr arg found.");
    SubstituteInExpr sie(reference, substitute);
    sie.mutate(expr);
    // if nothing substituted, then return the original expr
    return sie.expr_ == nullptr ? expr : sie.expr_;
  }

 protected:
  virtual void removeExpr(IrContainer*, Expr*) const override {}

  virtual void registerNewExpr(Expr* expr) override {
    expr_ = expr;
  }

 private:
  explicit SubstituteInExpr(Val* reference, Val* substitute) {
    mutations_[reference] = substitute;
  }

 private:
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
std::vector<Val*> producerValsOf(Val* val) {
  if (val->definition() == nullptr) {
    return {};
  }
  auto producer_vals = val->definition()->inputs();
  return uniqueEntries<Val>({producer_vals.begin(), producer_vals.end()});
}

// Return immediate consumers of val
std::vector<Val*> consumerValsOf(Val* val) {
  std::vector<Val*> consumer_vals;
  for (auto use_expr : val->uses()) {
    auto outputs = use_expr->outputs();
    consumer_vals.insert(consumer_vals.end(), outputs.begin(), outputs.end());
  }
  return uniqueEntries<Val>(consumer_vals);
}

// Return immediate siblings of val
std::vector<Val*> siblingValsOf(Val* val) {
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
std::vector<Val*> producerValsOf(const std::vector<Val*>& vals) {
  std::vector<Val*> all_producer_vals;
  for (auto val : vals) {
    auto producer_vals = producerValsOf(val);
    all_producer_vals.insert(
        all_producer_vals.end(), producer_vals.begin(), producer_vals.end());
  }

  return uniqueEntries<Val>(all_producer_vals);
}

// Return immediate consumers of val
std::vector<Val*> consumerValsOf(const std::vector<Val*>& vals) {
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

std::vector<TensorView*> siblingTvsOf(TensorView* tv) {
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

std::vector<Expr*> getReductionOps(Fusion* fusion) {
  std::vector<Expr*> red_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<ReductionOp>() || expr->isA<GroupedReductionOp>() ||
        expr->isA<WelfordOp>()) {
      red_ops.push_back(expr);
    }
  }

  return red_ops;
}

std::vector<IndexSelectOp*> getIndexSelectOps(Fusion* fusion) {
  std::vector<IndexSelectOp*> idx_sel_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<IndexSelectOp>()) {
      idx_sel_ops.push_back(expr->as<IndexSelectOp>());
    }
  }

  return idx_sel_ops;
}

std::vector<TorchGatherOp*> getTorchGatherOps(Fusion* fusion) {
  std::vector<TorchGatherOp*> torch_gather_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<TorchGatherOp>()) {
      torch_gather_ops.push_back(expr->as<TorchGatherOp>());
    }
  }

  return torch_gather_ops;
}

std::vector<SelectOp*> getSelectOps(Fusion* fusion) {
  std::vector<SelectOp*> select_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<SelectOp>()) {
      select_ops.push_back(expr->as<SelectOp>());
    }
  }

  return select_ops;
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

    // Some fusions, such as standalone rand_like, can have disconnected DAG, so
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
  return expr->isOneOf<
      ReductionOp,
      GroupedReductionOp,
      WelfordOp,
      GroupedWelfordOp,
      kir::GridWelford,
      kir::GroupedGridWelford>();
}

bool isReductionTvOp(const Expr* expr) {
  return ir_utils::isTvOp(expr) && isReductionOp(expr);
}

std::vector<ViewOp*> getViewOps(Fusion* fusion) {
  auto all_exprs = fusion->exprs();

  auto all_view_ops = ir_utils::filterByType<ViewOp>(all_exprs);

  std::vector<ViewOp*> view_ops;

  std::copy_if(
      all_view_ops.begin(),
      all_view_ops.end(),
      std::back_inserter(view_ops),
      [](ViewOp* view) {
        return std::any_of(
            view->outputs().begin(), view->outputs().end(), [](Val* v) {
              if (!v->isA<TensorView>()) {
                return false;
              }
              return v->as<TensorView>()->hasRFactor();
            });
      });

  return view_ops;
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
        val->isA<Int>() || val->isA<Bool>() || val->isA<NamedScalar>(),
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
      if (def->isOneOf<UnaryOp, BinaryOp, TernaryOp>()) {
        handle(val->definition());
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unexpected definition: ", def->toString())
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
    TORCH_INTERNAL_ASSERT(
        uop->out()->isA<Int>() || uop->out()->isA<Bool>(),
        "Unknown output type for expr ",
        uop->toInlineString());
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
    TORCH_INTERNAL_ASSERT(
        bop->out()->isA<Int>() || bop->out()->isA<Bool>(),
        "Unknown output type for expr ",
        bop->toInlineString());
    auto out = IrBuilder::create<Int>(c10::nullopt);
    IrBuilder::create<BinaryOp>(bop->getBinaryOpType(), out, lhs, rhs);
    last_visited_val_ = out;
  }

  // Clone expression after recurisvely replacing inputs
  void handle(TernaryOp* top) override {
    handle(top->in1());
    auto in1 = last_visited_val_;
    handle(top->in2());
    auto in2 = last_visited_val_;
    handle(top->in3());
    auto in3 = last_visited_val_;
    TORCH_INTERNAL_ASSERT(
        top->out()->isA<Int>() || top->out()->isA<Bool>(),
        "Unknown output type for expr ",
        top->toInlineString());
    auto out = IrBuilder::create<Int>(c10::nullopt);
    IrBuilder::create<TernaryOp>(top->getTernaryOpType(), out, in1, in2, in3);
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

bool isSqueezeInput(const TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<SqueezeOp>()) {
      return true;
    }
  }
  return false;
}

bool isSqueezedID(const TensorView* tv, const IterDomain* id) {
  auto root_dom = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  auto squeezes = ir_utils::filterByType<SqueezeOp>(tv->uses());
  for (auto i : c10::irange(root_dom.size())) {
    if (root_dom[i] != id) {
      continue;
    }
    for (auto squeeze : squeezes) {
      if (squeeze->isSqueezeDim(i)) {
        return true;
      }
    }
  }
  return false;
}

std::vector<IterDomain*> allIDsOf(const TensorView* tv) {
  const auto& root_domain = tv->getRootDomain();
  const auto& domain = tv->domain()->domain();
  // Grab all values in the history of the tensor view's domain
  auto all_vals = DependencyCheck::getAllValsBetween(
      {root_domain.begin(), root_domain.end()}, {domain.begin(), domain.end()});

  // Filter so we only have iteration domains (ignore Ints used in split)
  auto all_ids = ir_utils::filterByType<IterDomain>(all_vals);
  return std::vector<IterDomain*>(all_ids.begin(), all_ids.end());
}

bool isSelectInput(TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<SelectOp>()) {
      return true;
    }
  }
  return false;
}

bool isIndexSelectLookupTv(const TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<IndexSelectOp>()) {
      auto idx_sel = expr->as<IndexSelectOp>();
      if (idx_sel->input(0) == tv) {
        return true;
      }
    }
  }
  return false;
}

bool isIndexSelectIndicesTv(const TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<IndexSelectOp>()) {
      auto idx_sel = expr->as<IndexSelectOp>();
      if (idx_sel->input(1) == tv) {
        return true;
      }
    }
  }
  return false;
}

bool isTorchGatherLookupTv(const Val* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<TorchGatherOp>()) {
      auto idx_sel = expr->as<TorchGatherOp>();
      if (idx_sel->lookupTv() == tv) {
        return true;
      }
    }
  }
  return false;
}

bool isTorchGatherIndicesTv(const Val* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<TorchGatherOp>()) {
      auto idx_sel = expr->as<TorchGatherOp>();
      if (idx_sel->indexTv() == tv) {
        return true;
      }
    }
  }
  return false;
}

std::string varName(const Val* val) {
  if (val->isA<kir::TensorIndex>()) {
    return varName(val->as<kir::TensorIndex>()->view());
  }
  std::stringstream name;
  if (val->isA<TensorView>()) {
    name << "T";
  } else {
    name << typePrefix(val->dtype());
  }
  name << val->name();
  return name.str();
}

} // namespace ir_utils
} // namespace nvfuser
