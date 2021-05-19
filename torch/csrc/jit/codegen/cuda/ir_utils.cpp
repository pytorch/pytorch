#include <torch/csrc/jit/codegen/cuda/fusion.h>
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
// TODO: Support Welford operation
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
    expr_ = new UnaryOp(unary_expr->getUnaryOpType(), out, in);
  }

  void handle(BinaryOp* binary_expr) final {
    auto lhs = reference_->sameAs(binary_expr->lhs()) ? substitute_
                                                      : binary_expr->lhs();
    auto rhs = reference_->sameAs(binary_expr->rhs()) ? substitute_
                                                      : binary_expr->rhs();
    auto out = reference_->sameAs(binary_expr->out()) ? substitute_
                                                      : binary_expr->out();

    expr_ = new BinaryOp(binary_expr->getBinaryOpType(), out, lhs, rhs);
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
    expr_ = new TernaryOp(ternary_expr->getTernaryOpType(), out, in1, in2, in3);
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

    expr_ =
        new ReductionOp(reduction_expr->getReductionOpType(), init, out, in);
  }

  void handle(BroadcastOp* broadcast_expr) final {
    auto out = reference_->sameAs(broadcast_expr->out())
        ? substitute_
        : broadcast_expr->out();
    auto in = reference_->sameAs(broadcast_expr->in()) ? substitute_
                                                       : broadcast_expr->in();

    expr_ = new BroadcastOp(out, in, broadcast_expr->getBroadcastDimFlags());
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
    expr_ = new TransposeOp(out, in, transpose_expr->new2old());
  }

  void handle(ShiftOp* shift_expr) final {
    auto out =
        reference_->sameAs(shift_expr->out()) ? substitute_ : shift_expr->out();
    auto in =
        reference_->sameAs(shift_expr->in()) ? substitute_ : shift_expr->in();

    expr_ = new ShiftOp(out, in, shift_expr->offsets());
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

} // namespace ir_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
