#include <torch/csrc/jit/passes/normalize_ops.h>

#include <c10/util/Exception.h>

namespace torch::jit {

namespace {

// having multiple ops in our IR that do the same thing makes the IR more
// difficult to consumer for downstream user of the IR, such as our own
// optimization passes here, we convert op aliases into a standard form
bool normalizeOpAliases(graph_node_list_iterator& iter) {
  auto alias = getOperatorAliasMap().find(iter->kind());
  if (alias != getOperatorAliasMap().end()) {
    iter->replaceWithNewSymbol(alias->second);
    iter.destroyCurrent();
    return true;
  }
  return false;
}

// Normalize rsub such that `rsub(x,y) = sub(x,y)`
bool normalizeRSub(graph_node_list_iterator& iter) {
  if (iter->matches(
          "aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")) {
    ArrayRef<Value*> args = iter->inputs();
    Node* newSub = iter->replaceWithNewSymbol(aten::sub);
    newSub->replaceInput(0, args[1]);
    newSub->replaceInput(1, args[0]);
    iter.destroyCurrent();
    return true;
  }
  return false;
}

// Normalizes a `__is__` comparison with a bool to `eq` (and same with
// `__isnot__`)
bool normalizeIsBool(graph_node_list_iterator& iter) {
  ArrayRef<Value*> args = iter->inputs();
  if (args.size() == 2 && args[0]->type() == BoolType::get() &&
      args[1]->type() == BoolType::get()) {
    if (iter->kind() == aten::__is__) {
      iter->replaceWithNewSymbol(aten::eq);
      iter.destroyCurrent();
      return true;
    }
    if (iter->kind() == aten::__isnot__) {
      iter->replaceWithNewSymbol(aten::ne);
      iter.destroyCurrent();
      return true;
    }
  }
  return false;
}

void NormalizeOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    for (auto sub : it->blocks()) {
      NormalizeOps(sub);
    }

    if (normalizeRSub(it)) {
      continue;
    }

    if (normalizeOpAliases(it)) {
      continue;
    }

    if (normalizeIsBool(it)) {
      continue;
    }

    it++;
  }
}

} // namespace

const std::unordered_map<Symbol, Symbol>& getOperatorAliasMap() {
  // map from op alias -> normalized op
  static const std::unordered_map<Symbol, Symbol> alias_map = {
      {aten::absolute, aten::abs},
      {aten::absolute_, aten::abs_},
      {aten::clip, aten::clamp},
      {aten::clip_, aten::clamp_},
      {aten::det, aten::linalg_det},
      {aten::matrix_power, aten::linalg_matrix_power},
      {aten::matrix_exp, aten::linalg_matrix_exp},
      {aten::ger, aten::outer},
      {aten::arccos, aten::acos},
      {aten::arccos_, aten::acos_},
      {aten::arcsin, aten::asin},
      {aten::arcsin_, aten::asin_},
      {aten::arctan, aten::atan},
      {aten::arctan_, aten::atan_},
      {aten::arctan2, aten::atan2},
      {aten::arctan2_, aten::atan2_},
      {aten::arccosh, aten::acosh},
      {aten::arccosh_, aten::acosh_},
      {aten::arcsinh, aten::asinh},
      {aten::arcsinh_, aten::asinh_},
      {aten::arctanh, aten::atanh},
      {aten::arctanh_, aten::atanh_},
      {aten::fix, aten::trunc},
      {aten::fix_, aten::trunc_},
      {aten::negative, aten::neg},
      {aten::negative_, aten::neg_},
      {aten::subtract, aten::sub},
      {aten::subtract_, aten::sub_},
      {aten::greater_equal, aten::ge},
      {aten::greater_equal_, aten::ge_},
      {aten::greater, aten::gt},
      {aten::greater_, aten::gt_},
      {aten::less_equal, aten::le},
      {aten::less_equal_, aten::le_},
      {aten::less, aten::lt},
      {aten::less_, aten::lt_},
      {aten::not_equal, aten::ne},
      {aten::not_equal_, aten::ne_},
      {aten::divide, aten::div},
      {aten::divide_, aten::div_},
      {aten::multiply, aten::mul},
      {aten::multiply_, aten::mul_},
      {aten::linalg_matmul, aten::matmul},
      {aten::inverse, aten::linalg_inv},
      {aten::true_divide, aten::div},
      {aten::true_divide_, aten::div_},
      {aten::concat, aten::cat},
      {aten::concatenate, aten::cat},
      {aten::row_stack, aten::vstack},
      {aten::swapdims, aten::transpose},
      {aten::swapdims_, aten::transpose_},
      {aten::swapaxes, aten::transpose},
      {aten::swapaxes_, aten::transpose_},
      {aten::moveaxis, aten::movedim},
      {aten::special_erf, aten::erf},
      {aten::special_erfc, aten::erfc},
      {aten::special_erfinv, aten::erfinv},
      {aten::special_expit, aten::sigmoid},
      {aten::special_exp2, aten::exp2},
      {aten::special_expm1, aten::expm1},
      {aten::special_logit, aten::logit},
      {aten::special_logsumexp, aten::logsumexp},
      {aten::special_round, aten::round},
      {aten::special_log1p, aten::log1p},
      {aten::special_sinc, aten::sinc},
      {aten::special_digamma, aten::digamma},
      {aten::special_psi, aten::digamma},
      {aten::special_i0, aten::i0},
      {aten::special_xlogy, aten::xlogy},
      {aten::special_log_softmax, aten::log_softmax},
      {aten::orgqr, aten::linalg_householder_product},
      {aten::adjoint, aten::mH},
      {aten::special_multigammaln, aten::mvlgamma},
      {aten::special_polygamma, aten::polygamma},
      {aten::special_softmax, aten::softmax},
      {aten::special_gammainc, aten::igamma},
      {aten::special_gammaincc, aten::igammac},
      {aten::special_gammaln, aten::lgamma},
      {aten::mkldnn_rnn_layer, aten::onednn_rnn_layer},
      {aten::mkldnn_rnn_layer_backward, aten::onednn_rnn_layer_backward}};
  return alias_map;
}

void NormalizeOps(const std::shared_ptr<Graph>& graph) {
  NormalizeOps(graph->block());
}

} // namespace torch::jit
