#include <torch/csrc/jit/tensorexpr/ir_verifier.h>

#include <torch/csrc/jit/tensorexpr/ir.h>

namespace torch::jit::tensorexpr {

namespace detail {
template <typename T>
void deducer(BinaryOpNode<T>);

bool deducer(...);
} // namespace detail

template <
    typename D,
    std::enable_if_t<
        std::is_same_v<decltype(detail::deducer(std::declval<D>())), void>>* =
        nullptr>
static void verifyBitwiseOp(NodePtr<D> v, IRVerifier* verifier) {
  TORCH_CHECK(v->lhs()->dtype().is_integral(), "UNSUPPORTED DTYPE");
  TORCH_CHECK(
      v->lhs()->dtype() == v->rhs()->dtype(),
      "MALFORMED IR: lhs/rhs dtype mismatch");
}

void IRVerifier::visit(const AndPtr& v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const OrPtr& v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const XorPtr& v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const LshiftPtr& v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const RshiftPtr& v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const ModPtr& v) {
  TORCH_CHECK(
      v->dtype().is_integral() || v->dtype().is_floating_point(),
      "invalid dtype: " + std::to_string(v->dtype()));
  IRVisitor::visit(v);
}

void IRVerifier::visit(const CompareSelectPtr& v) {
  TORCH_CHECK(
      v->ret_val1()->dtype() == v->ret_val2()->dtype(),
      "MALFORMED IR: bad dtype in CompareSelect");
  TORCH_CHECK(
      v->lhs()->dtype() == v->rhs()->dtype(),
      "MALFORMED IR: bad dtype in CompareSelect");
  IRVisitor::visit(v);
}

void IRVerifier::visit(const RampPtr& v) {
  TORCH_CHECK(
      v->stride()->dtype() == v->base()->dtype(),
      "MALFORMED IR: Bad stride in Ramp");
  IRVisitor::visit(v);
}

void IRVerifier::visit(const LoadPtr& v) {
  auto indices = v->indices();
  TORCH_CHECK(
      indices.empty() || v->buf()->base_handle()->dtype() == kHandle,
      "MALFORMED IR: Load base handle dtype must be Handle - ",
      std::to_string(v->buf()->base_handle()));

  Dtype index_dtype = !indices.empty() ? indices.at(0)->dtype() : kInt;
  if (indices.size() > 1) {
    for (size_t i = 1; i < indices.size(); ++i) {
      TORCH_CHECK(
          indices.at(i)->dtype() == index_dtype,
          "MALFORMED IR: dtype mismatch in Load indices");
    }
  }
  TORCH_CHECK(
      indices.size() <= 1 || index_dtype.lanes() <= 1,
      "MALFORMED IR: Multilane is only allowed in a flattened index");
  TORCH_CHECK(
      index_dtype.scalar_type() == ScalarType::Int ||
          index_dtype.scalar_type() == ScalarType::Long,
      "MALFORMED IR: Index scalar dtype is not Int or Long!");

  IRVisitor::visit(v);
}

void IRVerifier::visit(const IfThenElsePtr& v) {
  TORCH_CHECK(v->condition()->dtype().is_integral(), "UNSUPPORTED DTYPE");
  TORCH_CHECK(v->condition()->dtype().lanes() == 1, "UNSUPPORTED DTYPE");
  TORCH_CHECK(
      v->true_value()->dtype() == v->false_value()->dtype(),
      "MALFORMED IR: Bad dtype in IfThenElse");
  IRVisitor::visit(v);
}

void IRVerifier::visit(const IntrinsicsPtr& v) {
  if (v->op_type() == kIsNan) {
    TORCH_CHECK(
        v->dtype().scalar_type() == c10::kInt,
        "MALFORMED IR: bad dtype in intrinsic arg");
    IRVisitor::visit(v);
    return;
  }
  // TODO: add a check for OpArgCount and op_type
  for (auto const& param : v->params()) {
    TORCH_CHECK(
        param->dtype() == v->dtype(),
        "MALFORMED IR: bad dtype in intrinsic arg");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const StorePtr& v) {
  auto indices = v->indices();
  TORCH_CHECK(
      indices.empty() || v->buf()->base_handle()->dtype() == kHandle,
      "MALFORMED IR: Store base handle dtype must be Handle - ",
      std::to_string(v->buf()->base_handle()));

  Dtype index_dtype = !indices.empty() ? indices.at(0)->dtype() : kInt;
  if (indices.size() > 1) {
    for (size_t i = 1; i < indices.size(); ++i) {
      TORCH_CHECK(
          indices.at(i)->dtype() == index_dtype,
          "MALFORMED IR: dtype mismatch in Store indices");
    }
  }
  TORCH_CHECK(
      indices.size() <= 1 || index_dtype.lanes() <= 1,
      "MALFORMED IR: Multilane is only allowed in a flattened index");
  TORCH_CHECK(
      index_dtype.scalar_type() == ScalarType::Int ||
          index_dtype.scalar_type() == ScalarType::Long,
      "MALFORMED IR: Index scalar dtype is not Int or Long!");
  TORCH_CHECK(
      v->buf()->dtype() == v->value()->dtype(),
      "MALFORMED IR: buf and value dtype mismatch in Store");

  IRVisitor::visit(v);
}

void IRVerifier::visit(const ForPtr& v) {
  TORCH_CHECK(v->var(), "MALFORMED IR: nullptr Var in For loop");
  TORCH_CHECK(v->start(), "MALFORMED IR: nullptr Start in For loop");
  TORCH_CHECK(v->stop(), "MALFORMED IR: nullptr Stop in For loop");
  TORCH_CHECK(v->body(), "MALFORMED IR: invalid Body in For loop");
  IRVisitor::visit(v);
}

void IRVerifier::visit(const BlockPtr& v) {
  for (const StmtPtr& s : v->stmts()) {
    TORCH_CHECK(
        s->get_parent() == v,
        "MALFORMED IR: Broken child-parent link inside a Block");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const ExternalCallPtr& v) {
  IRVisitor::visit(v);
}

void verify(const StmtPtr& s) {
  IRVerifier verifier;
  s->accept(&verifier);
}

void verify(const ExprPtr& e) {
  IRVerifier verifier;
  e->accept(&verifier);
}

void verify(const ExprHandle& e) {
  verify(e.node());
}

} // namespace torch::jit::tensorexpr
