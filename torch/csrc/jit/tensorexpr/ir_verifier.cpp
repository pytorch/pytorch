#include <torch/csrc/jit/tensorexpr/ir_verifier.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

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
void verifyBitwiseOp(NodePtr<D> v, IRVerifier* verifier) {
  if (!v->lhs()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  if (v->lhs()->dtype() != v->rhs()->dtype()) {
    throw malformed_ir("lhs/rhs dtype mismatch");
  }
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
  if (!v->dtype().is_integral() && !v->dtype().is_floating_point()) {
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const CompareSelectPtr& v) {
  if (v->ret_val1()->dtype() != v->ret_val2()->dtype()) {
    throw malformed_ir("bad dtype in CompareSelect");
  }
  if (v->lhs()->dtype() != v->rhs()->dtype()) {
    throw malformed_ir("bad dtype in CompareSelect");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const RampPtr& v) {
  if (v->stride()->dtype() != v->base()->dtype()) {
    throw malformed_ir("Bad stride in Ramp");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const LoadPtr& v) {
  auto indices = v->indices();
  if (!indices.empty() && v->buf()->base_handle()->dtype() != kHandle) {
    throw malformed_ir(
        "Load base handle dtype must be Handle", v->buf()->base_handle());
  }

  Dtype index_dtype = !indices.empty() ? indices.at(0)->dtype() : kInt;
  if (indices.size() > 1) {
    for (size_t i = 1; i < indices.size(); ++i) {
      if (indices.at(i)->dtype() != index_dtype) {
        throw malformed_ir("dtype mismatch in Load indices");
      }
    }
  }
  if (indices.size() > 1 && index_dtype.lanes() > 1) {
    throw malformed_ir("Multilane is only allowed in a flattened index");
  }
  if (index_dtype.scalar_type() != ScalarType::Int &&
      index_dtype.scalar_type() != ScalarType::Long) {
    throw malformed_ir("Index scalar dtype is not Int or Long!");
  }

  IRVisitor::visit(v);
}

void IRVerifier::visit(const IfThenElsePtr& v) {
  if (!v->condition()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  if (v->condition()->dtype().lanes() != 1) {
    throw unsupported_dtype();
  }
  if (v->true_value()->dtype() != v->false_value()->dtype()) {
    throw malformed_ir("Bad dtype in IfThenElse");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const IntrinsicsPtr& v) {
  if (v->op_type() == kIsNan) {
    if (v->dtype().scalar_type() != c10::kInt) {
      throw malformed_ir("bad dtype in intrinsic arg");
    }
    IRVisitor::visit(v);
    return;
  }
  // TODO: add a check for OpArgCount and op_type
  for (auto const& param : v->params()) {
    if (param->dtype() != v->dtype()) {
      throw malformed_ir("bad dtype in intrinsic arg");
    }
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const StorePtr& v) {
  auto indices = v->indices();
  if (!indices.empty() && v->buf()->base_handle()->dtype() != kHandle) {
    throw malformed_ir(
        "Store base handle dtype must be Handle", v->buf()->base_handle());
  }

  Dtype index_dtype = !indices.empty() ? indices.at(0)->dtype() : kInt;
  if (indices.size() > 1) {
    for (size_t i = 1; i < indices.size(); ++i) {
      if (indices.at(i)->dtype() != index_dtype) {
        throw malformed_ir("dtype mismatch in Store indices");
      }
    }
  }
  if (indices.size() > 1 && index_dtype.lanes() > 1) {
    throw malformed_ir("Multilane is only allowed in a flattened index");
  }
  if (index_dtype.scalar_type() != ScalarType::Int &&
      index_dtype.scalar_type() != ScalarType::Long) {
    throw malformed_ir("Index scalar dtype is not Int or Long!");
  }
  if (v->buf()->dtype() != v->value()->dtype()) {
    throw malformed_ir("buf and value dtype mismatch in Store");
  }

  IRVisitor::visit(v);
}

void IRVerifier::visit(const ForPtr& v) {
  if (!v->var()) {
    throw malformed_ir("nullptr Var in For loop");
  } else if (!v->start()) {
    throw malformed_ir("nullptr Start in For loop");
  } else if (!v->stop()) {
    throw malformed_ir("nullptr Stop in For loop");
  } else if (!v->body()) {
    throw malformed_ir("invalid Body in For loop");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const BlockPtr& v) {
  for (const StmtPtr& s : v->stmts()) {
    if (s->get_parent() != v) {
      throw malformed_ir("Broken child-parent link inside a Block");
    }
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const ExternalCallPtr& v) {
  IRVisitor::visit(v);
}

void verify(StmtPtr s) {
  IRVerifier verifier;
  s->accept(&verifier);
}

void verify(ExprPtr e) {
  IRVerifier verifier;
  e->accept(&verifier);
}

void verify(ExprHandle e) {
  verify(e.node());
}

} // namespace torch::jit::tensorexpr
