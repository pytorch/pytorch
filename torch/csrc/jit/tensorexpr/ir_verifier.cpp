#include <torch/csrc/jit/tensorexpr/ir_verifier.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
void verifyBitwiseOp(const BitwiseOpNode<Op>* v, IRVerifier* verifier) {
  if (!v->lhs()->dtype().is_integral()) {
    throw unsupported_dtype();
  }
  if (v->lhs()->dtype() != v->rhs()->dtype()) {
    throw malformed_ir("lhs/rhs dtype mismatch");
  }
}

void IRVerifier::visit(const And* v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Or* v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Xor* v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Lshift* v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Rshift* v) {
  verifyBitwiseOp(v, this);
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Mod* v) {
  if (!v->dtype().is_integral() && !v->dtype().is_floating_point()) {
    throw std::runtime_error("invalid dtype: " + std::to_string(v->dtype()));
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const CompareSelect* v) {
  if (v->ret_val1()->dtype() != v->ret_val2()->dtype()) {
    throw malformed_ir("bad dtype in CompareSelect");
  }
  if (v->lhs()->dtype() != v->rhs()->dtype()) {
    throw malformed_ir("bad dtype in CompareSelect");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Ramp* v) {
  if (v->stride()->dtype() != v->base()->dtype()) {
    throw malformed_ir("Bad stride in Ramp");
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Load* v) {
  const auto indices = v->indices();
  if (indices.size() > 0 && v->buf()->base_handle()->dtype() != kHandle) {
    throw malformed_ir(
        "Load base handle dtype must be Handle", v->buf()->base_handle());
  }

  Dtype index_dtype = indices.size() ? indices.at(0)->dtype() : kInt;
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
  if (index_dtype.scalar_type() != ScalarType::Int) {
    throw malformed_ir("Index scalar dtype is not Int!");
  }

  IRVisitor::visit(v);
}

void IRVerifier::visit(const IfThenElse* v) {
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

void IRVerifier::visit(const Intrinsics* v) {
  // TODO: add a check for OpArgCount and op_type
  IRVisitor::visit(v);
}

void IRVerifier::visit(const Store* v) {
  const auto indices = v->indices();
  if (indices.size() > 0 && v->buf()->base_handle()->dtype() != kHandle) {
    throw malformed_ir(
        "Store base handle dtype must be Handle", v->buf()->base_handle());
  }

  Dtype index_dtype = indices.size() ? indices.at(0)->dtype() : kInt;
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
  if (index_dtype.scalar_type() != ScalarType::Int) {
    throw malformed_ir("Index scalar dtype is not Int!");
  }
  if (v->buf()->dtype() != v->value()->dtype()) {
    throw malformed_ir("buf and value dtype mismatch in Store");
  }

  IRVisitor::visit(v);
}

void IRVerifier::visit(const For* v) {
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

void IRVerifier::visit(const Block* v) {
  for (Stmt* s : v->stmts()) {
    if (s->get_parent() != v) {
      throw malformed_ir("Broken child-parent link inside a Block");
    }
  }
  IRVisitor::visit(v);
}

void IRVerifier::visit(const ExternalCall* v) {
  IRVisitor::visit(v);
}

void verify(Stmt* s) {
  IRVerifier verifier;
  s->accept(&verifier);
}

void verify(const Expr* e) {
  IRVerifier verifier;
  e->accept(&verifier);
}

void verify(ExprHandle e) {
  verify(e.node());
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
