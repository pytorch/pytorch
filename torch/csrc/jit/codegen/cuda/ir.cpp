#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir.h>

#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/ir.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Statement member definitions & related functions
 */

// When we create a Val or EXPR we immediately register them with the active
// fusion.
Val::Val(ValType _vtype, DataType _dtype)
    : vtype_{_vtype}, dtype_{_dtype} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion != nullptr) {
    this->name_ = fusion->registerVal(this);
    this->fusion_ = fusion;
  } else {
    TORCH_CHECK(false, "No active fusion group found when creating a Val.");
  }
}

Expr* Val::getOrigin() {
  FusionGuard fg(fusion_);
  return (fusion_->origin(this));
}

Expr::Expr(ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    TORCH_CHECK(false, "No active fusion group found when creating an Expr.");
  this->fusion_ = fusion;
}

UnaryOp::UnaryOp(UnaryOpType _type, Val* _out, Val* _in)
    : Expr(ExprType::UnaryOp), unary_op_type_{_type}, out_{_out}, in_{_in} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BinaryOp::BinaryOp(
    BinaryOpType _type,
    Val* _out,
    Val* _lhs,
    Val* _rhs)
    : Expr(ExprType::BinaryOp),
      binary_op_type_{_type},
      out_{_out},
      lhs_{_lhs},
      rhs_{_rhs} {
  addOutput(_out);
  addInput(_lhs);
  addInput(_rhs);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

ForLoop::ForLoop(
    Int*        _index,
    IterDomain* _range,
    const std::vector<const Expr*> &_body)
    : Expr(ExprType::ForLoop),
      index_{_index},
      range_{_range},
      body_{_body}
{
  addInput(_index);
  addInput(_range);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

IfThenElse::IfThenElse(
    Val* _cond,
    const std::vector<const Expr*> &_if_body,
    const std::vector<const Expr*> &_else_body)
    : Expr(ExprType::IfThenElse),
      cond_{_cond},
      if_body_{_if_body},
      else_body_{_else_body}
{
  addInput(_cond);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Statement::~Statement() {}

/*
 * Val member definitions
 */

Val::~Val() {}

/*
 * IRInputOutput member definitions
 */

IRInputOutput::~IRInputOutput() {}

/*
 * Expr member definitions
 */

Expr::~Expr() {}

} // namespace fuser
} // namespace jit
} // namespace torch
