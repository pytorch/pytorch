
#include <torch/csrc/jit/codegen/cuda/ir_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

namespace torch {
namespace jit {
namespace fuser {

UnaryOp::UnaryOp(UnaryOpType _type, Val* _out, Val* _in)
    : Expr(ExprType::UnaryOp), unary_op_type_{_type}, out_{_out}, in_{_in} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BinaryOp::BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs)
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
    Int* _index,
    IterDomain* _range,
    const std::vector<const Expr*>& _body)
    : Expr(ExprType::ForLoop), index_{_index}, range_{_range}, body_{_body} {
  addInput(_index);
  addInput(_range);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

IfThenElse::IfThenElse(
    Val* _cond,
    const std::vector<const Expr*>& _if_body,
    const std::vector<const Expr*>& _else_body)
    : Expr(ExprType::IfThenElse),
      cond_{_cond},
      if_body_{_if_body},
      else_body_{_else_body} {
  addInput(_cond);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Split::Split(TensorDomain* _out, TensorDomain* _in, int _axis, Int* _factor)
    : Expr(ExprType::Split),
      out_{_out},
      in_{_in},
      axis_{_axis},
      factor_{_factor} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Merge::Merge(TensorDomain* _out, TensorDomain* _in, int _axis)
    : Expr(ExprType::Merge), out_{_out}, in_{_in}, axis_{_axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Reorder::Reorder(
    TensorDomain* _out,
    TensorDomain* _in,
    std::vector<int> _pos2axis)
    : Expr(ExprType::Reorder), out_{_out}, in_{_in}, pos2axis_{_pos2axis} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}


} // namespace fuser
} // namespace jit
} // namespace torch