#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
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
Val::Val(const ValType _vtype, const DataType _dtype)
    : vtype_{_vtype}, dtype_{_dtype} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion != nullptr) {
    this->name_ = fusion->registerVal(this);
    this->fusion_ = fusion;
  } else {
    throw std::runtime_error("No fusion group found when creating a Val.");
  }
}

const Expr* Val::getOrigin() {
  FusionGuard fg(fusion_);
  return (fusion_->origin(this));
}

Expr::Expr(const ExprType _type) : type_{_type} {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (fusion == nullptr)
    throw std::runtime_error("No fusion group found when creating an Expr.");
  this->fusion_ = fusion;
}

UnaryOp::UnaryOp(const UnaryOpType _type, const Val* _out, const Val* _in)
    : Expr(ExprType::UnaryOp), unary_op_type_{_type}, out_{_out}, in_{_in} {
  addOutput(_out);
  addInput(_in);
  this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BinaryOp::BinaryOp(
    const BinaryOpType _type,
    const Val* _out,
    const Val* _lhs,
    const Val* _rhs)
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

Statement::~Statement() {}

template <typename T>
T* ptr(T& obj) {
  return &obj;
}

template <typename T>
T* ptr(T* obj) {
  return obj;
}

/*
 * Generic dispatch for any handler that does not modify the IR directly.
 * For example we may want to walk the graph to construct a topologically sorted
 * set of exprs. This doesn't modify the IR directly. We also use this to print
 * the IR itself.
 * This dispatch is paired with a class that implements the functions:
 * template <typenname node_type>
 * int handler(const node_type* node)
 *
 * handler should call:
 * (statement* node_to_dispatch)->dispatch()
 *
 * It could also implement:
 * int handler(Statement* stmt){
 *   stmt->dispatch(this);
 * }
 *
 * And therefore dispatch should never call:
 * ptr(mutator)->handle(static_cast<const Statement*>(this));
 */

template <typename T>
void Val::dispatch(T handler) const {
  switch (*getValType()) {
    case ValType::TensorDomain:
      ptr(handler)->handle(static_cast<const TensorDomain*>(this));
      return;
    case ValType::TensorView:
      ptr(handler)->handle(static_cast<const TensorView*>(this));
      return;
    case ValType::IterDomain:
      ptr(handler)->handle(static_cast<const IterDomain*>(this));
      return;
    case ValType::Tensor:
      ptr(handler)->handle(static_cast<const Tensor*>(this));
      return;
    case ValType::Scalar:
      switch (*getDataType()) {
        case DataType::Float:
          ptr(handler)->handle(static_cast<const Float*>(this));
          return;
        case DataType::Int:
          ptr(handler)->handle(static_cast<const Int*>(this));
          return;
        default:
          break;
      }
    default:
      break;
  }
  throw std::runtime_error("Unknown valtype in dispatch!");
}

template <typename T>
void Expr::dispatch(T handler) const {
  switch (*getExprType()) {
    case ExprType::UnaryOp:
      ptr(handler)->handle(static_cast<const UnaryOp*>(this));
      return;
    case ExprType::BinaryOp:
      ptr(handler)->handle(static_cast<const BinaryOp*>(this));
      return;
    case ExprType::Split:
      ptr(handler)->handle(static_cast<const Split*>(this));
      return;
    case ExprType::Merge:
      ptr(handler)->handle(static_cast<const Merge*>(this));
      return;
    case ExprType::Reorder:
      ptr(handler)->handle(static_cast<const Reorder*>(this));
      return;
    default:
      throw std::runtime_error("Unknown exprtype in dispatch!");
  }
}

template <typename T>
void Statement::dispatch(T handler) const {
  if (isVal()) {
    ptr(handler)->handle(static_cast<const Val*>(this));
  } else if (isExpr()) {
    ptr(handler)->handle(static_cast<const Expr*>(this));
  } else
    throw std::runtime_error("Unknown stmttype in dispatch!");
}

/*
 * Generic dispatch for any handler that modifies the IR. This could be a
 * transformation on loop structures, or parallelizing a loop. This dispatch is
 * paired with a class that implements the functions template <typenname
 * node_type> const Statement* mutate(const node_type* node) mutate should call
 * (statement* node_to_dispatch)->dispatch_mutator() It could also implement
 * const Statement* mutate(Statement* stmt){
 *   stmt->dispatch_mutator(this);
 * }
 * And therefore dispatch_mutator should never call:
 *   ptr(mutator)->mutate(static_cast<const Statement*>(this));
 */
template <typename T>
const Statement* Val::dispatch_mutator(T mutator) const {
  switch (*getValType()) {
    case ValType::Tensor:
      return ptr(mutator)->mutate(static_cast<const Tensor*>(this));

    case ValType::TensorDomain:
      return ptr(mutator)->mutate(static_cast<const TensorDomain*>(this));

    case ValType::TensorView:
      return ptr(mutator)->mutate(static_cast<const TensorView*>(this));

    case ValType::IterDomain:
      return ptr(mutator)->mutate(static_cast<const IterDomain*>(this));

    case ValType::Scalar:
      switch (*getDataType()) {
        case DataType::Float:
          return ptr(mutator)->mutate(static_cast<const Float*>(this));

        case DataType::Int:
          return ptr(mutator)->mutate(static_cast<const Int*>(this));

        default:
          break;
      }
    default:
      break;
  }
  throw std::runtime_error("Unknown valtype in dispatch_mutator!");
}

template <typename T>
const Statement* Expr::dispatch_mutator(T mutator) const {
  switch (*getExprType()) {
    case ExprType::UnaryOp:
      return ptr(mutator)->mutate(static_cast<const UnaryOp*>(this));
    case ExprType::BinaryOp:
      return ptr(mutator)->mutate(static_cast<const BinaryOp*>(this));
    case ExprType::Split:
      return ptr(mutator)->mutate(static_cast<const Split*>(this));
    case ExprType::Merge:
      return ptr(mutator)->mutate(static_cast<const Merge*>(this));
    case ExprType::Reorder:
      return ptr(mutator)->mutate(static_cast<const Reorder*>(this));
    default:
      throw std::runtime_error("Unknown exprtype in dispatch_mutator!");
  }
}

template <typename T>
const Statement* Statement::dispatch_mutator(T mutator) const {
  if (isVal()) {
    return ptr(mutator)->mutate(static_cast<const Val*>(this));
  }
  if (isExpr()) {
    return ptr(mutator)->mutate(static_cast<const Expr*>(this));
  }
  throw std::runtime_error("Unknown stmttype in dispatch_mutator!");
}

/*
 * Handler template instantiations. These should only have to be done on base classes.
 * Actual visitors/mutators should inhereit from these classes and call ->dispatch(this)
 * to avoid needing an explicit instantiation.
 */
template void Statement::dispatch(IterVisitor) const;
template void Statement::dispatch(IterVisitor*) const;
template void Val::dispatch(IterVisitor) const;
template void Val::dispatch(IterVisitor*) const;
template void Expr::dispatch(IterVisitor) const;
template void Expr::dispatch(IterVisitor*) const;

template const Statement* Statement::dispatch_mutator(BaseMutator) const;
template const Statement* Statement::dispatch_mutator(BaseMutator*) const;
template const Statement* Val::dispatch_mutator(BaseMutator) const;
template const Statement* Val::dispatch_mutator(BaseMutator*) const;
template const Statement* Expr::dispatch_mutator(BaseMutator) const;
template const Statement* Expr::dispatch_mutator(BaseMutator*) const;

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
