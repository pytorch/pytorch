#pragma once

#include <string>
#include <vector>

#include "torch/csrc/jit/compiler/include/expr.h"

namespace torch {
namespace jit {
namespace compiler {

enum IRNodeType {
  kAdd,
  kSub,
  kMul,
  kDiv,
};

class Cast : public ExprNode<Cast> {
 public:
  const Expr& src_value() const {
    return src_value_;
  }
  static Expr make(Dtype dtype, const Expr& src_value) {
    return Expr(new Cast(dtype, src_value));
  }

 private:
  Cast(Dtype dtype, const Expr& src_value)
      : ExprNodeBase(dtype), src_value_(src_value) {}
  Expr src_value_;
};

template <typename T>
Expr cast(const Expr& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  const Expr& lhs() const {
    return this->lhs_;
  }
  const Expr& rhs() const {
    return this->rhs_;
  }
  IRNodeType expr_type() const {
    return expr_type_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) {
    return Expr(new Op(lhs, rhs));
  }

 protected:
  BinaryOpNode(const Expr& lhs_v, const Expr& rhs_v, IRNodeType expr_type)
      : ExprNode<Op>(BinaryOpDtype(lhs_v.dtype(), rhs_v.dtype())),
        lhs_(CastIfNeeded(lhs_v, ExprNode<Op>::dtype())),
        rhs_(CastIfNeeded(rhs_v, ExprNode<Op>::dtype())),
        expr_type_(expr_type) {}

 private:
  static Expr CastIfNeeded(const Expr& expr, Dtype dst_dtype) {
    if (expr.dtype() == dst_dtype) {
      return expr;
    }
    return Cast::make(dst_dtype, expr);
  }

  Expr lhs_;
  Expr rhs_;
  IRNodeType expr_type_;
};

class Add : public BinaryOpNode<Add> {
 private:
  Add(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kAdd) {}
  friend class BinaryOpNode<Add>;
};

class Sub : public BinaryOpNode<Sub> {
 private:
  Sub(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kSub) {}
  friend class BinaryOpNode<Sub>;
};

class Mul : public BinaryOpNode<Mul> {
 private:
  Mul(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMul) {}
  friend class BinaryOpNode<Mul>;
};

class Div : public BinaryOpNode<Div> {
 private:
  Div(const Expr& lhs, const Expr& rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kDiv) {}
  friend class BinaryOpNode<Div>;
};

// Encode an integer immediate value.
class IntImm : public ExprNode<IntImm> {
 public:
  int value() const {
    return value_;
  }
  static Expr make(int value) {
    return Expr(new IntImm(value));
  }

 private:
  IntImm(int value) : ExprNodeBase(kInt32), value_(value) {}
  int value_;
};

// Encode an fp32 immediate value.
class FloatImm : public ExprNode<FloatImm> {
 public:
  float value() const {
    return value_;
  }
  static Expr make(float value) {
    return Expr(new FloatImm(value));
  }

 private:
  FloatImm(float value) : ExprNodeBase(kFloat32), value_(value) {}
  float value_;
};

// The underlying representation node to a Variable.
// Currently, each Variable object represents a unique variable, even though the
// names might be the same. We should consider add a unique_name as well.
class Variable : public ExprNode<Variable> {
 public:
  static Expr make(const std::string& name_hint, Dtype dtype) {
    return Expr(new Variable(name_hint, dtype));
  }
  static Expr make(Dtype dtype) {
    return Expr(new Variable("", dtype));
  }

  // TODO: unique_name
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  Variable(const std::string& name_hint, Dtype dtype)
      : ExprNodeBase(dtype), name_hint_(name_hint) {}
  std::string name_hint_;
};

// An expression to construct the underlying variable node.
// Note: do not store any info here, since it is often possible to slice this
// object. For example: Var x('x'); Expr x2 = x;
class Var : public Expr {
 public:
  Var() : Expr(nullptr) {}
  explicit Var(Dtype dtype) : Expr(Variable::make(dtype)) {}
  Var(const std::string& name_hint, Dtype dtype)
      : Expr(Variable::make(name_hint, dtype)) {}
  explicit Var(Variable* node) : Expr(node) {}
  const Variable* node() const {
    return static_cast<const Variable*>(Expr::node());
  }
  bool operator==(const Var& other) const {
    return this->node() == other.node();
  }
  bool operator!=(const Var& other) const {
    return !(*this == other);
  }

  const std::string& name_hint() const {
    return this->node()->name_hint();
  }
  bool is_null() const {
    return (this->node() == nullptr);
  }
};

// Bind the value to the var and evaluate the body.
class Let : public ExprNode<Let> {
 public:
  const Expr& var() const {
    return var_;
  }
  const Expr& value() const {
    return value_;
  }
  const Expr& body() const {
    return body_;
  }

  static Expr make(const Expr& var, const Expr& value, const Expr& body) {
    return Expr(new Let(var, value, body));
  }

 private:
  Let(const Expr& var, const Expr& value, const Expr& body)
      : ExprNodeBase(body.dtype()), var_(var), value_(value), body_(body) {}

  Expr var_;
  Expr value_;
  Expr body_;
};

class Block : public StmtNode<Block> {
 public:
  static Stmt make(const std::vector<Stmt>& stmts) {
    return Stmt(new Block(stmts));
  }
  int nstmts() const {
    return stmts_.size();
  }
  const Stmt& stmt(int index) const {
    return stmts_[index];
  }

 private:
  explicit Block(const std::vector<Stmt>& stmts) : stmts_(stmts) {}
  std::vector<Stmt> stmts_;
};

class For : public StmtNode<For> {
 public:
  const Var& var() const {
    return var_;
  }
  const Expr& start() const {
    return start_;
  }
  const Expr& stop() const {
    return stop_;
  }
  const Stmt& body() const {
    return body_;
  }
  static Stmt make(
      const Var& var,
      const Expr& start,
      const Expr& stop,
      const Stmt& body) {
    return Stmt(new For(var, start, stop, body));
  }

 private:
  For(const Var& var, const Expr& start, const Expr& stop, const Stmt& body)
      : var_(var), start_(start), stop_(stop), body_(body) {}
  Var var_;
  Expr start_;
  Expr stop_;
  Stmt body_;
};

// Represents a ramp vector node:
//     [base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class Ramp : public ExprNode<Ramp> {
 public:
  const Expr& base() const {
    return base_;
  }
  const Expr& stride() const {
    return stride_;
  }
  static Expr make(const Expr& base, const Expr& stride, int lanes) {
    return Expr(new Ramp(base, stride, lanes));
  }
  int lanes() const {
    return lanes_;
  }

 private:
  Ramp(const Expr& base, const Expr& stride, int lanes)
      : ExprNodeBase(Dtype(base.dtype(), lanes)),
        base_(base),
        stride_(stride),
        lanes_(lanes) {
    CHECK_EQ(stride.dtype(), base.dtype());
  }

  Expr base_;
  Expr stride_;
  int lanes_;
};

class Buffer {
 public:
  Buffer(const Var& data, const Dtype& dtype, const std::vector<Expr>& dims)
      : data_(data), dtype_(dtype), dims_(dims) {
    CHECK_EQ(data.dtype(), kHandle);
  }
  const Var& data() const {
    return data_;
  }
  const Dtype& dtype() const {
    return dtype_;
  }
  int ndim() const {
    return dims_.size();
  }
  const Expr& dim(int index) const {
    return dims_[index];
  }

 private:
  Var data_;
  Dtype dtype_;
  std::vector<Expr> dims_;
  // TODO: add strides
};

class Load : public ExprNode<Load> {
 public:
  const Var& base_handle() const {
    return base_handle_;
  }
  const Expr& index() const {
    return index_;
  }
  const Expr& mask() const {
    return mask_;
  }
  static Expr make(const Buffer& buffer, const Expr& index, const Expr& mask) {
    return Expr(new Load(buffer, index, mask));
  }
  static Expr make(
      Dtype dtype,
      const Var& base_handle,
      const Expr& index,
      const Expr& mask) {
    return Expr(new Load(dtype, base_handle, index, mask));
  }

 private:
  Load(const Buffer& buffer, const Expr& index, const Expr& mask)
      : Load(
            ChooseDtype(buffer.dtype(), index.dtype()),
            buffer.data(),
            index,
            mask) {}
  Load(Dtype dtype, const Var& base_handle, const Expr& index, const Expr& mask)
      : ExprNodeBase(dtype),
        base_handle_(base_handle),
        index_(index),
        mask_(mask) {
    CHECK_EQ(base_handle_.dtype(), kHandle);
    CHECK_EQ(index.dtype().lanes(), mask.dtype().lanes());
    CHECK_EQ(index.dtype().scalar_type(), kInt32);
  }
  static Dtype ChooseDtype(
      const Dtype& buffer_dtype,
      const Dtype& index_dtype) {
    return Dtype(buffer_dtype, index_dtype.lanes());
  }

  Var base_handle_;
  Expr index_;
  Expr mask_;
};

class Store : public StmtNode<Store> {
 public:
  const Var& base_handle() const {
    return base_handle_;
  }
  const Expr& index() const {
    return index_;
  }
  const Expr& value() const {
    return value_;
  }
  const Expr& mask() const {
    return mask_;
  }

  static Stmt make(
      const Buffer& buffer,
      const Expr& index,
      const Expr& value,
      const Expr& mask) {
    return Stmt(new Store(buffer, index, value, mask));
  }

  static Stmt make(
      const Var& base_handle,
      const Expr& index,
      const Expr& value,
      const Expr& mask) {
    return Stmt(new Store(base_handle, index, value, mask));
  }

 private:
  // TODO: merge this with Load.
  Store(
      const Buffer& buffer,
      const Expr& index,
      const Expr& value,
      const Expr& mask)
      : Store(buffer.data(), index, value, mask) {
    CHECK_EQ(buffer.dtype().scalar_type(), value.dtype().scalar_type());
    CHECK_EQ(buffer.dtype().scalar_type(), value.dtype().scalar_type());
  }

  Store(
      const Var& base_handle,
      const Expr& index,
      const Expr& value,
      const Expr& mask)
      : base_handle_(base_handle), index_(index), value_(value), mask_(mask) {
    CHECK_EQ(base_handle_.dtype(), kHandle);
    CHECK_EQ(index.dtype().lanes(), mask.dtype().lanes());
    CHECK_EQ(index.dtype().lanes(), value.dtype().lanes());
    CHECK_EQ(index.dtype().scalar_type(), kInt32);
  }

  Var base_handle_;
  Expr index_;
  Expr value_;
  Expr mask_;
};

class Broadcast : public ExprNode<Broadcast> {
 public:
  const Expr& value() const {
    return value_;
  }
  int lanes() const {
    return lanes_;
  }
  static Expr make(const Expr& value, int lanes) {
    return Expr(new Broadcast(value, lanes));
  }

 private:
  Broadcast(const Expr& value, int lanes)
      : ExprNodeBase(Dtype(value.dtype(), lanes)),
        value_(value),
        lanes_(lanes) {}
  Expr value_;
  int lanes_;
};

} // namespace compiler
} // namespace jit
} // namespace torch
