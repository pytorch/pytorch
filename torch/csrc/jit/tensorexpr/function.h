#pragma once

#include <functional>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// represent a range [start, stop)
class Range {
 public:
  Range() {}
  Range(const ExprHandle& start, const ExprHandle& stop) : start_(start), stop_(stop) {}
  const ExprHandle& start() const {
    return start_;
  }
  const ExprHandle& stop() const {
    return stop_;
  }

 private:
  ExprHandle start_;
  ExprHandle stop_;
};

class Function : public KernelScopedObject {
 public:
  Function(
      const std::string& func_name,
      const std::vector<const Expr*>& dims,
      const std::vector<const Var*>& args,
      const Expr* body)
      : func_var_(VarHandle(func_name, kHandle).node()), dims_(dims), args_(args), body_(body) {}

  int ndim() const {
    return dims_.size();
  }
  const Expr* dim(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return dims_[index];
  }
  const std::vector<const Expr*>& dims() const {
    return dims_;
  }
  const Var* arg(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return args_[index];
  }
  const std::vector<const Var*>& args() const {
    return args_;
  }
  const Expr* body() const {
    return body_;
  }
  const Var* func_var() const {
    return func_var_;
  }
  Stmt* ElementStmt();

 private:
  const Var* func_var_;
  std::vector<const Expr*> dims_;
  std::vector<const Var*> args_;
  const Expr* body_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
