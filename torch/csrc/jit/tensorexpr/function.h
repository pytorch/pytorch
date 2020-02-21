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
  Range(const Expr& start, const Expr& stop) : start_(start), stop_(stop) {}
  const Expr& start() const {
    return start_;
  }
  const Expr& stop() const {
    return stop_;
  }

 private:
  Expr start_;
  Expr stop_;
};

class Function : public KernelScopedObject {
 public:
  Function(
      const std::string& func_name,
      const std::vector<Expr>& dims,
      const std::vector<Var>& args,
      const Expr& body)
      : func_var_(func_name, kHandle), dims_(dims), args_(args), body_(body) {}

  int ndim() const {
    return dims_.size();
  }
  const Expr& dim(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return dims_[index];
  }
  const std::vector<Expr>& dims() const {
    return dims_;
  }
  const Var& arg(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return args_[index];
  }
  const std::vector<Var>& args() const {
    return args_;
  }
  const Expr& body() const {
    return body_;
  }
  const Var& func_var() const {
    return func_var_;
  }
  Stmt ElementStmt();

 private:
  Var func_var_;
  std::vector<Expr> dims_;
  std::vector<Var> args_;
  Expr body_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
