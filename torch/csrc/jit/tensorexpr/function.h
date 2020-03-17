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
  Range(const ExprHandle& start, const ExprHandle& stop)
      : start_(start), stop_(stop) {}
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
      : func_vars_({VarHandle(func_name, kHandle).node()}),
        dims_(dims),
        args_(args),
        bodies_({body}) {}
  Function(
      const std::vector<std::string>& func_names,
      const std::vector<const Expr*>& dims,
      const std::vector<const Var*>& args,
      const std::vector<const Expr*>& bodies)
      : func_vars_(func_names.size()),
        dims_(dims),
        args_(args),
        bodies_(bodies) {
    for (size_t i = 0; i < func_names.size(); i++) {
      func_vars_[i] = new Var(func_names[i], kHandle);
    }
  }

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

  std::vector<const Expr*> bodies() const {
    return bodies_;
  }
  const Expr* body(size_t index) const {
    CHECK(index < bodies_.size());
    return bodies_[index];
  }

  std::vector<const Var*> func_vars() const {
    return func_vars_;
  }
  const Var* func_var(size_t index) const {
    CHECK(index < func_vars_.size());
    return func_vars_[index];
  }

  Stmt* ElementStmt(size_t index);

 private:
  std::vector<const Var*> func_vars_;
  std::vector<const Expr*> dims_;
  std::vector<const Var*> args_;
  std::vector<const Expr*> bodies_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
