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
      const std::vector<ExprHandle>& dims,
      const std::vector<VarHandle>& args,
      const ExprHandle& body)
      : func_var_(func_name, kHandle), dims_(dims), args_(args), body_(body) {}

  int ndim() const {
    return dims_.size();
  }
  const ExprHandle& dim(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return dims_[index];
  }
  const std::vector<ExprHandle>& dims() const {
    return dims_;
  }
  const VarHandle& arg(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return args_[index];
  }
  const std::vector<VarHandle>& args() const {
    return args_;
  }
  const ExprHandle& body() const {
    return body_;
  }
  const VarHandle& func_var() const {
    return func_var_;
  }
  Stmt* ElementStmt();

 private:
  VarHandle func_var_;
  std::vector<ExprHandle> dims_;
  std::vector<VarHandle> args_;
  ExprHandle body_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
