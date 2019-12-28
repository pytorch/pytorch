#pragma once

#include <functional>
#include <vector>

#include "torch/csrc/jit/compiler/include/expr.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/refcount.h"

namespace torch {
namespace jit {
namespace compiler {

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

class FunctionNode : public RefCounted {
 public:
  FunctionNode(
      const std::string& func_name,
      const std::vector<Expr>& dims,
      const std::vector<Var>& args,
      const Expr& body)
      : func_var_(func_name, body.dtype().scalar_type()),
        dims_(dims),
        args_(args),
        body_(body) {}

  int ndim() const {
    return dims_.size();
  }
  const Expr& dim(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, dims_.size()) << "index out of upper bound";
    return dims_[index];
  }
  const Var& arg(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, dims_.size()) << "index out of upper bound";
    return args_[index];
  }
  const Expr& body() const {
    return body_;
  }
  const Var& func_var() const {
    return func_var_;
  }

 private:
  Var func_var_;
  std::vector<Expr> dims_;
  std::vector<Var> args_;
  Expr body_;
};

class Function : public RefHandle<FunctionNode> {
 public:
  using BaseClass = RefHandle<FunctionNode>;
  Function(
      const std::string& func_name,
      const std::vector<Expr>& dims,
      const std::vector<Var>& args,
      const Expr& body)
      : BaseClass(new FunctionNode(func_name, dims, args, body)) {}
  int ndim() const {
    return node()->ndim();
  }
  const Expr& dim(int index) const {
    return node()->dim(index);
  }
  const Var& arg(int index) const {
    return node()->arg(index);
  }
  const Expr& body() const {
    return node()->body();
  }
  const Var& func_var() const {
    return node()->func_var();
  }
};

} // namespace compiler
} // namespace jit
} // namespace torch
