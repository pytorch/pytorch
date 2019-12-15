#ifndef NNC_INCLUDE_FUNCTION_H_INCLUDED__
#define NNC_INCLUDE_FUNCTION_H_INCLUDED__

#include <functional>
#include <vector>

#include "expr.h"
#include "ir.h"
#include "refcount.h"

namespace nnc {

// represent a range [start, stop)
class Range {
 public:
  Range(const Expr& start, const Expr& stop) : start_(start), stop_(stop) {}
  const Expr& start() const { return start_; }
  const Expr& stop() const { return stop_; }

 private:
  Expr start_;
  Expr stop_;
};

class FunctionNode : public RefCounted {
 public:
  FunctionNode(const std::vector<Expr>& dims, const std::vector<Var>& args, const Expr& body)
      : dims_(dims), args_(args), body_(body) {}

  int ndim() const { return dims_.size(); }
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
  const Expr& body() const { return body_; }

 private:
  std::vector<Expr> dims_;
  std::vector<Var> args_;
  Expr body_;
};

class Function : public RefHandle<FunctionNode> {
 public:
  using BaseClass = RefHandle<FunctionNode>;
  Function(const std::vector<Expr>& dims, const std::vector<Var>& args, const Expr& body)
      : BaseClass(new FunctionNode(dims, args, body)) {}
  int ndim() const { return node()->ndim(); }
  const Expr& dim(int index) const { return node()->dim(index); }
  const Var& arg(int index) const { return node()->arg(index); }
  const Expr& body() const { return node()->body(); }
};

}  // namespace nnc

#endif  // NNC_INCLUDE_FUNCTION_H_INCLUDED__
