#ifndef NNC_INCLUDE_TENSOR_H_INCLUDED__
#define NNC_INCLUDE_TENSOR_H_INCLUDED__

#include <vector>

#include "expr.h"
#include "function.h"
#include "refcount.h"

namespace torch {
namespace jit {
namespace compiler {

class TensorNode : public RefCounted {
 public:
  TensorNode(const Function& function, int output_index)
      : function_(function), output_index_(output_index) {}

  int ndim() const { return function_.ndim(); }
  const Expr& dim(int index) const { return function_.dim(index); }
  const Function& function() const { return function_; }
  int output_index() const { return output_index_; }

 private:
  Function function_;
  int output_index_;
};

class Tensor : public RefHandle<TensorNode> {
 public:
  using BaseClass = RefHandle<TensorNode>;
  Tensor(const Function& function, int output_index)
      : BaseClass(new TensorNode(function, output_index)) {}

  int ndim() const { return node()->ndim(); }
  const Expr& dim(int index) const { return node()->dim(index); }
  const Function& function() const { return node()->function(); }
  int output_index() const { return node()->output_index(); }
};

Tensor Compute(const std::vector<Expr>& dims, std::vector<std::string> arg_name_hints,
               std::function<Expr(const Var&)> body_func);
Tensor Compute(const std::vector<Expr>& dims, std::vector<std::string> arg_name_hints,
               std::function<Expr(const Var&, const Var&)> body_func);
Tensor Compute(const std::vector<Expr>& dims, std::vector<std::string> arg_name_hints,
               std::function<Expr(const Var&, const Var&, const Var&)> body_func);
Tensor Compute(const std::vector<Expr>& dims, std::vector<std::string> arg_name_hints,
               std::function<Expr(const Var&, const Var&, const Var&, const Var&)> body_func);
Tensor Compute(const std::vector<Expr>& dims, std::vector<std::string> arg_name_hints,
               std::function<Expr(const std::vector<Var>&)> body_func);

} // namespace compiler
} // namespace jit
} // namespace torch

#endif  // NNC_INCLUDE_TENSOR_H_INCLUDED__
