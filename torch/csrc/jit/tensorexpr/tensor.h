#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/function.h"

namespace torch {
namespace jit {
namespace tensorexpr {
namespace schedule {
class TensorExprNode;
class ScheduleNode;
} // namespace schedule

using schedule::TensorExprNode;

class TORCH_API TensorOperation : public KernelScopedObject {
 public:
  void SplitWithTail(
      const Var& loop_var,
      int factor,
      bool factor_on_inner,
      Var* outer_var,
      Var* inner_var,
      Var* tail_var,
      TensorOperation** tail_op);

  void SplitWithMask(
      const Var& loop_var,
      int factor,
      bool factor_on_inner,
      Var* outer_var,
      Var* inner_var);

  void ComputeInline();

  void GPUExecConfig(
      const std::vector<Var>& blockIdx,
      const std::vector<Var>& threadIdx);

  TensorExprNode* expr_node() {
    return expr_node_;
  }

 protected:
  TensorOperation() {}
  explicit TensorOperation(TensorExprNode* expr_node) : expr_node_(expr_node) {}

 private:
  void check_expr_node();

  friend class schedule::ScheduleNode;
  TensorExprNode* expr_node_ = nullptr;
};

class Tensor : public TensorOperation {
 public:
  Function* function() const {
    return function_;
  }
  int output_index() const {
    return output_index_;
  }
  const Var& arg(int index) const {
    return function_->arg(index);
  }

  Tensor(Function* function, int output_index)
      : function_(function), output_index_(output_index) {}
  template <typename... Ts>
  inline Expr operator()(const Ts&... ts);
  template <typename T>
  inline Expr call(const std::vector<T>& args);
  template <typename... Ts>
  inline Expr call(const Ts&... ts);

 private:
  Function* function_;
  int output_index_;
};

// A helper structure to store the arguments to specify dimensions. In the
// Compute arugments for dim_args, all of the following is supported. For
// example:
//    dim_args: {1, 2, 3, 4}
//    dim_args: {{1, "x"}, {2, "y"}, {3, "z"}}
//    dim_args: {1, 2, {3, "x"}}
class DimArg {
 public:
  // Intentionally leave out explicit to allow implicit conversions.
  DimArg(const Expr& dim) : dim_(dim) {}
  DimArg(const Expr& dim, const std::string& name_hint)
      : dim_(dim), name_hint_(name_hint) {}
  const Expr& dim() const {
    return dim_;
  }
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  Expr dim_;
  std::string name_hint_;
};

TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&)> body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&)> body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&, const Var&)> body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&, const Var&, const Var&)>
        body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const std::vector<Var>&)> body_func);

class FunctionCall : public CallNode<FunctionCall> {
 public:
  using BaseClass = CallNode<FunctionCall>;
  static Expr make(Tensor* tensor, const std::vector<Expr>& params) {
    return Expr(new FunctionCall(tensor, params));
  }

  const Tensor* tensor() const {
    return tensor_;
  }
  Tensor* tensor() {
    return tensor_;
  }

 private:
  Expr DefaultMutator(const std::vector<Expr>& new_params) const override {
    return FunctionCall::make(tensor_, new_params);
  }

  std::string func_name() const {
    return tensor_->function()->func_var().name_hint();
  }

  FunctionCall(Tensor* tensor, const std::vector<Expr>& params)
      : BaseClass(tensor->function()->body().dtype(), kFunctionCall, params),
        tensor_(tensor) {}
  Tensor* tensor_;
};
template <typename... Ts>
inline Expr Tensor::operator()(const Ts&... ts) {
  std::vector<Expr> params({Expr(ts)...});
  return FunctionCall::make(this, std::move(params));
}

template <typename... Ts>
inline Expr Tensor::call(const Ts&... ts) {
  std::vector<Expr> params({Expr(ts)...});
  return FunctionCall::make(this, std::move(params));
}

template <typename T>
inline Expr Tensor::call(const std::vector<T>& args) {
  std::vector<Expr> params(args.begin(), args.end());
  return FunctionCall::make(this, params);
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
