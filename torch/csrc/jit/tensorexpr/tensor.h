#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/function.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Tensor {
 public:
  Function* function() const {
    return function_;
  }
  int output_index() const {
    return output_index_;
  }

  // Wrappers over accessors to fields of the underlying function
  const Expr* body() const {
    return function()->body(output_index());
  }
  const Var* func_var() const {
    return function()->func_var(output_index());
  }
  int ndim() const {
    return function()->dims().size();
  }
  const Expr* dim(int index) const {
    return function()->dim(index);
  }
  const std::vector<const Expr*>& dims() const {
    return function()->dims();
  }
  const Var* arg(int index) const {
    return function()->arg(index);
  }
  const std::vector<const Var*>& args() const {
    return function()->args();
  }

  Tensor(Function* function, int output_index)
      : function_(function), output_index_(output_index) {}
  template <typename... Ts>
  inline ExprHandle operator()(const Ts&... ts);
  template <typename T>
  inline ExprHandle call(const std::vector<T>& args);
  template <typename... Ts>
  inline ExprHandle call(const Ts&... ts);

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
  DimArg(const ExprHandle& dim) : dim_(dim) {}
  DimArg(const ExprHandle& dim, const std::string& name_hint)
      : dim_(dim), name_hint_(name_hint) {}
  const ExprHandle& dim() const {
    return dim_;
  }
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  ExprHandle dim_;
  std::string name_hint_;
};

TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&)>& body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);

class FunctionCall : public CallNode<FunctionCall> {
 public:
  using BaseClass = CallNode<FunctionCall>;
  static ExprHandle make(
      Tensor* tensor,
      const std::vector<ExprHandle>& params) {
    std::vector<const Expr*> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return ExprHandle(new FunctionCall(tensor, params_nodes));
  }

  const Tensor* tensor() const {
    return tensor_;
  }
  Tensor* tensor() {
    return tensor_;
  }

  FunctionCall(Tensor* tensor, const std::vector<const Expr*>& params)
      : BaseClass(
            tensor->function()->body(tensor->output_index())->dtype(),
            kFunctionCall,
            params),
        tensor_(tensor) {}

 private:
  const Expr* DefaultMutator(
      const std::vector<const Expr*>& new_params) const override {
    return new FunctionCall(tensor_, new_params);
  }

  std::string func_name() const {
    return tensor_->func_var()->name_hint();
  }

  Tensor* tensor_;
};
template <typename... Ts>
inline ExprHandle Tensor::operator()(const Ts&... ts) {
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return FunctionCall::make(this, std::move(params));
}

template <typename... Ts>
inline ExprHandle Tensor::call(const Ts&... ts) {
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return FunctionCall::make(this, std::move(params));
}

template <typename T>
inline ExprHandle Tensor::call(const std::vector<T>& args) {
  std::vector<ExprHandle> params(args.begin(), args.end());
  return FunctionCall::make(this, params);
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
