#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <functional>
#include <vector>

#include <torch/csrc/jit/tensorexpr/dim_arg.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Function : public KernelScopedObject {
 public:
  Function(
      const std::string& func_name,
      const std::vector<const Expr*>& dims,
      const std::vector<const Var*>& args,
      const Expr* body)
      // TODO: Function should not create buffers, they should be created
      // manually before constructing a function.
      : func_vars_({new Buf(func_name, dims, body->dtype())}),
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
      func_vars_[i] = new Buf(func_names[i], dims, bodies[i]->dtype());
    }
  }
  Function(
      const std::string& func_name,
      Buf* func_var,
      const std::vector<const Expr*>& dims,
      const std::vector<const Var*>& args,
      const Expr* body)
      : func_vars_({func_var}), dims_(dims), args_(args), bodies_({body}) {}

  size_t ndim() const {
    return dims_.size();
  }

  const Expr* dim(size_t index) const {
    if (index < 0 || index >= dims_.size()) {
      throw out_of_range_index();
    }

    return dims_[index];
  }
  const std::vector<const Expr*>& dims() const {
    return dims_;
  }

  const Var* arg(size_t index) const {
    if (index < 0 || index >= args_.size()) {
      throw out_of_range_index();
    }

    return args_[index];
  }
  const std::vector<const Var*>& args() const {
    return args_;
  }

  std::vector<const Expr*> bodies() const {
    return bodies_;
  }
  const Expr* body(size_t index) const {
    if (index >= bodies_.size()) {
      throw out_of_range_index();
    }

    return bodies_[index];
  }

  std::vector<const Buf*> func_vars() const {
    return func_vars_;
  }
  const Buf* func_var(size_t index) const {
    if (index >= func_vars_.size()) {
      throw out_of_range_index();
    }
    return func_vars_[index];
  }

  Stmt* ElementStmt(size_t index);

 private:
  std::vector<const Buf*> func_vars_;
  std::vector<const Expr*> dims_;
  std::vector<const Var*> args_;
  std::vector<const Expr*> bodies_;
};

class Tensor : KernelScopedObject {
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
  const Buf* func_var() const {
    return function()->func_var(output_index());
  }
  int ndim() const {
    return buf_->dims().size();
  }
  const Expr* dim(int index) const {
    return buf_->dim(index);
  }
  std::vector<const Expr*> dims() const {
    return buf_->dims();
  }
  const Var* arg(int index) const {
    return function()->arg(index);
  }
  const std::vector<const Var*>& args() const {
    return function()->args();
  }

  const Buf* buf() const {
    return buf_;
  }

  void initializeTo(const Expr* initializer) {
    initializer_ = initializer;
  }
  const Expr* initializer() const {
    return initializer_;
  }

  Tensor(const Buf* buf, Function* function, int output_index)
      : buf_(buf), function_(function), output_index_(output_index) {}
  template <typename... Ts>
  inline ExprHandle operator()(const Ts&... ts);
  template <typename T>
  inline ExprHandle call(const std::vector<T>& args);
  template <typename... Ts>
  inline ExprHandle call(const Ts&... ts);

 private:
  const Buf* buf_;
  Function* function_;
  int output_index_;
  const Expr* initializer_{nullptr};
};

class Buffer {
 public:
  Buffer(const BufHandle& data) : data_(data.node()) {
    if (data_->base_handle()->dtype() != kHandle) {
      throw malformed_input("Buffer dtype must be Handle");
    }

    std::vector<ExprHandle> stride_handles(ndim());
    for (int i = (int)ndim() - 1; i >= 0; i--) {
      if (i == ndim() - 1) {
        stride_handles[i] = 1;
      } else {
        stride_handles[i] = stride_handles[i + 1] * ExprHandle(dim(i + 1));
      }
    }
    strides_ = ExprHandleVectorToExprVector(stride_handles);
  }
  Buffer(
      const std::string& name,
      const Dtype& dtype,
      const std::vector<ExprHandle>& dims)
      : Buffer(BufHandle(name, dims, dtype)) {}

  const Buf* data() const {
    return data_;
  }
  Dtype dtype() const {
    return data_->dtype();
  }
  int ndim() const {
    return data_->ndim();
  }
  const Expr* dim(int index) const {
    return data_->dim(index);
  }
  std::vector<const Expr*> dims() const {
    return data_->dims();
  }

  // TODO: consider defer the storage flatten to a later stage.
  template <typename... Args>
  ExprHandle operator()(Args... args) const {
    return LoadValue(std::forward<Args>(args)...);
  }

  ExprHandle LoadValue(
      const ExprHandle& x,
      const ExprHandle& y,
      const ExprHandle& z) const {
    return Load::make(*this, {x, y, z}, ExprHandle(1));
  }
  ExprHandle LoadValue(const ExprHandle& x, const ExprHandle& y) const {
    return Load::make(*this, {x, y}, ExprHandle(1));
  }
  ExprHandle LoadValue(const ExprHandle& x) const {
    return Load::make(*this, {x}, ExprHandle(1));
  }

  template <typename T>
  ExprHandle call(const std::vector<T>& args) const {
    std::vector<ExprHandle> params(args.begin(), args.end());
    return LoadValue(params);
  }

 private:
  ExprHandle LoadValue(const std::vector<ExprHandle>& indices) const;

  const Buf* data_;
  std::vector<const Expr*> strides_;
};

inline ExprHandle Buffer::LoadValue(
    const std::vector<ExprHandle>& indices) const {
  return Load::make(*this, indices, ExprHandle(1));
}

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

inline void unpack_dim_args(
    const std::vector<DimArg>& dim_args,
    std::vector<const Expr*>* dims,
    std::vector<const Var*>* vars) {
  dims->clear();
  vars->clear();
  for (const DimArg& dim_arg : dim_args) {
    dims->push_back(dim_arg.dim().node());
    vars->push_back(new Var(dim_arg.name_hint(), kInt));
  }
}

// Handle reductions over a Reducer and a body_func which produces values.
template <typename BodyFunc>
Tensor* Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const BodyFunc& body_func,
    const std::vector<DimArg>& reduce_args) {
  std::vector<const Expr*> dims;
  std::vector<const Var*> vars;
  unpack_dim_args(dim_args, &dims, &vars);

  std::vector<const Expr*> reduce_dims;
  std::vector<const Var*> reduce_vars;
  unpack_dim_args(reduce_args, &reduce_dims, &reduce_vars);

  std::vector<const Var*> all_vars;
  all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  all_vars.insert(all_vars.end(), reduce_vars.begin(), reduce_vars.end());

  ExprHandle body =
      Reducer::getReduceBody(body_func, VarVectorToVarHandleVector(all_vars));
  std::vector<const Expr*> output_args(vars.begin(), vars.end());
  Buf* func_result = new Buf(func_name, dims, body.dtype());
  const ReduceOp* reduce_op =
      reducer(func_result, body, output_args, reduce_vars);
  dims.insert(dims.end(), reduce_dims.begin(), reduce_dims.end());
  Function* func =
      new Function(func_name, func_result, dims, all_vars, reduce_op);
  Tensor* t = new Tensor(func_result, func, 0);
  t->initializeTo(new Cast(body.dtype(), reducer.initializer()));
  return t;
}

// Overload which allows inline lambda functions for the body_func.
template <typename BodyFunc>
Tensor* Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const BodyFunc&& body_func,
    const std::vector<DimArg>& reduce_args) {
  return Reduce(func_name, dim_args, reducer, body_func, reduce_args);
}

// Overload for the common case of all dimensions of a Buffer.
TORCH_API Tensor* Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const Buffer& buffer,
    const std::vector<DimArg>& reduce_args);

// Overload for the common case of all dimensions of a prevously Computed
// Tensor.
TORCH_API Tensor* Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    Tensor* tensor,
    const std::vector<DimArg>& reduce_args);

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

  std::string func_name() const override {
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
