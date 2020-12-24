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

class TORCH_API Tensor : KernelScopedObject {
 public:
  Tensor(
      const std::string& name,
      const std::vector<const Expr*>& dims,
      const std::vector<const Var*>& args,
      const Expr* body)
      // TODO: Function should not create buffers, they should be created
      // manually before constructing a function.
      : buf_(new Buf(name, dims, body->dtype())), args_(args), body_(body) {}

  Tensor(Buf* buf, const std::vector<const Var*>& args, const Expr* body)
      : buf_(buf), args_(args), body_(body) {}

  Tensor(
      const Buf* buf,
      const std::vector<const Var*>& args,
      const std::vector<const Expr*>& reduce_dims,
      const std::vector<const Var*>& reduce_args,
      const Expr* body)
      : buf_(buf),
        args_(args),
        body_(body),
        reduce_dims_(reduce_dims),
        reduce_args_(reduce_args) {}

  virtual ~Tensor() {}

  // Wrappers over accessors to fields of the underlying function
  const Expr* body() const {
    return body_;
  }
  const Buf* buf() const {
    return buf_;
  }
  size_t ndim() const {
    return buf()->ndim();
  }
  const Expr* dim(size_t index) const {
    if (index >= ndim()) {
      throw out_of_range_index();
    }
    return buf()->dim(index);
  }
  std::vector<const Expr*> dims() const {
    return buf()->dims();
  }
  const Var* arg(size_t index) const {
    if (index >= ndim()) {
      throw out_of_range_index();
    }
    return args_[index];
  }
  const std::vector<const Var*>& args() const {
    return args_;
  }
  size_t reduce_ndim() const {
    return reduce_dims_.size();
  }
  std::vector<const Expr*> reduce_dims() const {
    return reduce_dims_;
  }
  std::vector<const Var*> reduce_args() const {
    return reduce_args_;
  }
  const Expr* reduce_dim(size_t index) const {
    if (index >= reduce_ndim()) {
      throw out_of_range_index();
    }
    return reduce_dims_[index];
  }
  const Var* reduce_arg(size_t index) const {
    if (index >= reduce_ndim()) {
      throw out_of_range_index();
    }
    return reduce_args_[index];
  }

  void initializeTo(const Expr* initializer) {
    initializer_ = initializer;
  }
  const Expr* initializer() const {
    return initializer_;
  }
  virtual Stmt* ElementStmt() const;

  template <typename... Ts>
  inline ExprHandle operator()(const Ts&... ts);
  template <typename T>
  inline ExprHandle call(const std::vector<T>& args);
  template <typename... Ts>
  inline ExprHandle call(const Ts&... ts);

 private:
  const Buf* buf_;
  std::vector<const Var*> args_;
  const Expr* body_;
  std::vector<const Expr*> reduce_dims_;
  std::vector<const Var*> reduce_args_;

  const Expr* initializer_{nullptr};
};

class TORCH_API CompoundTensor : public Tensor {
 public:
  CompoundTensor(
      const Buf* buf,
      const std::vector<const Var*>& args,
      Stmt* stmt)
      : Tensor(buf, args, {}, {}, nullptr), stmt_(stmt) {}

  virtual ~CompoundTensor() {}

  Stmt* ElementStmt() const override {
    return stmt_;
  }

 private:
  Stmt* stmt_;
};

class Placeholder {
 public:
  Placeholder(const BufHandle& data) : data_(data.node()) {
    if (data_->base_handle()->dtype() != kHandle) {
      throw malformed_input("Placeholder dtype must be Handle");
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
  Placeholder(
      const std::string& name,
      const Dtype& dtype,
      const std::vector<ExprHandle>& dims)
      : Placeholder(BufHandle(name, dims, dtype)) {}

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

  template <typename... Ts>
  inline ExprHandle load(const Ts&... ts) const;

  template <typename T>
  inline ExprHandle load(const std::vector<T>& args) const;

  inline ExprHandle loadWithMask(
      const std::vector<ExprHandle>& args,
      const ExprHandle& mask) const {
    return ExprHandle(
        new Load(data(), ExprHandleVectorToExprVector(args), mask.node()));
  }

  inline Store* store(
      const std::vector<ExprHandle>& args,
      const ExprHandle& val) const {
    return new Store(
        data(), ExprHandleVectorToExprVector(args), val.node(), new IntImm(1));
  }

  inline Store* storeWithMask(
      const std::vector<ExprHandle>& args,
      const ExprHandle& val,
      const ExprHandle& mask) const {
    return new Store(
        data(), ExprHandleVectorToExprVector(args), val.node(), mask.node());
  }

 private:
  const Buf* data_;
  std::vector<const Expr*> strides_;
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
  Tensor* t =
      new Tensor(func_result, vars, reduce_dims, reduce_vars, reduce_op);
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

// Overload for the common case of all dimensions of a Placeholder.
TORCH_API Tensor* Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const Placeholder& buffer,
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
      : BaseClass(tensor->buf()->dtype(), kFunctionCall, params),
        tensor_(tensor) {}

 private:
  const Expr* DefaultMutator(
      const std::vector<const Expr*>& new_params) const override {
    return new FunctionCall(tensor_, new_params);
  }

  std::string func_name() const override {
    return tensor_->buf()->name_hint();
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

template <typename... Ts>
inline ExprHandle Placeholder::load(const Ts&... ts) const {
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return ExprHandle(
      new Load(data(), ExprHandleVectorToExprVector(params), new IntImm(1)));
}

template <typename T>
inline ExprHandle Placeholder::load(const std::vector<T>& args) const {
  std::vector<ExprHandle> params(args.begin(), args.end());
  return ExprHandle(
      new Load(data(), ExprHandleVectorToExprVector(params), new IntImm(1)));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
