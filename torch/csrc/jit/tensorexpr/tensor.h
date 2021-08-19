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

class TORCH_API Tensor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Tensor(BufPtr buf, const std::vector<VarPtr>& args, ExprPtr body)
      : buf_(buf) {
    stmt_ = constructStmt(args, body, {}, {});
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Tensor(
      BufPtr buf,
      const std::vector<VarPtr>& args,
      const std::vector<ExprPtr>& reduce_dims,
      const std::vector<VarPtr>& reduce_args,
      ExprPtr body)
      : buf_(buf) {
    stmt_ = constructStmt(args, body, reduce_dims, reduce_args);
  }

  Tensor(BufPtr buf, StmtPtr stmt) : buf_(buf), stmt_(stmt) {}

  BufPtr buf() const {
    return buf_;
  }

  StmtPtr stmt() const {
    return stmt_;
  }

  template <typename T>
  inline ExprHandle load(const std::vector<T>& args) const;
  template <typename... Ts>
  inline ExprHandle load(const Ts&... ts) const;

 private:
  StmtPtr constructStmt(
      const std::vector<VarPtr>& args,
      ExprPtr body,
      const std::vector<ExprPtr>& reduce_dims,
      const std::vector<VarPtr>& reduce_args) const;

  BufPtr buf_;
  StmtPtr stmt_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class Placeholder {
 public:
  Placeholder() = default;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Placeholder(const BufHandle& data) : data_(data.node()) {
    if (data_->base_handle()->dtype() != kHandle) {
      throw malformed_input("Placeholder dtype must be Handle");
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<ExprHandle> stride_handles(ndim());
    for (int i = (int)ndim() - 1; i >= 0; i--) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (i == ndim() - 1) {
        stride_handles[i] = 1;
      } else {
        stride_handles[i] = stride_handles[i + 1] * ExprHandle(dim(i + 1));
      }
    }
    strides_ = ExprHandleVectorToExprVector(stride_handles);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Placeholder(
      const std::string& name,
      const Dtype& dtype,
      const std::vector<ExprHandle>& dims)
      : Placeholder(BufHandle(name, dims, dtype)) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Placeholder(const std::vector<ExprHandle>& dims, const Dtype& dtype)
      : Placeholder(BufHandle("_", dims, dtype)) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit Placeholder(const std::vector<ExprHandle>& dims)
      : Placeholder(BufHandle("_", dims, kFloat)) {}

  BufPtr data() const {
    return data_;
  }
  BufHandle handle() const {
    return BufHandle(data());
  }
  Dtype dtype() const {
    return data_->dtype();
  }
  int ndim() const {
    return data_->ndim();
  }
  ExprPtr dim(int index) const {
    return data_->dim(index);
  }
  std::vector<ExprPtr> dims() const {
    return data_->dims();
  }

  template <typename... Ts>
  inline ExprHandle load(const Ts&... ts) const;

  template <typename T>
  inline ExprHandle load(const std::vector<T>& args) const;

  inline ExprHandle load(const std::vector<ExprHandle>& args) const;

  inline StorePtr store(
      const std::vector<ExprHandle>& args,
      const ExprHandle& val) const {
    return alloc<Store>(data(), ExprHandleVectorToExprVector(args), val.node());
  }

 private:
  BufPtr data_;
  std::vector<ExprPtr> strides_;
};

TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);

inline void unpack_dim_args(
    const std::vector<DimArg>& dim_args,
    std::vector<ExprPtr>* dims,
    std::vector<VarPtr>* vars) {
  dims->clear();
  vars->clear();
  for (const DimArg& dim_arg : dim_args) {
    ExprPtr expr = dim_arg.dim().node();
    dims->push_back(expr);
    vars->push_back(alloc<Var>(
        dim_arg.name_hint(),
        expr->dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
}

// Handle reductions over a Reducer and a body_func which produces values.
template <typename InitFunc, typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const InitFunc& init_func,
    const BodyFunc& body_func,
    const std::vector<DimArg>& reduce_args) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprPtr> dims;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<VarPtr> vars;
  unpack_dim_args(dim_args, &dims, &vars);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprPtr> reduce_dims;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<VarPtr> reduce_vars;
  unpack_dim_args(reduce_args, &reduce_dims, &reduce_vars);

  // If reduce_vars is empty, then it's not a reduction, but rather a simple
  // copy
  if (reduce_vars.empty()) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ExprPtr body =
        Reducer::getReduceBody(body_func, VarVectorToVarHandleVector(vars))
            .node();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    BufPtr func_result = alloc<Buf>(func_name, dims, body->dtype());
    return Tensor(func_result, vars, body);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<VarPtr> all_vars;
  all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  all_vars.insert(all_vars.end(), reduce_vars.begin(), reduce_vars.end());

  ExprHandle body =
      Reducer::getReduceBody(body_func, VarVectorToVarHandleVector(all_vars));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprPtr> output_args(vars.begin(), vars.end());
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ExprPtr init_expr = alloc<Cast>(
      body.dtype(), init_func(VarVectorToVarHandleVector(vars)).node());
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  BufPtr func_result = alloc<Buf>(func_name, dims, body.dtype(), init_expr);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ReduceOpPtr reduce_op = reducer(func_result, body, output_args, reduce_vars);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Tensor t =
      Tensor(func_result, vars, reduce_dims, reduce_vars, reduce_op);
  return t;
}

template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const BodyFunc& body_func,
    const std::vector<DimArg>& reduce_args) {
  return Reduce(
      func_name,
      dim_args,
      reducer,
      [&](ParameterList p) { return ExprHandle(reducer.initializer()); },
      body_func,
      reduce_args);
}

// Overload which allows inline lambda functions for the body_func.
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const BodyFunc&& body_func,
    const std::vector<DimArg>& reduce_args) {
  return Reduce(func_name, dim_args, reducer, body_func, reduce_args);
}

// Overload for the common case of all dimensions of a Placeholder.
TORCH_API Tensor Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const Placeholder& buffer,
    const std::vector<DimArg>& reduce_args);

TORCH_API Tensor Reduce(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const BufHandle& buffer,
    const std::vector<DimArg>& reduce_args);

// Overload for the common case of all dimensions of a prevously Computed
// Tensor.
TORCH_API Tensor Reduce(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    Tensor tensor,
    const std::vector<DimArg>& reduce_args);

template <typename... Ts>
inline ExprHandle Tensor::load(const Ts&... ts) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return Load::make(BufHandle(this->buf()), params);
}

template <typename T>
inline ExprHandle Tensor::load(const std::vector<T>& args) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params(args.begin(), args.end());
  return Load::make(BufHandle(this->buf()), params);
}

template <typename... Ts>
inline ExprHandle Placeholder::load(const Ts&... ts) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return ExprHandle(alloc<Load>(data(), ExprHandleVectorToExprVector(params)));
}

template <typename T>
inline ExprHandle Placeholder::load(const std::vector<T>& args) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params(args.begin(), args.end());
  return ExprHandle(alloc<Load>(data(), ExprHandleVectorToExprVector(params)));
}

inline ExprHandle Placeholder::load(const std::vector<ExprHandle>& args) const {
  return this->template load<ExprHandle>(args);
}

template <typename... Ts>
inline ExprHandle BufHandle::load(const Ts&... ts) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return Load::make(*this, params);
}

template <typename T>
inline ExprHandle BufHandle::load(const std::vector<T>& args) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params(args.begin(), args.end());
  return Load::make(*this, params);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
