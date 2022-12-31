#pragma once

#include <torch/csrc/Export.h>
#include <functional>
#include <utility>
#include <vector>

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API Tensor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Tensor(BufPtr buf, const std::vector<VarPtr>& args, ExprPtr body)
      : buf_(std::move(buf)) {
    stmt_ = constructStmt(args, std::move(body), {}, {});
  }
  Tensor(BufHandle buf, const std::vector<VarHandle>& args, ExprHandle body)
      : Tensor(buf.node(), VarHandleVectorToVarVector(args), body.node()) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Tensor(
      BufPtr buf,
      const std::vector<VarPtr>& args,
      const std::vector<ExprPtr>& reduce_dims,
      const std::vector<VarPtr>& reduce_args,
      ExprPtr body)
      : buf_(std::move(buf)) {
    stmt_ = constructStmt(args, std::move(body), reduce_dims, reduce_args);
  }
  Tensor(
      BufHandle buf,
      const std::vector<VarHandle>& args,
      const std::vector<ExprHandle>& reduce_dims,
      const std::vector<VarHandle>& reduce_args,
      ExprHandle body)
      : Tensor(
            buf.node(),
            VarHandleVectorToVarVector(args),
            ExprHandleVectorToExprVector(reduce_dims),
            VarHandleVectorToVarVector(reduce_args),
            body.node()) {}

  Tensor(BufPtr buf, StmtPtr stmt)
      : buf_(std::move(buf)), stmt_(std::move(stmt)) {}

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

TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);
TORCH_API Tensor Compute(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func);

inline std::vector<VarHandle> create_index_vars(
    const std::vector<ExprHandle>& dims) {
  std::vector<VarHandle> vars;
  vars.reserve(dims.size());
  for (const ExprHandle& dim : dims) {
    vars.emplace_back(alloc<Var>(
        "i", dim.dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  return vars;
}

// Handle reductions over a Reducer and a body_func which produces values.
template <typename InitFunc, typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    const InitFunc& init_func,
    const BodyFunc& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  std::vector<VarHandle> vars = create_index_vars(dims);
  std::vector<VarHandle> reduce_vars = create_index_vars(reduce_dims);

  // If reduce_vars is empty, then it's not a reduction, but rather a simple
  // copy
  if (reduce_vars.empty()) {
    ExprHandle body = Reducer::getReduceBody(body_func, vars);
    BufHandle func_result = Buf::make(
        func_name, dims, body.dtype(), c10::nullopt, std::move(strides));
    return Tensor(std::move(func_result), vars, std::move(body));
  }

  std::vector<VarHandle> all_vars;
  all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  all_vars.insert(all_vars.end(), reduce_vars.begin(), reduce_vars.end());

  ExprHandle body = Reducer::getReduceBody(body_func, all_vars);
  std::vector<ExprHandle> output_args(vars.begin(), vars.end());
  ExprHandle init_expr = Cast::make(body.dtype(), init_func(vars));
  BufHandle func_result = Buf::make(func_name, dims, body.dtype(), init_expr);

  ExprHandle reduce_op = reducer(func_result, body, output_args, reduce_vars);
  if (body.dtype() == kBFloat16) {
    ExprHandle init_expr_acc = Cast::make(kFloat, init_func(vars));
    BufHandle func_result_acc =
        Buf::make(func_name + "_acc", dims, kFloat, init_expr_acc);
    reduce_op = reducer(
        func_result,
        std::move(func_result_acc),
        std::move(body),
        output_args,
        reduce_vars);
  }

  Tensor t = Tensor(
      std::move(func_result),
      vars,
      reduce_dims,
      reduce_vars,
      std::move(reduce_op));
  return t;
}
template <typename InitFunc, typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    const InitFunc& init_func,
    const BodyFunc& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce<InitFunc, BodyFunc>(
      func_name,
      dims,
      c10::nullopt,
      reducer,
      init_func,
      body_func,
      reduce_dims);
}

template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    const BodyFunc& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(
      func_name,
      dims,
      strides,
      reducer,
      [&](ParameterList p) { return ExprHandle(reducer.initializer()); },
      body_func,
      reduce_dims);
}
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    const BodyFunc& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce<BodyFunc>(
      func_name, dims, c10::nullopt, reducer, body_func, reduce_dims);
}

// Overload which allows inline lambda functions for the body_func.
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    const BodyFunc&& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(func_name, dims, strides, reducer, body_func, reduce_dims);
}
template <typename BodyFunc>
Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    const BodyFunc&& body_func,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(func_name, dims, c10::nullopt, reducer, body_func, reduce_dims);
}

TORCH_API Tensor Reduce(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    const BufHandle& buffer,
    const std::vector<ExprHandle>& reduce_dims);
TORCH_API Tensor Reduce(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    const BufHandle& buffer,
    const std::vector<ExprHandle>& reduce_dims);

// Overload for the common case of all dimensions of a prevously Computed
// Tensor.
TORCH_API Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    Tensor tensor,
    const std::vector<ExprHandle>& reduce_dims);
TORCH_API Tensor Reduce(
    const std::string& func_name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    Tensor tensor,
    const std::vector<ExprHandle>& reduce_dims);

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
inline ExprHandle BufHandle::load(const Ts&... ts) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params({ExprHandle(ts)...});
  return ExprHandle(alloc<Load>(node(), ExprHandleVectorToExprVector(params)));
}

template <typename T>
inline ExprHandle BufHandle::load(const std::vector<T>& args) const {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<ExprHandle> params(args.begin(), args.end());
  return ExprHandle(alloc<Load>(node(), ExprHandleVectorToExprVector(params)));
}

inline ExprHandle BufHandle::load(const std::vector<ExprHandle>& args) const {
  return this->template load<ExprHandle>(args);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
