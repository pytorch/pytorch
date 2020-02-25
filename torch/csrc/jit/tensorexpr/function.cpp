#include "torch/csrc/jit/tensorexpr/function.h"

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

static void unpack_dim_args(
    const std::vector<DimArg>& dim_args,
    std::vector<ExprHandle>* dims,
    std::vector<VarHandle>* vars) {
  dims->clear();
  vars->clear();
  for (size_t i = 0; i < dim_args.size(); i++) {
    dims->push_back(dim_args[i].dim());
    vars->push_back(VarHandle(dim_args[i].name_hint(), kInt32));
  }
}

} // namespace

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandle(const std::vector<VarHandle>&)> body_func) {
  std::vector<ExprHandle> dims;
  std::vector<VarHandle> args;
  unpack_dim_args(dim_args, &dims, &args);
  ExprHandle body = body_func(args);
  Function* func = new Function(
      func_name, std::move(dims), std::move(args), std::move(body));
  return new Tensor(func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandle(const VarHandle&)> body_func) {
  CHECK_EQ(dim_args.size(), 1ULL);
  std::vector<ExprHandle> dims;
  std::vector<VarHandle> args;
  unpack_dim_args(dim_args, &dims, &args);
  ExprHandle body = body_func(args[0]);
  Function* func =
      new Function(func_name, std::move(dims), std::move(args), std::move(body));
  return new Tensor(func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandle(const VarHandle&, const VarHandle&)> body_func) {
  CHECK_EQ(dim_args.size(), 2ULL);
  std::vector<ExprHandle> dims;
  std::vector<VarHandle> args;
  unpack_dim_args(dim_args, &dims, &args);
  ExprHandle body = body_func(args[0], args[1]);
  Function* func = new Function(
      func_name, std::move(dims), std::move(args), std::move(body));
  return new Tensor(func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)> body_func) {
  CHECK_EQ(dim_args.size(), 3ULL);
  std::vector<ExprHandle> dims;
  std::vector<VarHandle> args;
  unpack_dim_args(dim_args, &dims, &args);
  ExprHandle body = body_func(args[0], args[1], args[2]);
  Function* func = new Function(
      func_name, std::move(dims), std::move(args), std::move(body));
  return new Tensor(func, 0);
}

Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&, const VarHandle&)>
        body_func) {
  CHECK_EQ(dim_args.size(), 4ULL);
  std::vector<ExprHandle> dims;
  std::vector<VarHandle> args;
  unpack_dim_args(dim_args, &dims, &args);
  ExprHandle body = body_func(args[0], args[1], args[2], args[3]);
  Function* func = new Function(
      func_name, std::move(dims), std::move(args), std::move(body));
  return new Tensor(func, 0);
}

Stmt* Function::ElementStmt() {
  std::vector<ExprHandle> strides(dims_.size());
  for (size_t i = 0; i < strides.size(); i++) {
    if (i == strides.size() - 1) {
      strides[i] = ExprHandle(1);
      continue;
    }
    ExprHandle stride = dims_[i + 1];
    for (size_t j = i + 2; j < dims_.size(); j++) {
      stride = stride * dims_[j];
    }
    strides[i] = stride;
  }

  ExprHandle total_index;
  for (size_t i = 0; i < dims_.size(); i++) {
    ExprHandle index = this->args_[i] * strides[i];
    if (i == 0) {
      total_index = index;
    } else {
      total_index = total_index + index;
    }
  }

  ExprHandle mask = 1;

  Stmt* update_stmt = Store::make(func_var(), total_index, body(), mask);
  return update_stmt;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
