#include "torch/csrc/jit/tensorexpr/function.h"

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

static void unpack_dim_args(
    const std::vector<DimArg>& dim_args,
    std::vector<Expr>* dims,
    std::vector<Var>* vars) {
  dims->clear();
  vars->clear();
  for (size_t i = 0; i < dim_args.size(); i++) {
    dims->push_back(dim_args[i].dim());
    vars->push_back(Var(dim_args[i].name_hint(), kInt32));
  }
}

} // namespace

Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const std::vector<Var>&)> body_func) {
  std::vector<Expr> dims;
  std::vector<Var> args;
  unpack_dim_args(dim_args, &dims, &args);
  Expr body = body_func(args);
  Function func =
      Function(func_name, std::move(dims), std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&)> body_func) {
  CHECK_EQ(dim_args.size(), 1ULL);
  std::vector<Expr> dims;
  std::vector<Var> args;
  unpack_dim_args(dim_args, &dims, &args);
  Expr body = body_func(args[0]);
  Function func =
      Function(func_name, std::move(dims), std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&)> body_func) {
  CHECK_EQ(dim_args.size(), 2ULL);
  std::vector<Expr> dims;
  std::vector<Var> args;
  unpack_dim_args(dim_args, &dims, &args);
  Expr body = body_func(args[0], args[1]);
  Function func =
      Function(func_name, std::move(dims), std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&, const Var&)> body_func) {
  CHECK_EQ(dim_args.size(), 3ULL);
  std::vector<Expr> dims;
  std::vector<Var> args;
  unpack_dim_args(dim_args, &dims, &args);
  Expr body = body_func(args[0], args[1], args[2]);
  Function func =
      Function(func_name, std::move(dims), std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&, const Var&, const Var&)>
        body_func) {
  CHECK_EQ(dim_args.size(), 4ULL);
  std::vector<Expr> dims;
  std::vector<Var> args;
  unpack_dim_args(dim_args, &dims, &args);
  Expr body = body_func(args[0], args[1], args[2], args[3]);
  Function func =
      Function(func_name, std::move(dims), std::move(args), std::move(body));
  return Tensor(func, 0);
}

Stmt FunctionNode::ElementStmt() {
  std::vector<Expr> strides(dims_.size());
  for (size_t i = 0; i < strides.size(); i++) {
    if (i == strides.size() - 1) {
      strides[i] = Expr(1);
      continue;
    }
    Expr stride = dims_[i + 1];
    for (size_t j = i + 2; j < dims_.size(); j++) {
      stride = stride * dims_[j];
    }
    strides[i] = stride;
  }

  Expr total_index;
  for (size_t i = 0; i < dims_.size(); i++) {
    Expr index = this->args_[i] * strides[i];
    if (i == 0) {
      total_index = index;
    } else {
      total_index = total_index + index;
    }
  }

  Expr mask = 1;

  Stmt update_stmt = Store::make(func_var(), total_index, body(), mask);
  return update_stmt;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
