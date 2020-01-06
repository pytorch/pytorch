#include "torch/csrc/jit/compiler/include/function.h"

#include "torch/csrc/jit/compiler/include/logging.h"
#include "torch/csrc/jit/compiler/include/tensor.h"

namespace torch {
namespace jit {
namespace compiler {

namespace {

static std::vector<Var> arg_name_hints_to_args(
    int ndim,
    std::vector<std::string>& arg_name_hints) {
  std::vector<Var> args;
  CHECK_LE(arg_name_hints.size(), ndim);
  for (int i = 0; i < ndim; i++) {
    if (i < arg_name_hints.size()) {
      args.push_back(Var(arg_name_hints[i], kInt32));
    } else {
      args.push_back(Var(kInt32));
    }
  }
  return args;
}

} // namespace

Tensor Compute(
    const std::string& func_name,
    const std::vector<Expr>& dims,
    std::vector<std::string> arg_name_hints,
    std::function<Expr(const std::vector<Var>&)> body_func) {
  std::vector<Var> args = arg_name_hints_to_args(dims.size(), arg_name_hints);
  Expr body = body_func(args);
  Function func = Function(func_name, dims, std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<Expr>& dims,
    std::vector<std::string> arg_name_hints,
    std::function<Expr(const Var&)> body_func) {
  CHECK_EQ(dims.size(), 1);
  std::vector<Var> args = arg_name_hints_to_args(dims.size(), arg_name_hints);
  Expr body = body_func(args[0]);
  Function func = Function(func_name, dims, std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<Expr>& dims,
    std::vector<std::string> arg_name_hints,
    std::function<Expr(const Var&, const Var&)> body_func) {
  CHECK_EQ(dims.size(), 2);
  std::vector<Var> args = arg_name_hints_to_args(dims.size(), arg_name_hints);
  Expr body = body_func(args[0], args[1]);
  Function func = Function(func_name, dims, std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<Expr>& dims,
    std::vector<std::string> arg_name_hints,
    std::function<Expr(const Var&, const Var&, const Var&)> body_func) {
  CHECK_EQ(dims.size(), 3);
  std::vector<Var> args = arg_name_hints_to_args(dims.size(), arg_name_hints);
  Expr body = body_func(args[0], args[1], args[2]);
  Function func = Function(func_name, dims, std::move(args), std::move(body));
  return Tensor(func, 0);
}

Tensor Compute(
    const std::string& func_name,
    const std::vector<Expr>& dims,
    std::vector<std::string> arg_name_hints,
    std::function<Expr(const Var&, const Var&, const Var&, const Var&)>
        body_func) {
  CHECK_EQ(dims.size(), 4);
  std::vector<Var> args = arg_name_hints_to_args(dims.size(), arg_name_hints);
  Expr body = body_func(args[0], args[1], args[2], args[3]);
  Function func = Function(func_name, dims, std::move(args), std::move(body));
  return Tensor(func, 0);
}

Stmt FunctionNode::ElementStmt() {
  std::vector<Expr> strides(dims_.size());
  for (int i = 0; i < strides.size(); i++) {
    if (i == strides.size() - 1) {
      strides[i] = Expr(1);
      continue;
    }
    Expr stride = dims_[i + 1];
    for (int j = i + 2; j < dims_.size(); j++) {
      stride = stride * dims_[j];
    }
    strides[i] = stride;
  }

  Expr total_index;
  for (int i = 0; i < dims_.size(); i++) {
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

} // namespace compiler
} // namespace jit
} // namespace torch
