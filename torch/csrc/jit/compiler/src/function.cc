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

} // namespace compiler
} // namespace jit
} // namespace torch
