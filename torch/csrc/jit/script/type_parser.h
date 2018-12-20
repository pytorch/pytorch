#pragma once
#include <ATen/core/jit_type.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
namespace torch {
namespace jit {
namespace script {
struct Expr;
TORCH_API c10::optional<std::string> parseBaseTypeName(const Expr& expr);
TORCH_API c10::TypePtr parseTypeFromExpr(const Expr& expr);
}
} // namespace jit
} // namespace torch
