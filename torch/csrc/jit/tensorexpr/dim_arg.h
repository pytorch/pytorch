#pragma once
#include <torch/csrc/jit/tensorexpr/expr.h>

namespace torch {
namespace jit {
namespace tensorexpr {
// A helper structure to store the arguments to specify dimensions. In the
// Compute arguments for dim_args, all of the following is supported. For
// example:
//    dim_args: {1, 2, 3, 4}
//    dim_args: {{1, "x"}, {2, "y"}, {3, "z"}}
//    dim_args: {1, 2, {3, "x"}}
class DimArg {
 public:
  // Intentionally leave out explicit to allow implicit conversions.
  DimArg(const ExprHandle& dim) : dim_(dim) {}
  DimArg(const ExprHandle& dim, std::string name_hint)
      : dim_(dim), name_hint_(std::move(name_hint)) {}
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

} // namespace tensorexpr
} // namespace jit
} // namespace torch
