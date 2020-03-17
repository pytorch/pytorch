#pragma once

#include "torch/csrc/jit/codegen/fuser/cuda/resource_strings.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// Walk the Statment looking for Half size loads/stores.
class CudaHalfChecker : public IRVisitor {
 public:
  bool hasHalf() {
    return hasHalf_;
  }

  void visit(const Load* v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
  }
  void visit(const Store* v) override {
    hasHalf_ |= v->value()->dtype().scalar_type() == ScalarType::Half;
  }

 private:
  bool hasHalf_{false};
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
