#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <c10/util/irange.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Define pretty printing functions for IR nodes
//!
//! This class is intended for debug printing, so it attempts
//! to handle invalid states as well.
//!
class TORCH_CUDA_CU_API IrPrinter : public OptInConstDispatch {
 public:
  explicit IrPrinter(std::ostream& os) : os_(os) {}

  // Indent the generated code
  void indent() {
    for (const auto i : c10::irange(indent_size_)) {
      (void)i; // Suppress unused variable warning
      os_ << "  ";
    }
  }

  void resetIndent() {
    indent_size_ = 0;
  }

  bool printInline() const {
    return print_inline_;
  }

  virtual void handle(Fusion* f);

  // handle calls some non const fusion ops,
  // eventhough fusion should remain unchanged.
  // Need to look into this.
  virtual void handle(const Fusion* f) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    handle(const_cast<Fusion*>(f));
  }

  virtual void handle(Fusion& f) {
    handle(&f);
  }

  void handle(const Statement* s) override;
  void handle(const Val* v) override;
  void handle(const Expr* e) override;

  void handle(const TensorDomain*) override;
  void handle(const TensorView*) override;
  void handle(const IterDomain*) override;

  void handle(const Bool*) override;
  void handle(const Double*) override;
  void handle(const Int*) override;
  void handle(const NamedScalar*) override;

  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;
  void handle(const TernaryOp*) override;
  void handle(const ReductionOp*) override;
  void handle(const WelfordOp*) override;
  void handle(const BroadcastOp*) override;
  void handle(const TransposeOp*) override;
  void handle(const ShiftOp*) override;
  void handle(const GatherOp*) override;

  void handle(const Split*) override;
  void handle(const Merge*) override;

  void print_inline(const Statement* stmt) {
    bool prev = print_inline_;
    print_inline_ = true;
    handle(stmt);
    print_inline_ = prev;
  }

 protected:
  std::ostream& os() {
    return os_;
  }

 private:
  std::ostream& os_;
  bool print_inline_ = false;
  int indent_size_ = 0;
};

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    const Statement* stmt);

TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream& os, Fusion* f);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream& os, Fusion& f);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
