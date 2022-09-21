#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <c10/util/irange.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Fusion;
namespace kir {
class Kernel;
class Scope;
} // namespace kir

//! Define pretty printing functions for IR nodes
//!
//! This class is intended for debug printing, so it attempts
//! to handle invalid states as well.
//!
class TORCH_CUDA_CU_API IrPrinter : public OptInConstDispatch {
  static constexpr char const* kTab = "  ";

 public:
  explicit IrPrinter(std::ostream& os) : os_(os) {}

  // Indent the generated code
  std::ostream& indent() {
    for (const auto i : c10::irange(indent_size_)) {
      (void)i; // Suppress unused variable warning
      os_ << "  ";
    }
    return os_;
  }

  void resetIndent() {
    indent_size_ = 0;
  }

  bool printInline() const {
    return print_inline_;
  }

  using OptInConstDispatch::handle;

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

  virtual void handle(const kir::Kernel* kernel);
  virtual void handle(kir::Kernel& kernel);

  void handleScope(const kir::Scope& scope);

  void handle(const Statement* s) final;
  void handle(const Val* v) final;
  void handle(const Expr* e) final;

  void handle(const IterDomain*) final;
  void handle(const TensorDomain*) final;
  void handle(const TensorView*) final;

  void handle(const Bool*) final;
  void handle(const Double*) final;
  void handle(const Int*) final;
  void handle(const ComplexDouble*) final;
  void handle(const NamedScalar*) final;

  void handle(const ARangeOp*) final;
  void handle(const UnaryOp*) final;
  void handle(const BinaryOp*) final;
  void handle(const TernaryOp*) final;
  void handle(const RNGOp*) final;
  void handle(const ReductionOp*) final;
  void handle(const GroupedReductionOp*) final;
  void handle(const WelfordOp*) final;
  void handle(const GroupedWelfordOp*) final;
  void handle(const LoadStoreOp*) final;
  void handle(const MmaOp*) final;
  void handle(const BroadcastOp*) final;
  void handle(const TransposeOp*) final;
  void handle(const ExpandOp*) final;
  void handle(const ShiftOp*) final;
  void handle(const GatherOp*) final;
  void handle(const ViewAsScalar*) final;
  void handle(const ViewOp*) final;

  void handle(const kir::Predicate*) final;
  void handle(const kir::TensorIndex*) final;
  void handle(const kir::IntPair*) final;

  void handle(const kir::GridBroadcast*) final;
  void handle(const kir::GridReduction*) final;
  void handle(const kir::GroupedGridReduction*) final;
  void handle(const kir::GridWelford*) final;
  void handle(const kir::GroupedGridWelford*) final;
  void handle(const kir::ForLoop*) final;
  void handle(const kir::IfThenElse*) final;
  void handle(const kir::Allocate*) final;
  void handle(const kir::BlockSync*) final;
  void handle(const kir::GridSync*) final;
  void handle(const kir::CpAsyncWait*) final;
  void handle(const kir::CpAsyncCommit*) final;
  void handle(const kir::InitMagicZero*) final;
  void handle(const kir::UpdateMagicZero*) final;
  void handle(const kir::AllocateFusedReduction*) final;
  void handle(const kir::Swizzle2DInt*) final;
  void handle(const kir::PairSelect*) final;

  // IR math printer overrides these to prevent them from printing, keep
  // override
  void handle(const Split*) override;
  void handle(const Merge*) override;
  void handle(const Swizzle2D*) override;

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
