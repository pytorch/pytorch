#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

//! Define pretty printing functions for Kernel IR nodes
//!
//! This class is intended for debug printing, so it attempts
//! to handle invalid IR states as much as possible.
//!
class TORCH_CUDA_CU_API IrPrinter : private OptInConstDispatch {
  static constexpr char* kTab = "  ";

 public:
  //! Constructs a new IrPrinter which outputs to the specified stream
  explicit IrPrinter(std::ostream& os) : os_(os) {}

  //! Print a single Kernel IR node
  void printNode(const Statement* stmt);

  //! Print a complete Kernel definition
  void printKernel(const Kernel* kernel);

 private:
  static std::string gen(const Statement* stmt);

  std::ostream& indent();

  void startBlock();
  void endBlock();
  void handleBlock(const kir::Scope& scope);

  void handle(const Statement*) final;
  void handle(const Val*) final;
  void handle(const Expr*) final;

  void handle(const kir::Bool*) final;
  void handle(const kir::Float*) final;
  void handle(const kir::Half*) final;
  void handle(const kir::Int*) final;
  void handle(const kir::NamedScalar*) final;

  void handle(const kir::TensorIndex*) final;
  void handle(const kir::IterDomain*) final;
  void handle(const kir::TensorDomain*) final;
  void handle(const kir::TensorView*) final;

  void handle(const kir::UnaryOp*) final;
  void handle(const kir::BinaryOp*) final;
  void handle(const kir::TernaryOp*) final;
  void handle(const kir::ReductionOp*) final;
  void handle(const kir::BroadcastOp*) final;

  void handle(const kir::GridReduction*) final;
  void handle(const kir::ForLoop*) final;
  void handle(const kir::IfThenElse*) final;
  void handle(const kir::Allocate*) final;
  void handle(const kir::Sync*) final;

 private:
  std::ostream& os_;
  int indent_level_ = 0;
};

//! Returns the string representation of a Kernel IR node
std::string toString(const Statement* stmt);

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
