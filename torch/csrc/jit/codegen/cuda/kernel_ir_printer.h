#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

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
class TORCH_CUDA_API IrPrinter : private kir::IrVisitor {
  static constexpr char* kTab = "  ";

 public:
  //! Constructs a new IrPrinter which outputs to the specified stream
  explicit IrPrinter(std::ostream& os) : os_(os) {}

  //! Print a single Kernel IR node
  void printNode(const kir::Node* stmt);

  //! Print a complete Kernel definition
  void printKernel(const Kernel* kernel);

 private:
  static std::string gen(const kir::Node* stmt);

  std::ostream& indent();

  void startBlock();
  void endBlock();
  void handleBlock(const kir::Scope& scope);

  void visit(const kir::Bool*) final;
  void visit(const kir::Float*) final;
  void visit(const kir::Half*) final;
  void visit(const kir::Int*) final;
  void visit(const kir::NamedScalar*) final;

  void visit(const kir::TensorIndex*) final;
  void visit(const kir::IterDomain*) final;
  void visit(const kir::TensorDomain*) final;
  void visit(const kir::TensorView*) final;

  void visit(const kir::UnaryOp*) final;
  void visit(const kir::BinaryOp*) final;
  void visit(const kir::TernaryOp*) final;
  void visit(const kir::ReductionOp*) final;
  void visit(const kir::BroadcastOp*) final;

  void visit(const kir::GridReduction*) final;
  void visit(const kir::ForLoop*) final;
  void visit(const kir::IfThenElse*) final;
  void visit(const kir::Allocate*) final;
  void visit(const kir::Sync*) final;

 private:
  std::ostream& os_;
  int indent_level_ = 0;
};

//! Returns the string representation of a Kernel IR node
std::string toString(const kir::Node* stmt);

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
