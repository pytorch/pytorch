#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

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
  void printNode(const kir::Node* node);

  //! Print a complete Kernel definition
  void printKernel(const Kernel* kernel);

 private:
  // Generates a string representation of an IR node
  //
  // If `top_level` is true, all the value uses are tracked and
  // their definitions are implicitly printed before the node itself
  //
  std::string gen(const kir::Node* node, bool top_level = false);

  // Generate a string representation of an used value
  // (this helps automatically tracking the value uses)
  std::string use(const kir::Val* val);

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

  // Current indentation level
  int indent_level_ = 0;

  // Internal IR generation stream
  std::stringstream ir_str_;

  // Tracks the set of nodes which have been printed
  std::unordered_set<const kir::Node*> visited_;

  // Optional left margin printed after the indentation
  const char* margin_ = "";

  // The set of values used by the current top-level IR node
  std::unordered_set<const kir::Val*> uses_;
};

//! Returns the string representation of a Kernel IR node
std::string toString(const kir::Node* stmt);

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
