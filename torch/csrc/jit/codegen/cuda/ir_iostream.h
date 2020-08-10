#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

class Fusion;

class Statement;

class Val;
class Expr;

class UnaryOp;
class BinaryOp;
class TernaryOp;
class ReductionOp;
class BroadcastOp;

class ForLoop;
class IfThenElse;

class TensorDomain;
class TensorView;
class IterDomain;
class TensorIndex;

class Split;
class Merge;

class Bool;
class Float;
class Half;
class Int;
class Add;

/*
 * Define pretty printing functions for all nodes. handle is used so we can take
 * advantage of OptInConstDispatch. Where we will throw an error if a print
 * function is not defined for a node. Stream operator << is also provided for
 * Fusion&, Fusion* and Statement* which allow us to print any node through
 * stream operator <<.
 */

class TORCH_CUDA_API IRPrinter : public OptInConstDispatch {
 public:
  std::ostream& os;
  bool print_inline_ = false;

  // Track the indentation size for pretty printing
  int indent_size = 0;

  // Handle value mapping
  bool follow_val_map = true;

  // Indent the generated code
  void indent() {
    for (int i = 0; i < indent_size; i++)
      os << "  ";
  }

  void resetIndent() {
    indent_size = 0;
  }

  void printHeader(Fusion* fusion, const std::string& kernel_name_);

  IRPrinter(std::ostream& _os) : os(_os) {}

  virtual void handle(Fusion* f);

  // handle calls some non const fusion ops,
  // eventhough fusion should remain unchanged.
  // Need to look into this.
  virtual void handle(const Fusion* f) {
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
  void handle(const TensorIndex*) override;

  void handle(const Bool*) override;
  void handle(const Float*) override;
  void handle(const Half*) override;
  void handle(const Int*) override;
  void handle(const NamedScalar*) override;

  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;
  void handle(const TernaryOp*) override;
  void handle(const ReductionOp*) override;
  void handle(const BroadcastOp*) override;

  void handle(const ForLoop*) override;
  void handle(const IfThenElse*) override;
  void handle(const Allocate*) override;

  void handle(const Split*) override;
  void handle(const Merge*) override;

  void print_inline(const Statement* stmt) {
    bool prev = print_inline_;
    print_inline_ = true;
    handle(stmt);
    print_inline_ = prev;
  }

  void printReductionOps(Fusion* fusion);

  void printKernel(
      const std::vector<Expr*>& exprs,
      const std::string& kernel_name);
};

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& os,
    const Statement* stmt);
TORCH_CUDA_API std::ostream& operator<<(std::ostream& os, Fusion* f);
TORCH_CUDA_API std::ostream& operator<<(std::ostream& os, Fusion& f);

} // namespace fuser
} // namespace jit
} // namespace torch
