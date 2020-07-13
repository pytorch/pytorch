#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

struct Statement;

struct Val;
struct Expr;

struct UnaryOp;
struct BinaryOp;
struct TernaryOp;
struct ReductionOp;
struct BroadcastOp;

struct ForLoop;
struct IfThenElse;

struct TensorDomain;
struct TensorView;
struct IterDomain;
struct TensorIndex;

struct TensorContiguity;

struct Split;
struct Merge;

struct Bool;
struct Float;
struct Half;
struct Int;
struct Add;

/*
 * Define pretty printing functions for all nodes. handle is used so we can take
 * advantage of OptInConstDispatch. Where we will throw an error if a print
 * function is not defined for a node. Stream operator << is also provided for
 * Fusion&, Fusion* and Statement* which allow us to print any node through
 * stream operator <<.
 */

struct TORCH_CUDA_API IRPrinter : public OptInConstDispatch {
 public:
  std::ostream& os;
  bool print_inline_ = false;

  // Track the indentation size for pretty printing
  int indent_size = 0;

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

  virtual void handle(Fusion* const f);

  // handle calls some non const fusion ops,
  // eventhough fusion should remain unchanged.
  // Need to look into this.
  virtual void handle(const Fusion* const f) {
    handle(const_cast<Fusion*>(f));
  }
  virtual void handle(Fusion& f) {
    handle(&f);
  }

  virtual void handle(const Statement* const s) {
    OptInConstDispatch::handle(s);
  };

  virtual void handle(const Val* const v) {
    OptInConstDispatch::handle(v);
  };
  virtual void handle(const Expr* const e) {
    OptInConstDispatch::handle(e);
  };

  virtual void handle(const TensorDomain* const) override;
  virtual void handle(const TensorView* const) override;
  virtual void handle(const IterDomain* const) override;
  virtual void handle(const TensorIndex* const) override;

  virtual void handle(const Bool* const) override;
  virtual void handle(const Float* const) override;
  virtual void handle(const Half* const) override;
  virtual void handle(const Int* const) override;
  virtual void handle(const NamedScalar* const) override;

  virtual void handle(const UnaryOp* const) override;
  virtual void handle(const BinaryOp* const) override;
  virtual void handle(const TernaryOp* const) override;
  virtual void handle(const ReductionOp* const) override;
  virtual void handle(const BroadcastOp* const) override;

  virtual void handle(const ForLoop* const) override;
  virtual void handle(const IfThenElse* const) override;
  virtual void handle(const Allocate* const) override;

  virtual void handle(const Split* const) override;
  virtual void handle(const Merge* const) override;

  void print_inline(const Statement* const stmt) {
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
    const Statement* const stmt);
TORCH_CUDA_API std::ostream& operator<<(std::ostream& os, Fusion* f);
TORCH_CUDA_API std::ostream& operator<<(std::ostream& os, Fusion& f);

} // namespace fuser
} // namespace jit
} // namespace torch
