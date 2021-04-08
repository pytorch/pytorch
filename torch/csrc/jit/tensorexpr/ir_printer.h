#pragma once

#include <iostream>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Tensor;

class TORCH_API IRPrinter : public IRVisitor {
 public:
  explicit IRPrinter(std::ostream& os) : printer_os_(this, os) {}

  void print(ExprHandle);
  void print(const Expr&);
  void print(const Stmt&);
  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const Mod* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const And* v) override;
  void visit(const Or* v) override;
  void visit(const Xor* v) override;
  void visit(const Lshift* v) override;
  void visit(const Rshift* v) override;
  void visit(const CompareSelect* v) override;
#define IMM_PRINT_VISIT(Type, Name) void visit(const Name##Imm* v) override;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT
  void visit(const Cast* v) override;
  void visit(const Var* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const Broadcast* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const Term* v) override;
  void visit(const Polynomial* v) override;
  void visit(const RoundOff* v) override;
  void visit(const MaxTerm* v) override;
  void visit(const MinTerm* v) override;
  void visit(const ReduceOp* v) override;

  void visit(const AtomicAdd* v) override;
  void visit(const SyncThreads* v) override;
  void visit(const ExternalCall* v) override;
  void visit(const Store* v) override;
  void visit(const For* v) override;
  void visit(const Cond* v) override;
  void visit(const Block* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Let* v) override;

  std::ostream& os() {
    return printer_os_;
  }

  class PrinterStream : public std::ostream {
   public:
    PrinterStream(IRPrinter* printer, std::ostream& os)
        : std::ostream(os.rdbuf()), printer_(printer) {}

    IRPrinter* printer() {
      return printer_;
    }

   private:
    IRPrinter* printer_ = nullptr;
  };

 protected:
  UniqueNameManager* name_manager() {
    return &name_manager_;
  }
  void emitIndent();

  int indent_ = 0;

 private:
  PrinterStream printer_os_;
  UniqueNameManager name_manager_;
};

TORCH_API std::ostream& operator<<(std::ostream& stream, const Expr&);
TORCH_API std::ostream& operator<<(std::ostream& stream, const ExprHandle&);
TORCH_API std::ostream& operator<<(std::ostream& stream, const Stmt&);
TORCH_API std::ostream& operator<<(std::ostream& stream, const Tensor&);

TORCH_API void print(const Expr* expr);
TORCH_API void print(const Stmt* stmt);
TORCH_API void print(const Tensor* t);

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

using torch::jit::tensorexpr::Expr;
using torch::jit::tensorexpr::Stmt;
using torch::jit::tensorexpr::Tensor;

TORCH_API std::string to_string(const Expr* expr);
TORCH_API std::string to_string(const Stmt* stmt);
TORCH_API std::string to_string(const Tensor* t);
} // namespace std
