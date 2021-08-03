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
  void print(Expr&);
  void print(Stmt&);
  void visit(Add* v) override;
  void visit(Sub* v) override;
  void visit(Mul* v) override;
  void visit(Div* v) override;
  void visit(Mod* v) override;
  void visit(Max* v) override;
  void visit(Min* v) override;
  void visit(And* v) override;
  void visit(Or* v) override;
  void visit(Xor* v) override;
  void visit(Lshift* v) override;
  void visit(Rshift* v) override;
  void visit(CompareSelect* v) override;
#define IMM_PRINT_VISIT(Type, Name) void visit(const Name##Imm* v) override;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT
  void visit(Cast* v) override;
  void visit(Var* v) override;
  void visit(Ramp* v) override;
  void visit(Load* v) override;
  void visit(Broadcast* v) override;
  void visit(IfThenElse* v) override;
  void visit(Intrinsics* v) override;
  void visit(Term* v) override;
  void visit(Polynomial* v) override;
  void visit(RoundOff* v) override;
  void visit(MaxTerm* v) override;
  void visit(MinTerm* v) override;
  void visit(ReduceOp* v) override;

  void visit(AtomicAdd* v) override;
  void visit(SyncThreads* v) override;
  void visit(ExternalCall* v) override;
  void visit(Store* v) override;
  void visit(For* v) override;
  void visit(Cond* v) override;
  void visit(Block* v) override;
  void visit(Allocate* v) override;
  void visit(Free* v) override;
  void visit(Let* v) override;

  // A child class may have a difference rule for generating dtype
  // string, e.g. CUDA needs int64_t to be generated as long long.
  virtual std::string dtypeToCppString(const Dtype& dtype);

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

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
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
