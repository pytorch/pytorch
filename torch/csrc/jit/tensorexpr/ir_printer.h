#pragma once

#include <iostream>

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
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
  void visit(AddPtr v) override;
  void visit(SubPtr v) override;
  void visit(MulPtr v) override;
  void visit(DivPtr v) override;
  void visit(ModPtr v) override;
  void visit(MaxPtr v) override;
  void visit(MinPtr v) override;
  void visit(AndPtr v) override;
  void visit(OrPtr v) override;
  void visit(XorPtr v) override;
  void visit(LshiftPtr v) override;
  void visit(RshiftPtr v) override;
  void visit(CompareSelectPtr v) override;
#define IMM_PRINT_VISIT(Type, Name) void visit(Name##ImmPtr v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT
  void visit(CastPtr v) override;
  void visit(BitCastPtr v) override;
  void visit(VarPtr v) override;
  void visit(RampPtr v) override;
  void visit(LoadPtr v) override;
  void visit(BroadcastPtr v) override;
  void visit(IfThenElsePtr v) override;
  void visit(IntrinsicsPtr v) override;
  void visit(TermPtr v) override;
  void visit(PolynomialPtr v) override;
  void visit(RoundOffPtr v) override;
  void visit(MaxTermPtr v) override;
  void visit(MinTermPtr v) override;
  void visit(ReduceOpPtr v) override;

  void visit(AtomicAddPtr v) override;
  void visit(SyncThreadsPtr v) override;
  void visit(ExternalCallPtr v) override;
  void visit(StorePtr v) override;
  void visit(ForPtr v) override;
  void visit(CondPtr v) override;
  void visit(BlockPtr v) override;
  void visit(AllocatePtr v) override;
  void visit(FreePtr v) override;
  void visit(LetPtr v) override;

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
  std::string to_string(CompareSelectOperation op);

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

TORCH_API void print(ExprPtr expr);
TORCH_API void print(StmtPtr stmt);
TORCH_API void print(const Tensor& t);

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

using torch::jit::tensorexpr::Expr;
using torch::jit::tensorexpr::ExprPtr;
using torch::jit::tensorexpr::Stmt;
using torch::jit::tensorexpr::StmtPtr;
using torch::jit::tensorexpr::Tensor;

TORCH_API std::string to_string(ExprPtr expr);
TORCH_API std::string to_string(StmtPtr stmt);
TORCH_API std::string to_string(const Tensor& t);
} // namespace std
