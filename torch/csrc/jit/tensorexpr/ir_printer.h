#pragma once

#include <ostream>

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch::jit::tensorexpr {

class Tensor;

class TORCH_API IRPrinter : public IRVisitor {
 public:
  explicit IRPrinter(std::ostream& os) : printer_os_(this, os) {}

  void print(ExprHandle);
  void print(Expr&);
  void print(Stmt&);
  void visit(const AddPtr& v) override;
  void visit(const SubPtr& v) override;
  void visit(const MulPtr& v) override;
  void visit(const DivPtr& v) override;
  void visit(const ModPtr& v) override;
  void visit(const MaxPtr& v) override;
  void visit(const MinPtr& v) override;
  void visit(const AndPtr& v) override;
  void visit(const OrPtr& v) override;
  void visit(const XorPtr& v) override;
  void visit(const LshiftPtr& v) override;
  void visit(const RshiftPtr& v) override;
  void visit(const CompareSelectPtr& v) override;
#define IMM_PRINT_VISIT(Type, Name) void visit(const Name##ImmPtr& v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT);
#undef IMM_PRINT_VISIT
  void visit(const CastPtr& v) override;
  void visit(const BitCastPtr& v) override;
  void visit(const VarPtr& v) override;
  void visit(const BufPtr& v) override;
  void visit(const RampPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const BroadcastPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const IntrinsicsPtr& v) override;
  void visit(const TermPtr& v) override;
  void visit(const PolynomialPtr& v) override;
  void visit(const RoundOffPtr& v) override;
  void visit(const MaxTermPtr& v) override;
  void visit(const MinTermPtr& v) override;
  void visit(const ReduceOpPtr& v) override;

  void visit(const AtomicAddPtr& v) override;
  void visit(const SyncThreadsPtr& v) override;
  void visit(const ExternalCallPtr& v) override;
  void visit(const ExternalCallWithAllocPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const CondPtr& v) override;
  void visit(const BlockPtr& v) override;
  void visit(const AllocatePtr& v) override;
  void visit(const FreePtr& v) override;
  void visit(const FreeExtPtr& v) override;
  void visit(const PlacementAllocatePtr& v) override;
  void visit(const LetPtr& v) override;

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

TORCH_API void print(const ExprPtr& expr);
TORCH_API void print(const StmtPtr& stmt);
TORCH_API void print(const Tensor& t);

} // namespace torch::jit::tensorexpr

namespace std {

using torch::jit::tensorexpr::Expr;
using torch::jit::tensorexpr::ExprPtr;
using torch::jit::tensorexpr::Stmt;
using torch::jit::tensorexpr::StmtPtr;
using torch::jit::tensorexpr::Tensor;

TORCH_API std::string to_string(const ExprPtr& expr);
TORCH_API std::string to_string(const StmtPtr& stmt);
TORCH_API std::string to_string(const Tensor& t);
} // namespace std
