#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>

namespace torch::jit::tensorexpr {

class CppVarNameRewriter;

// Generates C++ code from the IR.
//
// Vector operations are unrolled.
// For example:
// C[Ramp(0, 1, 3)] = A[Ramp(0, 2, 3)] + B[Ramp(0, 3, 3)];
// is unrolled into:
// C[0] = A[0] + B[0];
// C[1] = A[2] + B[3];
// C[2] = A[4] + B[6];
class TORCH_API CppPrinter : public IRPrinter {
 public:
  explicit CppPrinter(std::ostream* os);
  ~CppPrinter() override;

  void printPrologue();

  using IRPrinter::visit;

  // Binary expressions.
  void visit(const ModPtr&) override;
  void visit(const MaxPtr&) override;
  void visit(const MinPtr&) override;

  // Conditional expressions.
  void visit(const CompareSelectPtr&) override;
  void visit(const IfThenElsePtr&) override;

  // Tensor operations.
  void visit(const AllocatePtr&) override;
  void visit(const FreePtr&) override;
  void visit(const LoadPtr&) override;
  void visit(const StorePtr&) override;

  // Casts.
  void visit(const CastPtr&) override;
  void visit(const BitCastPtr&) override;

  // Calls.
  void visit(const IntrinsicsPtr&) override;
  void visit(const ExternalCallPtr&) override;

  // Vars.
  void visit(const LetPtr&) override;
  void visit(const VarPtr&) override;

  // Vector data types.
  void visit(const RampPtr&) override;
  void visit(const BroadcastPtr&) override;

 private:
  int lane_;
  std::unordered_map<VarPtr, ExprPtr> vector_vars_;
};

class TORCH_API CppCodeGen : public CodeGen {
 public:
  CppCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func");

  ~CppCodeGen() override;

  void call(const std::vector<CallArg>& args) override;
  void call_raw(const std::vector<void*>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

  std::string getCodeText(const std::string& attr = "") override {
    return oss_.str();
  }

 private:
  void init();

  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;
  std::unique_ptr<CppPrinter> printer_;
  std::unique_ptr<CppVarNameRewriter> var_name_rewriter_;
};

} // namespace torch::jit::tensorexpr
