#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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
  void visit(const Mod*) override;
  void visit(const Max*) override;
  void visit(const Min*) override;

  // Conditional expressions.
  void visit(const CompareSelect*) override;
  void visit(const IfThenElse*) override;

  // Tensor operations.
  void visit(const Allocate*) override;
  void visit(const Free*) override;
  void visit(const Load*) override;
  void visit(const Store*) override;

  // Casts.
  void visit(const Cast*) override;
  void visit(const BitCast*) override;

  // Calls.
  void visit(const Intrinsics*) override;
  void visit(const ExternalCall*) override;

  // Vars.
  void visit(const Let*) override;
  void visit(const Var*) override;

  // Vector data types.
  void visit(const Ramp*) override;
  void visit(const Broadcast*) override;

 private:
  int lane_;
  std::unordered_map<const Var*, const Expr*> vector_vars_;
};

class TORCH_API CppCodeGen : public CodeGen {
 public:
  CppCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func");

  ~CppCodeGen() override;

  void call(const std::vector<CallArg>& args) override;

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

} // namespace tensorexpr
} // namespace jit
} // namespace torch
