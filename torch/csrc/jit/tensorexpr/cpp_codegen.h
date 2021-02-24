#pragma once

#include <torch/csrc/jit/tensorexpr/cpp_tensor.h>
#include <torch/csrc/jit/tensorexpr/cpp_vector.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Generates C++ code from the IR.
//
// The generated C++ code relies on:
// 1. Vector defined in cpp_vector.h;
// 2. Tensor defined in cpp_tensor.h.
class TORCH_API CppPrinter : public IRPrinter {
 public:
  explicit CppPrinter(std::ostream* os) : IRPrinter(*os) {}

  void printPrologue() {
    os() << "#include <cassert>" << std::endl;
    os() << "#include <cmath>" << std::endl;
    os() << "#include <vector>" << std::endl;
    os() << "#include <array>" << std::endl;
    os() << "#include <algorithm>" << std::endl;
    os() << std::endl;

    os() << "#define POS_INFINITY INFINITY" << std::endl;
    os() << "#define NEG_INFINITY -INFINITY" << std::endl;
    os() << std::endl;

    os() << cpp_vector_definition << std::endl;
    os() << std::endl;

    os() << cpp_tensor_definition << std::endl;
    os() << std::endl;
  }

  using IRPrinter::visit;

  // Vector data types.
  void visit(const Ramp*) override;
  void visit(const Broadcast*) override;

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

 private:
  std::string to_lambda(CompareSelectOperation op, const std::string& ty);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
