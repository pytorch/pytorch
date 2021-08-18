#pragma once

#include <torch/csrc/jit/tensorexpr/ir_printer.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace tensorexpr {

// Generates C++ code from the IR.
class TORCH_API CppPrinter : public IRPrinter {
 public:
  explicit CppPrinter(std::ostream* os) : IRPrinter(*os) {}

  using IRPrinter::visit;
  void visit(AllocatePtr) override;
  void visit(FreePtr) override;

 private:
  std::unordered_set<VarPtr> allocated_on_heap_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
