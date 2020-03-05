#pragma once

#include <torch/csrc/jit/fuser/common/iriostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_API IRMathPrinter : public IRPrinter {
 public:
  IRMathPrinter(std::ostream& os) : IRPrinter(os) {}

  void print(const Split* const) override {}
  void print(const Merge* const) override {}
  void print(const Reorder* const) override {}

  void print(Fusion* f) override {
    IRPrinter::print(f);
  }
};

class TORCH_API IRTransformPrinter : public IRPrinter {
 public:
  IRTransformPrinter(std::ostream& os) : IRPrinter(os) {}

  // Tensor Expressions
  void print(const UnaryOp* const uop) override {
    if(print_inline_)
      IRPrinter::print(uop);
  }

  void print(const BinaryOp* const bop) override {
    if(print_inline_)
      IRPrinter::print(bop);
  }
  
  void print(const ForLoop* const) override {}
  void print(const IfThenElse* const) override {}

  void print(Fusion* f) override {
    IRPrinter::print(f);
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
