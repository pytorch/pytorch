#pragma once

#include <torch/csrc/jit/fuser/common/iriostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_API IRMathPrinter : public IRPrinter {
 public:
  IRMathPrinter(std::ostream& os) : IRPrinter(os) {}

  void handle(const Split* const) override {}
  void handle(const Merge* const) override {}
  void handle(const Reorder* const) override {}

  void handle(Fusion* f) override {
    IRPrinter::handle(f);
  }
};

class TORCH_API IRTransformPrinter : public IRPrinter {
 public:
  IRTransformPrinter(std::ostream& os) : IRPrinter(os) {}

  // Tensor Expressions
  void handle(const UnaryOp* const uop) override {
    if(print_inline_)
      IRPrinter::handle(uop);
  }

  void handle(const BinaryOp* const bop) override {
    if(print_inline_)
      IRPrinter::handle(bop);
  }
  
  void handle(const ForLoop* const) override {}
  void handle(const IfThenElse* const) override {}

  void handle(Fusion* f) override {
    IRPrinter::handle(f);
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
