#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <iostream>

/*
 * IRMathPrinter and IRTransformPrinter allow the splitting up of fusion print
 * functions. IRMathPrinter as its name implies focuses solely on what tensor
 * computations are taking place. Resulting TensorView math will reflect the
 * series of split/merge/computeAts that have taken place, however these
 * nodes will not be displayed in what is printed. IRTransformPrinter does not
 * print any mathematical functions and only lists the series of
 * split/merge calls that were made. Both of these printing methods are
 * quite verbose on purpose as to show accurately what is represented in the IR
 * of a fusion.
 */

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API IRMathPrinter : public IRPrinter {
 public:
  IRMathPrinter(std::ostream& os) : IRPrinter(os) {}

  void handle(const Split* const) override {}
  void handle(const Merge* const) override {}

  void handle(Fusion* f) override {
    IRPrinter::handle(f);
  }
};

class TORCH_CUDA_API IRTransformPrinter : public IRPrinter {
 public:
  IRTransformPrinter(std::ostream& os) : IRPrinter(os) {}

  // Tensor Expressions
  void handle(const UnaryOp* const uop) override {
    if (print_inline_)
      IRPrinter::handle(uop);
  }

  void handle(const BinaryOp* const bop) override {
    if (print_inline_)
      IRPrinter::handle(bop);
  }

  void handle(Fusion* f) override {
    IRPrinter::handle(f);
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
