#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Statement;
struct Tensor;
struct Float;
struct Int;
struct Add;
struct Val;
struct Expr;

/*
 * Visitors are mechanisms to traverse the graph without modifying anything in it. This provides
 * safety to make sure nothing is actually modified in your pass. These passes could be analysis
 * done to make sure things are correct, or to collect information that will be used later.
 * This could be a dependency analysis, to create an ordering of all Exprs that respects
 * dataflow, or to simply print the IR.
 * 
 * TODO: Create a BaseHandler that instantiates all node types and simply traverses the graph.
 * This could then be derived by other passes where only some nodes need to be specialized.
 */ 

//TODO: Make BaseHandler that other vistiors can inherit from.
struct TORCH_API SimpleHandler {
  int handle(const Statement* const statement);
  int handle(const Float* const f);
  int handle(const Int* const i);
};

struct TORCH_API IRPrinter {
  virtual ~IRPrinter() = default;
  IRPrinter() = delete;
  IRPrinter(
    std::ostream& _out)
  : out_{_out} { }

  IRPrinter(const IRPrinter& other) = default;
  IRPrinter& operator=(const IRPrinter& other) = default;

  IRPrinter(IRPrinter&& other) = default;
  IRPrinter& operator=(IRPrinter&& other) = default;

  int handle(const Statement* const statement);
  int handle(const Float* const f);
  int handle(const Tensor* const t);
  int handle(const Int* const i);
  int handle(const Add* const add);

private:
  std::ostream& out_;

  static std::ostream& printValPreamble(std::ostream&, const Val* const);
};

}}} // torch::jit::fuser
