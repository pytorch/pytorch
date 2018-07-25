#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/lexer.h"
#include "torch/csrc/WindowsTorchApiMacro.h"

// helpers for handling constants in the IR
// - create constant nodes from ints, floats, intlist, Tensors, and other types
// - implement primitive constant ops.
namespace torch { namespace jit {

TORCH_API Value* insertConstant(
    Graph& g,
    IValue val,
    at::optional<script::SourceRange> loc = at::nullopt);


//////////////////////////////////////////////////////////////////////////////////
// Helper for retrieving constants
//////////////////////////////////////////////////////////////////////////////////

// attempt to convert a (possibly constant) Value* into an intepreter value (IValue).
// returns at::nullopt if the Value* was not constant
TORCH_API at::optional<IValue> toIValue(Value* v);

// if a value is a constant then try to turn into type T using the
// same rules as the interpreter
template<typename T>
at::optional<T> constant_as(Value* v) {
  if(auto ivalue = toIValue(v)) {
    return ivalue->to<T>();
  }
  return at::nullopt;
}

}}
