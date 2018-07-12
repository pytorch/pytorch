#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/lexer.h"


// helpers for handling constants in the IR
// - create constant nodes from ints, floats, intlist, Tensors, and other types
// - implement primitive constant ops.
namespace torch { namespace jit {

Value* createConstant(
    Graph& g,
    IValue val,
    at::optional<script::SourceRange> loc = at::nullopt);


}}
