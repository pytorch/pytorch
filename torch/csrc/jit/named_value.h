#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/source_range.h"

namespace torch { namespace jit {

struct NamedValue {
  NamedValue(const SourceRange& loc, const std::string& name, Value* value)
  : loc(loc), name(name), value(value) {}
  NamedValue(const SourceRange& loc, int i, Value* value)
  : loc(loc), name("argument " + std::to_string(i)), value(value) {}

  SourceRange loc;
  std::string name;
  Value* value;
};

}}
