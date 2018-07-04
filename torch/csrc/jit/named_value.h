#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/tree.h"

namespace torch { namespace jit {

struct NamedValue {
  NamedValue(const script::SourceRange& loc, const std::string& name, Value* value)
  : loc(loc), name(name), value(value) {}
  NamedValue(const script::SourceRange& loc, int i, Value* value)
  : loc(loc), name("argument " + std::to_string(i)), value(value) {}

  script::SourceRange loc;
  std::string name;
  Value* value;
};

}}
