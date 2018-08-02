#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/source_range.h"

namespace torch { namespace jit {

struct NamedValue {
  NamedValue(const SourceRange& loc, const std::string& name, Value* value)
  : loc_(loc), name_(name), value_(value) {}
  NamedValue(const SourceRange& loc, int i, Value* value)
  : loc_(loc), name_("argument " + std::to_string(i)), value_(value) {}

  const SourceRange& loc() const {
    return loc_;
  }

  Value* value() const {
    return value_;
  }

  const std::string& name() const {
    return name_;
  }

private:
  SourceRange loc_;
  std::string name_;
  Value* value_;
};

}}
