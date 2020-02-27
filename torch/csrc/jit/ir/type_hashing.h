
copy: fbcode/caffe2/torch/csrc/jit/ir/type_hashing.h
copyrev: bf180ebb6d390115657f42650104fda235b9ba6d

#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct HashType {
  size_t operator()(const TypePtr& type) const;
};

struct EqualType {
  bool operator()(const TypePtr& a, const TypePtr& b) const;
};

} // namespace jit
} // namespace torch
