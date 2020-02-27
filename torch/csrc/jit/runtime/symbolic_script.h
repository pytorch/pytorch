
copy: fbcode/caffe2/torch/csrc/jit/runtime/symbolic_script.h
copyrev: 3af42971a5556e44eee1b6238d6bf6e0beec8d74

#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <c10/util/Optional.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {
struct GradientPair {
  std::shared_ptr<Graph> forward;
  std::shared_ptr<Graph> backward;
};

TORCH_API c10::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema);
TORCH_API bool hasGradientInfoForSchema(const FunctionSchema& schema);
} // namespace jit
} // namespace torch
