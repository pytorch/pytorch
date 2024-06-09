#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <c10/util/Optional.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/module.h>

namespace torch::jit {
struct GradientPair {
  std::shared_ptr<Graph> forward;
  std::shared_ptr<Graph> backward;
};

TORCH_API std::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema);
TORCH_API bool hasGradientInfoForSchema(const FunctionSchema& schema);
} // namespace torch::jit
