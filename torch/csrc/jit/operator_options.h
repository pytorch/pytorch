#pragma once

#include <torch/csrc/jit/passes/alias_analysis.h>
#include <ATen/core/op_registration/operator_metadata.h>

namespace torch {
namespace jit {

using AliasAnalysisKind = c10::AliasAnalysisKind;

} // namespace jit
} // namespace torch
