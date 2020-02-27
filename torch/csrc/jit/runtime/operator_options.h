
copy: fbcode/caffe2/torch/csrc/jit/runtime/operator_options.h
copyrev: 67bf8fa84ca93b56e7dbacc59d525bca99dc02d0

#pragma once

#include <ATen/core/dispatch/OperatorOptions.h>

namespace torch {
namespace jit {

using AliasAnalysisKind = c10::AliasAnalysisKind;

} // namespace jit
} // namespace torch
