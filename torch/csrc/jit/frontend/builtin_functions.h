
copy: fbcode/caffe2/torch/csrc/jit/frontend/builtin_functions.h
copyrev: 58ccdd0e021fd29018c74ba31750752c5afb2310

#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name);
}
} // namespace jit
} // namespace torch
