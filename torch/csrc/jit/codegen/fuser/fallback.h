
copy: fbcode/caffe2/torch/csrc/jit/codegen/fuser/fallback.h
copyrev: cc3ca6203d4c47c7822b4b6df6baad4818854231

#pragma once

#include <ATen/core/stack.h>

#include <cstdlib>

namespace torch {
namespace jit {
namespace fuser {

void runFallback(int64_t key, Stack& stack);

} // namespace fuser
} // namespace jit
} // namespace torch
