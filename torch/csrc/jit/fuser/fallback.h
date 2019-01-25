#pragma once

#include <torch/csrc/jit/stack.h>

#include <cstdlib>

namespace torch {
namespace jit {
namespace fuser {

void runFallback(int64_t key, Stack& stack);

} // namespace fuser
} // namespace jit
} // namespace torch
