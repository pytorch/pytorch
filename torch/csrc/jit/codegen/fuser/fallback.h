#pragma once

#include <ATen/core/stack.h>

#include <cstdlib>

namespace torch::jit::fuser {

void runFallback(int64_t key, Stack& stack);

} // namespace torch::jit::fuser
