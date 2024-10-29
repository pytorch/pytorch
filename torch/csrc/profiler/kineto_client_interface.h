#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch {

//
TORCH_API void
global_kineto_init(void);

} // namespace torch
