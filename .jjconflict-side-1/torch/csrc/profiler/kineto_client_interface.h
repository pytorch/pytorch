#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch {

// declare global_kineto_init for libtorch_cpu.so to call
TORCH_API void global_kineto_init();

} // namespace torch
