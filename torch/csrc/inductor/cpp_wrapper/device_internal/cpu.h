#pragma once

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>
// This header is required by cpp.py, but include it here so that it gets caught
// by header precompilation.
#include <torch/csrc/inductor/cpp_prefix.h>
