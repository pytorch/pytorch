#pragma once
#include <torch/csrc/inductor/cpp_wrapper/common.h>
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
#include <torch/csrc/inductor/cpp_wrapper/lazy_triton_compile.h>

#ifdef TORCH_INDUCTOR_PRECOMPILE_HEADERS
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
