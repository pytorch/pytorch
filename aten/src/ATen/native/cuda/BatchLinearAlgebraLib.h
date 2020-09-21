#pragma once

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
// some cusolver functions doesn't work well on cuda 9.2, cusolver is used on cuda >= 10.0
#define USE_CUSOLVER
#endif

#ifdef USE_CUSOLVER

namespace at {
namespace native {

Tensor _inverse_helper_cuda_lib(const Tensor& self);

}}  // namespace at::native

#endif  // USE_CUSOLVER
