#pragma once

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 10200
// some cusolver functions don't work well on cuda 9.2 or cuda 10.1.105, cusolver is used on cuda >= 10.1.243
#define USE_CUSOLVER
#endif

// comment out this line for the emergency break if there is something wrong with parallel launch
// e.g. if you have a script failure with linear algebra and set environment variable CUDA_LAUNCH_BLOCKING=1
// solves the problem, then you can try removing the cuda parallel stream launch
#define _USE_PARALLEL_LAUNCH


#ifdef _USE_PARALLEL_LAUNCH

#define CUDA_PARALLEL_STREAM_LAUNCH(i, _batch_size, _stmt)  \
do {                                                        \
  auto _main_stream = at::cuda::getCurrentCUDAStream();     \
  at::cuda::CUDAEvent _main_event;                          \
  _main_event.record(_main_stream);                         \
  for (int i = 0; i < _batch_size; i++) {                   \
    auto _stream = at::cuda::getStreamFromPool();           \
    at::cuda::CUDAStreamGuard _guard(_stream);               \
    _main_event.block(_stream);                             \
    _stmt();                                                \
    at::cuda::CUDAEvent _finished;                          \
    _finished.record(_stream);                              \
    _finished.block(_main_stream);                          \
  }                                                         \
} while(0)

#else // _USE_PARALLEL_LAUNCH is not defined

#define CUDA_PARALLEL_STREAM_LAUNCH(i, _batch_size, _stmt)  \
do {                                                        \
  for (int i = 0; i < _batch_size; i++) {                   \
    _stmt();                                                \
  }                                                         \
} while (0)

#endif  // ifdef _USE_PARALLEL_LAUNCH

#ifdef USE_CUSOLVER

namespace at {
namespace native {

Tensor _inverse_helper_cuda_lib(const Tensor& self);

}}  // namespace at::native

#endif  // USE_CUSOLVER
