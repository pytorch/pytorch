#pragma once

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
// some cusolver functions doesn't work well on cuda 9.2, cusolver is used on cuda >= 10.0
#define USE_CUSOLVER
#endif

#define ALLOCATE_ARRAY(name, type, size)            \
  auto storage_##name = pin_memory<type>(size);     \
  name = static_cast<type*>(storage_##name.data());

#ifdef USE_CUSOLVER

namespace at {
namespace native {

template<class scalar_t>
static void _apply_single_inverse_helper(scalar_t* self_ptr, scalar_t* self_inv_ptr, int* ipiv_ptr, int* info_ptr, int n);

Tensor _inverse_helper_cuda_lib(const Tensor& self);

}}  // namespace at::native

#endif  // USE_CUSOLVER