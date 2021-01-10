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

#ifdef USE_CUSOLVER

namespace at {
namespace native {

// entrance of calculations of `inverse` using cusolver getrf + getrs, cublas getrfBatched + getriBatched
Tensor _inverse_helper_cuda_lib(const Tensor& self);
Tensor& _linalg_inv_out_helper_cuda_lib(Tensor& result, Tensor& infos_getrf, Tensor& infos_getrs);

// entrance of calculations of `svd` using cusolver gesvdj and gesvdjBatched
std::tuple<Tensor, Tensor, Tensor> _svd_helper_cuda_lib(const Tensor& self, bool some, bool compute_uv);

}}  // namespace at::native

#endif  // USE_CUSOLVER
