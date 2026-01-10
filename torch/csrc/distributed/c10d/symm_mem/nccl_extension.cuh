#pragma once

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>

#ifdef USE_C10D_NCCL
#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_SYMMEM_SUPPORT
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#endif
#endif

namespace c10d::nccl_extension {

TORCH_API bool is_nccl_symmem_available();

TORCH_API void nccl_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_get(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal);

TORCH_API void nccl_put_with_signal(at::Tensor& tensor, int64_t signal, int64_t peer);

} // namespace c10d::nccl_extension
