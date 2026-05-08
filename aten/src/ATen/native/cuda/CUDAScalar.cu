#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense_native.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at::native {

namespace {

bool is_cuda_caching_allocator_tensor(const Tensor& self) {
  auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  if (cuda_allocator == nullptr) {
    return false;
  }
  // SymmMem/NVSHMEM/rocSHMEM tensors are typically backed by custom
  // from_blob-style deleters, so this check filters them out and keeps
  // the direct dereference path limited to allocator-managed CUDA memory.
  return self.storage().data_ptr().get_deleter() == cuda_allocator->raw_deleter();
}

template <typename scalar_t>
void _local_scalar_dense_cuda_impl(const Tensor& self, Scalar& r) {
#if defined(USE_ROCM) && (ROCM_VERSION >= 70200)
  // If this is a large BAR device, we can just read directly from VRAM
  if (
      at::cuda::getCurrentDeviceProperties()->isLargeBar &&
      is_cuda_caching_allocator_tensor(self)) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hipStreamCaptureStatus captureStatus;
    C10_CUDA_CHECK(hipStreamGetCaptureInfo(stream, &captureStatus, nullptr));
    if (C10_LIKELY(captureStatus == hipStreamCaptureStatusNone)) {
      at::cuda::stream_synchronize(stream);
      r = Scalar(*self.template const_data_ptr<scalar_t>());
    } else {
      C10_CUDA_CHECK(hipErrorStreamCaptureUnsupported);
    }
    return;
  }
#endif

  // Create pinned memory for the scalar value to avoid implicit
  // locking/sync in cuda library due to pageable memory
  auto value = at::detail::empty_cpu(
    {1}, /* size */
    c10::CppTypeToScalarType<scalar_t>(), /* dtype */
    std::nullopt, /* layout */
    std::nullopt, /* device */
    true, /* pin_memory */
    std::nullopt /* memory format */
  );
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::cuda::memcpy_and_sync(value.template mutable_data_ptr<scalar_t>(), self.template const_data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);
  r = Scalar(*value.template const_data_ptr<scalar_t>());
}

} // anonymous namespace

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");
    AT_DISPATCH_V2(
      self.scalar_type(), "_local_scalar_dense_cuda", AT_WRAP([&] {
        _local_scalar_dense_cuda_impl<scalar_t>(self, r);
      }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return r;
}

} // at::native
