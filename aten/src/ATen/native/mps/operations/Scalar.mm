//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/ops/_local_scalar_dense_native.h>

using namespace at::mps;

namespace at::native {

Scalar _local_scalar_dense_mps(const Tensor& self) {
  Scalar r;
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");

  const void* data_ptr = self.storage().data();
  auto [cpu_ptr, retain_count] = getIMPSAllocator()->getSharedBufferPtr(data_ptr);

  // On Apple Silicon (unified memory), all MPS tensors use MTLStorageModeShared --
  // the CPU can read the buffer directly after syncing without a blit copy.
  std::optional<at::Tensor> cpu_staging;
  const char* src;
  if (cpu_ptr) {
    getCurrentMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
    src = static_cast<const char*>(cpu_ptr) + self.storage_offset() * self.itemsize();
  } else {
    // Fallback for private storage (discrete GPU or future hardware)
    cpu_staging = at::empty_like(self, TensorOptions(kCPU));
    mps::mps_copy_(*cpu_staging, self, false);
    src = static_cast<const char*>(cpu_staging->data_ptr());
  }

  AT_DISPATCH_V2(self.scalar_type(),
                 "_local_scalar_dense_mps",
                 AT_WRAP([&] {
                   r = Scalar(*reinterpret_cast<const scalar_t*>(src));
                 }),
                 AT_EXPAND(AT_ALL_TYPES),
                 AT_EXPAND(AT_COMPLEX_TYPES),
                 at::ScalarType::ComplexHalf,
                 at::ScalarType::Half,
                 at::ScalarType::Bool,
                 at::ScalarType::BFloat16,
                 at::ScalarType::UInt16,
                 at::ScalarType::UInt32,
                 at::ScalarType::UInt64);

  return r;
}

} // namespace at::native
