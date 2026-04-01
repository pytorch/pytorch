#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <ATen/hipdnn/Handle.h>
#include <c10/hip/HIPStream.h>

#include <ATen/hip/Exceptions.h>
#include <ATen/hipdnn/Exceptions.h>
#include <hipdnn/frontend/hipdnn_frontend/Handle.hpp>

namespace at::native {
namespace {

void createHipdnnHandle(hipdnnHandle_t* handle) {
  HIPDNN_CHECK(hipdnnCreate(handle));
}

void destroyHipdnnHandle(hipdnnHandle_t handle) {
  // Intentionally not destroying handle to avoid shutdown ordering issues.
  // See comments in the miopen equivalent (Handle.cpp).
}

using HipDNNPoolType = at::cuda::DeviceThreadHandlePool<
    hipdnnHandle_t,
    createHipdnnHandle,
    destroyHipdnnHandle>;

} // namespace

hipdnnHandle_t getHipdnnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::hip::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<HipDNNPoolType>();
  thread_local std::unique_ptr<HipDNNPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  HIPDNN_CHECK(hipdnnSetStream(handle, c10::hip::getCurrentHIPStream()));
  return handle;
}

} // namespace at::native
