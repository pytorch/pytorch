#include <ATen/miopen/Exceptions.h>
#include <ATen/miopen/Handle.h>
#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <c10/hip/HIPStream.h>

namespace at { namespace native {
namespace {

void createMIOpenHandle(miopenHandle_t *handle) {
  MIOPEN_CHECK(miopenCreate(handle));
}

void destroyMIOpenHandle(miopenHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and I decided to not destroy
// the handle as a workaround.
//   - @soumith
//
// Further note: this is now disabled globally, because we are seeing
// the same issue as mentioned above in CUDA 11 CI.
//   - @zasdfgbnm
//
// #ifdef NO_MIOPEN_DESTROY_HANDLE
// #else
//   miopenDestroy(handle);
// #endif
}

using MIOpenPoolType = at::cuda::DeviceThreadHandlePool<miopenHandle_t, createMIOpenHandle, destroyMIOpenHandle>;

} // namespace

miopenHandle_t getMiopenHandle() {
  int device;
  HIP_CHECK(hipGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<MIOpenPoolType>();
  thread_local std::unique_ptr<MIOpenPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  MIOPEN_CHECK(miopenSetStream(handle, at::hip::getCurrentHIPStream()));
  return handle;
}

}} // namespace at::native
