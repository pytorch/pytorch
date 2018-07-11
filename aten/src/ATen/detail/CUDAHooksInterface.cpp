#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Error.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

void default_set_device(int32_t) {
  AT_ERROR(
      "DynamicCUDAInterface::set_device called "
      "before CUDA library was loaded");
}

void default_get_device(int32_t*) {
  AT_ERROR(
      "DynamicCUDAInterface::get_device called "
      "before CUDA library was loaded");
}

void default_unchecked_set_device(int32_t) {
  AT_ERROR(
      "DynamicCUDAInterface::unchecked_set_device called "
      "before CUDA library was loaded");
}

void default_cuda_stream_create_with_priority(cudaStream_t*, int32_t, int32_t) {
  AT_ERROR(
    "DynamicCUDAInterface::cuda_stream_create_with_priority called "
    "before CUDA library was loaded");
}

void default_cuda_stream_destroy(cudaStream_t) {
  AT_ERROR(
    "DynamicCUDAInterface::cuda_stream_destroy called "
    "before CUDA library was loaded");
}

// Default the static members of DynamicCUDAInterface.
void (*DynamicCUDAInterface::set_device)(int32_t) = default_set_device;
void (*DynamicCUDAInterface::get_device)(int32_t*) = default_get_device;
void (*DynamicCUDAInterface::unchecked_set_device)(int32_t) =
    default_unchecked_set_device;
void (*DynamicCUDAInterface::cuda_stream_create_with_priority)(cudaStream_t*, int32_t, int32_t) 
  = default_cuda_stream_create_with_priority;
void (*DynamicCUDAInterface::cuda_stream_destroy)(cudaStream_t) 
  = default_cuda_stream_destroy;
  

const CUDAHooksInterface& getCUDAHooks() {
  static std::unique_ptr<CUDAHooksInterface> cuda_hooks;
  // NB: The once_flag here implies that if you try to call any CUDA
  // functionality before libATen_cuda.so is loaded, CUDA is permanently
  // disabled for that copy of ATen.  In principle, we can relax this
  // restriction, but you might have to fix some code.  See getVariableHooks()
  // for an example where we relax this restriction (but if you try to avoid
  // needing a lock, be careful; it doesn't look like Registry.h is thread
  // safe...)
  static std::once_flag once;
  std::call_once(once, [] {
    cuda_hooks = CUDAHooksRegistry()->Create("CUDAHooks", CUDAHooksArgs{});
    if (!cuda_hooks) {
      cuda_hooks =
          std::unique_ptr<CUDAHooksInterface>(new CUDAHooksInterface());
    }
  });
  return *cuda_hooks;
}
} // namespace detail

AT_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs)

} // namespace at
