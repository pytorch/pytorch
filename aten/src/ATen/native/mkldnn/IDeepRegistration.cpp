#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <c10/core/Allocator.h>
// needs to be included only once in library.
#include <ideep_pin_singletons.hpp>
#include <ATen/native/mkldnn/IDeepRegistration.h>
#include <ATen/native/mkldnn/detail/PrimitiveCache.h>

using namespace ideep;

static RegisterEngineAllocator cpu_alloc(
  engine::cpu_engine(),
  [](size_t size) {
    return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);
  },
  [](void* p) {
    c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);
  }
);

namespace at::native::mkldnn{
void clear_computation_cache() {
  // Reset computation_cache for forward convolutions
  // As it also caches max number of OpenMP workers
#ifndef __aarch64__
  utils::computation_cache<dnnl::convolution_forward::primitive_desc>::t_store()
      .clear();
#endif
}

} // namespace  at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
