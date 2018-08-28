#include "ATen/ATen.h"
#include "THC.h"  // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA
static magma_queue_t createMagmaQueue(const Tensor& tensor) {
  auto& context = tensor.type().get_context();
  magma_queue_t magma_queue;
  magma_queue_create_from_cuda(
      tensor.get_device(),
      at::cuda::getCurrentCUDAStream(),
      THCState_getCurrentBlasHandle(context.getTHCState()),
      THCState_getCurrentSparseHandle(context.getTHCState()),
      &magma_queue);
  return magma_queue;
}

static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  if (static_cast<int64_t>(result) != value) {
    AT_ERROR("magma: The value of %s (%lld) is too large to fit into a magma_int_t (%llu bytes)",
             varname, (long long)value, sizeof(magma_int_t));
  }
  return result;
}
#endif

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline Storage pin_memory(int64_t size, Tensor dummy) {
  int64_t adjusted_size = size * sizeof(T);
  auto* allocator = cuda::getPinnedMemoryAllocator();
  auto& backend = dummy.type().toBackend(Backend::CPU).toScalarType(kByte);
  return backend.storageWithAllocator(adjusted_size, allocator);
}
  
} // namespace native
} // namespace at

