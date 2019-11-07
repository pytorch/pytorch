#include <ATen/ATen.h>
#include <THC/THC.h>  // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA

// RAII for a MAGMA Queue
struct MAGMAQueue {

  // Default constructor without a device will cause
  // destroying a queue which has not been initialized.
  MAGMAQueue() = delete;

  // Constructor
  explicit MAGMAQueue(int64_t device_id) {
    auto& context = at::globalContext();
    magma_queue_create_from_cuda(
      device_id,
      at::cuda::getCurrentCUDAStream(),
      at::cuda::getCurrentCUDABlasHandle(),
      at::cuda::getCurrentCUDASparseHandle(),
      &magma_queue_);
  }

  // Getter
  magma_queue_t get_queue() const { return magma_queue_; }

  // Destructor
  ~MAGMAQueue() {
    magma_queue_destroy(magma_queue_);
  }

 private:
  magma_queue_t magma_queue_;
};

static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  if (static_cast<int64_t>(result) != value) {
    AT_ERROR("magma: The value of ", varname, "(", (long long)value,
             ") is too large to fit into a magma_int_t (", sizeof(magma_int_t), " bytes)");
  }
  return result;
}
#endif

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline Storage pin_memory(int64_t size) {
  auto* allocator = cuda::getPinnedMemoryAllocator();
  int64_t adjusted_size = size * sizeof(T);
  return Storage(
      caffe2::TypeMeta::Make<uint8_t>(),
      adjusted_size,
      allocator,
      /*resizable=*/false
  );
}

} // namespace native
} // namespace at
