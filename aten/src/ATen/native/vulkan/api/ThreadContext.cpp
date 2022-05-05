#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/ThreadContext.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

ThreadContext::ThreadContext(const GPU& gpu)
  : gpu_(gpu) {
}

template<typename T>
ThreadContext::SingletonThreadLocalObject<T>::SingletonThreadLocalObject(const GPU& gpu) {
  TORCH_INTERNAL_ASSERT(false, "SingletonThreadLocalObject doesn't support the generalized template constructor!");
}

//
// Specialized template functions
//

template<>
ThreadContext::SingletonThreadLocalObject<Command>::SingletonThreadLocalObject(const GPU& gpu)
  : object_(gpu) {
}

template<>
ThreadContext::SingletonThreadLocalObject<Descriptor>::SingletonThreadLocalObject(const GPU& gpu)
  : object_(gpu) {
}

template<>
ThreadContext::SingletonThreadLocalObject<Resource>::SingletonThreadLocalObject(const GPU& gpu)
  : object_(gpu) {
}

template<>
ThreadContext::SingletonThreadLocalObject<QueryPool>::SingletonThreadLocalObject(const GPU& gpu)
  : object_(gpu.device,
      gpu.adapter->timestamp_compute_and_graphics(),
      gpu.adapter->timestamp_period()) {
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
