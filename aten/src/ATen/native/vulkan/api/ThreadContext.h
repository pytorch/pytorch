#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/Resource.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Vulkan Thread Context holds onto all per-thread Vulkan states such as
// Command, Descriptor and Resource objects.
//

class ThreadContext final {
 public:
  ThreadContext() = delete;
  explicit ThreadContext(const GPU& gpu);
  ThreadContext(const ThreadContext&) = delete;
  ThreadContext(ThreadContext&&) = default;
  ThreadContext& operator=(const ThreadContext&) = delete;
  ThreadContext& operator=(ThreadContext&&) = default;

  Command& command();
  Descriptor& descriptor();
  Resource& resource();
  QueryPool& querypool();

 private:
  GPU gpu_;

 private:
  template<typename T>
  class SingletonThreadLocalObject final {
   public:
    explicit SingletonThreadLocalObject(const GPU& gpu);
    SingletonThreadLocalObject(const SingletonThreadLocalObject&) = delete;
    SingletonThreadLocalObject& operator=(const SingletonThreadLocalObject&) = delete;
    SingletonThreadLocalObject(SingletonThreadLocalObject&&) = default;
    SingletonThreadLocalObject& operator=(SingletonThreadLocalObject&&) = default;
    inline static T& get(const GPU& gpu) {
      static thread_local SingletonThreadLocalObject<T> object(gpu);
      return object.object_;
    }
   private:
    T object_;
  };
};

//
// Impl
//

inline Command& ThreadContext::command() {
  return SingletonThreadLocalObject<Command>::get(gpu_);
}

inline Descriptor& ThreadContext::descriptor() {
  return SingletonThreadLocalObject<Descriptor>::get(gpu_);
}

inline Resource& ThreadContext::resource() {
  return SingletonThreadLocalObject<Resource>::get(gpu_);
}

inline QueryPool& ThreadContext::querypool() {
  return SingletonThreadLocalObject<QueryPool>::get(gpu_);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
