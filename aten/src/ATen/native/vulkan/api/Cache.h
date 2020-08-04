#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

template<typename Factory>
class Cache final {
 public:
  explicit Cache(Factory factory);
  ~Cache() = default;

  // Factory must have the following symbols defined.

  typedef typename Factory::Descriptor Descriptor;
  typedef typename Factory::Handle Handle;
  typedef typename Factory::Hasher Hasher;

  // Create or retrieve a resource.
  //
  // This operation is a simple cache lookup and returns the Handle corresponding
  // to the descriptor if the object is already present in the cache.  Otherwise,
  // Factory is used to create the object, after which point the object is added
  // to the cache.  Regardless, this function returns with the object in the cache.

  Handle retrieve(const Descriptor& descriptor);

 private:
  struct Configuration final {
    static constexpr uint32_t kReserve = 64u;
  };

  Factory factory_;
  ska::flat_hash_map<Descriptor, Handle, Hasher> cache_;
};

template<typename Factory>
inline Cache<Factory>::Cache(Factory factory)
  : factory_(std::move(factory)) {
    cache_.reserve(Configuration::kReserve);
}

template<typename Factory>
inline typename Cache<Factory>::Handle Cache<Factory>::retrieve(
    const Descriptor& descriptor) {
  auto iterator = cache_.find(descriptor);
  if (cache_.cend() == iterator) {
    iterator = cache_.insert({descriptor, factory_(*descriptor)}).first;
  }

  return iterator->second.get();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
