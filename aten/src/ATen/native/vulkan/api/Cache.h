#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A generic cache for immutable Vulkan objects, when there will not be many
// instances of those objects required at runtime.  The previous sentence puts
// two constraints on proper use of this cache: 1) First, the objects should
// preferably be immutable otherwise much care is required to synchronize
// their usage.  2) Second, this cache is only intended for objects that
// we will not have many instances of during the entire execution of the
// program, otherwise the cache must be _infrequently_ purged.  Proper usage
// model for this cache is in direct contrast with Vulkan object pools, which
// indeed are required to be _frequently_ purged.  That is an important
// distinction.
//

template<typename Factory>
class Cache final {
 public:
  explicit Cache(Factory factory);
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;
  Cache(Cache&&) = default;
  Cache& operator=(Cache&&) = default;
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

  auto retrieve(const Descriptor& descriptor);

  // Only call this function infrequently, if ever.  This cache is only intended
  // for immutable Vulkan objects of which a small finite instances are required
  // at runtime.  A good place to call this function is between model loads.

  void purge();

 private:
  struct Configuration final {
    static constexpr uint32_t kReserve = 64u;
  };

  ska::flat_hash_map<Descriptor, Handle, Hasher> cache_;
  Factory factory_;
};

//
// Impl
//

template<typename Factory>
inline Cache<Factory>::Cache(Factory factory)
  : factory_(std::move(factory)) {
    cache_.reserve(Configuration::kReserve);
}

template<typename Factory>
inline auto Cache<Factory>::retrieve(
    const Descriptor& descriptor) {
  auto iterator = cache_.find(descriptor);
  if C10_UNLIKELY(cache_.cend() == iterator) {
    iterator = cache_.insert({descriptor, factory_(descriptor)}).first;
  }

  return iterator->second.get();
}

template<typename Factory>
inline void Cache<Factory>::purge() {
  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
