#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

template<typename Key, typename Value, typename Factory>
class Cache final {
 public:
  Cache() = delete;
  explicit Cache(Factory factory);
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;
  Cache(Cache&&) = default;
  Cache& operator=(Cache&&) = default;
  ~Cache() = default;

  typedef typename Factory::Descriptor Descriptor;

  // Create or retrieve a resource.
  //
  // If descriptor is null, this operation is a simple cache lookup and returns
  // the Value corresponding to Key if present in the cache, or a default-constructed
  // Value otherwise, which in the case of Vulkan objects, can be conveniently compared
  // against VK_NULL_HANDLE in case the intended behavior is to check the presence of
  // an item in the cache.
  //
  // On the other hand, if descriptor is not null, this operation is a cache lookup
  // in addition to a resource construction using the provided Factory if and only if
  // the resource is not already present in the cache.  Regardless, this function
  // returns  with the already present, or newly created item, available in the cache
  // for future use.

  Value retrieve(Key key, const Descriptor* descriptor = nullptr);

 private:
  struct Configuration final {
    static constexpr uint32_t kReserve = 16u;
  };

  Factory factory_;
  ska::flat_hash_map<Key, Handle<Value, typename Factory::Deleter>> cache_;

 private:
  static_assert(
      std::is_default_constructible<Value>::value,
      "Value must be default constructible.");
};

template<typename Key, typename Value, typename Factory>
inline Cache<Key, Value, Factory>::Cache(Factory factory)
  : factory_(std::move(factory)) {
    cache_.reserve(Configuration::kReserve);
}

template<typename Key, typename Value, typename Factory>
inline Value Cache<Key, Value, Factory>::retrieve(
    const Key key,
    const Descriptor* const descriptor) {
  auto iterator = cache_.find(key);
  if (cache_.cend() == iterator) {
    if (!descriptor) {
      return Value{};
    }

    cache_.insert({key, factory_(*descriptor)});
  }

  return iterator->second.get();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
