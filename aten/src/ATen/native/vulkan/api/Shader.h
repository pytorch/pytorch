#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {
namespace api {

// A construct to abstract away source vs binary differences from the rest of
// the codebase.

class Shader final {
 public:
  // A Cache does own the underlying data, and acts as the factory through which
  // Shader objects are to be created.
  class Cache final {
   public:
    Cache() = default;
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = default;
    Cache& operator=(Cache&&) = default;
    ~Cache() = default;

    Shader create_or_retrieve(const char* key, const char* glsl = nullptr);
    Shader create_or_retrieve(const char* key, const uint32_t* spirv = nullptr, uint32_t count = 0u);

   private:
    template <typename Type>
    struct Entry final {
      std::unique_ptr<Type[]> data;
      uint32_t count;
    };

    VkDevice device_;
    std::vector<Entry<char>> sources_;
    std::vector<Entry<uint32_t>> binaries_;
  };

  // A View does NOT own the underlying data and as the name suggests is merely
  // a view onto the data owned elsewhere.
  template <typename Type>
  struct View final {
    Type* data;
    uint32_t count;
  };

  typedef View<char> Source;
  typedef View<uint32_t> Binary;

  Source source;
  Binary binary;

 private:
  explicit Shader(const char* glsl);
  explicit Shader(const uint32_t* spirv);
};

} // namespace api
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
