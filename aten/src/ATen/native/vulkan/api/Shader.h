#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {
namespace api {

// A construct to abstract away shader source and binary differences from the
// rest of the code.  Shaders are small programs that run on the GPU, and for
// the purpose of our implementation come in either source code or binary
// format.  If the shaders are in human readable [GLSL] source code format, we
// refer to them simply as 'shader sources' below, while we use the term
// 'shader binaries' to refer to shader programs that are compiled into machine
// executable format.

// It is critical to keep in mind that a `Shader` (i.e. an object of class
// Shader) does NOT own its underlying data and is merely a view onto data
// owned elsewhere.  Depending on usage, 'elsewhere' is typically one of the
// following:
//
// 1) The shader cache (i.e. Shader::Cache) - in case we are dealing with shader
//    source code whose string representation is either loaded from files or
//    compiled into PyTorch's [readonly] data section in the form of global
//    variables with a static lifetime.  For this usage model, access the shaders
//    through the cache to allow the cache compile and manage the shader binaries'
//    lifetimes.
//
// 2) The shader cache (i.e. Shader::Cache) - in case we are dealing with shader
//    binaries whose byte representation is loaded from a file with the intent
//    of having the cache manage the shader's lifetime.  For this usage model,
//    access the shaders through the cache to offload memory management.
//
// 3) The current process - in case we are dealing with binary shaders whose
//    byte representation is compiled into PyTorch's [readonly] data section
//    in the form of static or global variables.  For this model, directly
//    construct Shader objects by passing the pointers to Shader without going
//    through the cache since, as global variables, the shader binaries'
//    lifetimes is already tied to the runtime of the program and there is no
//    need in maintaining an extra copy in the cache.

class Shader final {
 public:
  // A Cache owns the underlying data, and acts as the factory through which
  // Shader objects are to be retrieved.
  class Cache final {
   public:
    Cache();
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = default;
    Cache& operator=(Cache&&) = default;
    ~Cache() = default;

    Shader retrieve(const char* key, const char* glsl = nullptr);
    Shader retrieve(const char* key, const uint32_t* spirv = nullptr, uint32_t count = 0u);

   private:
    template <typename Type>
    struct Entry final {
      std::vector<uint32_t> data;
    };

    typedef Entry<char> Source;
    typedef Entry<uint32_t> Binary;

    ska::flat_hash_map<const char*, Binary> shaders_;

    struct Compiler;
    std::unique_ptr<Compiler> compiler_;
  };

  const uint32_t* data;
  uint32_t count;
};

} // namespace api
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
