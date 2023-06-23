#pragma once

#include <memory>
#include <mutex>

namespace at::native {

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  static_assert(std::is_standard_layout_v<Params>, "Params is not POD");

  size_t operator()(const Params& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange(sizeof(Params))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Params>
struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contents as char* when comparing
  static_assert(std::is_standard_layout_v<Params>, "Params is not POD");

  bool operator()(const Params& a, const Params& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

// Provide explicit byte-for-byte constructors to avoid uwittingly leaving
// padding bytes unitialized (e.g., when passing Params by value)
template <typename T>
struct ParamsWrapper {
  T pod;

  ParamsWrapper() {
    memset(&(this->pod), 0, sizeof(T));
  }

  ParamsWrapper(const ParamsWrapper<T> &other) {
    memcpy(&(this->pod), &(other.pod), sizeof(T));
  }

  ParamsWrapper(ParamsWrapper<T> &&other) {
    memcpy(&(this->pod), &(other.pod), sizeof(T));
  }
};

// Wrapped version: this allows the outer struct to have custom copy and move
// constructors for additional safety
template <typename ParamsWrapper>
struct ParamsWrapperHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  static_assert(std::is_standard_layout_v<decltype(ParamsWrapper::pod)>, "ParamsWrapper wraps non-POD data");

  size_t operator()(const ParamsWrapper& params_wrapper) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&(params_wrapper.pod));
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange(sizeof(params_wrapper.pod))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename ParamsWrapper>
struct ParamsWrapperEqual {
  // Params must be a POD because we read out its memory
  // contents as char* when comparing
  static_assert(std::is_standard_layout_v<decltype(ParamsWrapper::pod)>, "ParamsWrapper wraps non-POD data");

  bool operator()(const ParamsWrapper& a, const ParamsWrapper& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&(a.pod));
    auto ptr2 = reinterpret_cast<const uint8_t*>(&(b.pod));
    return memcmp(ptr1, ptr2, sizeof(a.pod)) == 0;
  }
};


}  // at::native
