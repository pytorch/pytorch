#pragma once

#include <c10/util/irange.h>
#include <memory>
#include <mutex>
#include <type_traits>

namespace at::native {

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  static_assert(std::is_standard_layout_v<Params>, "Params is not POD");

  size_t operator()(const Params& params) const noexcept {
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

  bool operator()(const Params& a, const Params& b) const noexcept {
    return memcmp(&a, &b, sizeof(Params)) == 0;
  }
};

// Provide explicit byte-for-byte constructors to avoid uwittingly leaving
// padding bytes uninitialized (e.g., when passing Params by value)
template <typename T>
struct ParamsWrapper {
  T pod;
  static_assert(
      std::is_standard_layout_v<T>,
      "ParamsWrapper cannot wrap non-POD data");
  static_assert(std::is_trivially_copyable_v<T>,
      "ParamsWrapper requires trivially copyable T");

  ParamsWrapper() noexcept {
    memset(&(this->pod), 0, sizeof(this->pod));
  }

  ParamsWrapper(const ParamsWrapper& other) noexcept {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
  }

  ParamsWrapper& operator=(const ParamsWrapper& other) noexcept {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
    return *this;
  }

  ParamsWrapper(ParamsWrapper&& other) = delete;
  ParamsWrapper& operator=(ParamsWrapper&& other) = delete;

  inline friend bool operator==(
      const ParamsWrapper& lhs,
      const ParamsWrapper& rhs) noexcept {
    return memcmp(&lhs.pod, &rhs.pod, sizeof(T)) == 0;
  }
};

// Wrapped version: this allows the outer struct to have custom copy and move
// constructors for additional safety
template <typename ParamsWrapper>
struct ParamsWrapperHash {
  size_t operator()(const ParamsWrapper& params_wrapper) const noexcept {
    ParamsHash<decltype(ParamsWrapper::pod)> hasher;
    return hasher(params_wrapper.pod);
  }
};

} // namespace at::native
