#pragma once

#include <memory>

namespace torch {

// Reference:
// https://github.com/llvm-mirror/libcxx/blob/master/include/memory#L3091

template <typename T>
struct unique_type_for {
  using value = std::unique_ptr<T>;
};

template <typename T>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
struct unique_type_for<T[]> {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  using unbounded_array = std::unique_ptr<T[]>;
};

template <typename T, size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
struct unique_type_for<T[N]> {
  using bounded_array = void;
};

template <typename T, typename... Args>
typename unique_type_for<T>::value make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
typename unique_type_for<T>::unbounded_array make_unique(size_t size) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[size]());
}

template <typename T, size_t N, typename... Args>
typename unique_type_for<T>::bounded_array make_unique(Args&&...) = delete;
} // namespace torch
