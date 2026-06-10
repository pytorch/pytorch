#pragma once
#include <torch/headeronly/macros/Macros.h>
#include <cstring>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
namespace detail {

template <typename T>
struct LoadImpl {
  C10_HOST_DEVICE static T apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  C10_HOST_DEVICE static bool apply(const void* src) {
    static_assert(sizeof(bool) == sizeof(char));
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

} // namespace detail

template <typename T>
C10_HOST_DEVICE constexpr T load(const void* src) {
  return detail::LoadImpl<T>::apply(src);
}

template <typename scalar_t>
C10_HOST_DEVICE constexpr scalar_t load(const scalar_t* src) {
  return detail::LoadImpl<scalar_t>::apply(src);
}

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
using torch::headeronly::load;
} // namespace c10
