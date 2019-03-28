#pragma once

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>

namespace at { namespace detail {

template <typename T>
inline T load(const void* data, ScalarType src_type) {
  return AT_DISPATCH_ALL_TYPES(src_type, "load", [&]() {
    return at::convert<T>(*(scalar_t*)data);
  });
}

template <typename T>
inline void store(T value, void* dst, ScalarType dst_type) {
  AT_DISPATCH_ALL_TYPES(dst_type, "store", [&]() {
    *(scalar_t*)dst = at::convert<scalar_t>(value);
  });
}

}} // namespace at::detail
