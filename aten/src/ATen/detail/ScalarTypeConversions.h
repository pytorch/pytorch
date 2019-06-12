#pragma once

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarType.h>
#include <ATen/core/Half.h>

namespace at { namespace detail {

template <typename T>
inline T load(const void* data, ScalarType src_type) {
  return AT_DISPATCH_ALL_TYPES(CPU(src_type), "load", [&]() {
    return at::convert<T>(*(scalar_t*)data);
  });
}

template <typename T>
inline void store(T value, void* dst, ScalarType dst_type) {
  AT_DISPATCH_ALL_TYPES(CPU(dst_type), "store", [&]() {
    *(scalar_t*)dst = at::convert<scalar_t>(value);
  });
}

}} // namespace at::detail
