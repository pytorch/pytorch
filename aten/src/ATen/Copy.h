#pragma once

#include "ATen/ATen.h"

namespace at {

template <typename T>
struct inter_copy_type {
  using type = T;
};

template <>
struct inter_copy_type<uint8_t> {
  using type = int64_t;
};

template <typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;

Tensor copy_cpu(Tensor& dst, const Tensor& src);

} // namespace at
