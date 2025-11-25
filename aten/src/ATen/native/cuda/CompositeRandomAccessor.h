#pragma once

#include <ATen/native/CompositeRandomAccessorCommon.h>

#ifndef USE_ROCM
#include <cuda/std/tuple>
#include <cuda/std/utility>
#endif

namespace at { namespace native {

struct TupleInfoCPU {
  template <typename ...Types>
  using tuple = NO_ROCM(::cuda)::std::tuple<Types...>;

  template <typename ...Types>
  static constexpr auto tie(Types&... args) noexcept {
    return NO_ROCM(::cuda)::std::tie(args...);
  }
};

template <typename KeyAccessor, typename ValueAccessor>
using CompositeRandomAccessorCPU =
  CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfoCPU>;

template <typename Values, typename References>
void swap(
  references_holder<Values, References> rh1,
  references_holder<Values, References> rh2
) {
  return NO_ROCM(::cuda)::std::swap(rh1.data(), rh2.data());
}

template <int N, typename Values, typename References>
auto get(references_holder<Values, References> rh) -> decltype(NO_ROCM(::cuda)::std::get<N>(rh.data())) {
  return NO_ROCM(::cuda)::std::get<N>(rh.data());
}

}} // namespace at::native
