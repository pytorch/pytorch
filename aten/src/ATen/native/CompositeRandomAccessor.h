#pragma once

#include <ATen/native/CompositeRandomAccessorCommon.h>

namespace at::native {

struct TupleInfoCPU {
  template <typename ...Types>
  using tuple = std::tuple<Types...>;

  template <typename ...Types>
  static constexpr auto tie(Types&... args) noexcept {
    return std::tie(args...);
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
  return std::swap(rh1.data(), rh2.data());
}

template <int N, typename Values, typename References>
auto get(references_holder<Values, References> rh) -> decltype(std::get<N>(rh.data())) {
  return std::get<N>(rh.data());
}

} // namespace at::native
