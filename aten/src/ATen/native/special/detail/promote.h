#pragma once

#include <ATen/native/special/detail/promote.h>
#include <ATen/native/special/detail/promotion_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1, typename... T>
struct promote {
  using type = decltype(promotion_t<std::decay_t<T1>>{} + typename promote<T...>::type{});
};

template<typename T1>
struct promote<T1> {
  using type = decltype(promotion_t<std::decay_t<T1>>{});
};
}
}
}
}
