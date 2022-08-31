#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, bool = std::is_integral<T1>::value>
struct promotion {
  using type = double;
};

template<typename T1>
struct promotion<T1, false> {};

template<>
struct promotion<float> {
  using type = float;
};

template<>
struct promotion<double> {
  using type = double;
};

template<>
struct promotion<long double> {
  using type = long double;
};

template<typename... T>
using promotion_t = typename promotion<T...>::type;

template<typename T1>
struct promotion<c10::complex<T1>, false> {
 private:
  using value_type = typename c10::complex<T1>::value_type;
 public:
  using type = decltype(c10::complex<promotion_t<value_type>>{});
};

template<typename T1, typename... T>
struct promote {
  using type = decltype(promotion_t<std::decay_t<T1>>{} + typename promote<T...>::type{});
};

template<typename T1>
struct promote<T1> {
  using type = decltype(promotion_t<std::decay_t<T1>>{});
};

template<typename... T>
using promote_t = typename promote<T...>::type;
}
}
}
}
