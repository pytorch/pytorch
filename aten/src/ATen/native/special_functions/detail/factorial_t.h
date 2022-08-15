#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct factorial_t {
  int n;
  T1 factorial;
  T1 log_factorial;
};
}
}
}
}
