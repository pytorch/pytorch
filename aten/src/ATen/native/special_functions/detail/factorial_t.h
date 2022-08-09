#pragma once

namespace at::native::special_functions::detail {
template<typename T1>
struct factorial_t {
  int n;
  T1 factorial;
  T1 log_factorial;
};
}
