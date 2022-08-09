#pragma once

namespace at::native::special_functions::detail {
template<typename T1>
T1
lah_number_recurrence(unsigned int n, unsigned int k) {
  if (k > n) {
    return T1(0);
  } else if (n == 0) {
    if (k == 0) {
      return T1(1);
    } else {
      return T1(0);
    }
  } else {
    T1 Lnn = 1;

    for (unsigned int i = 1u; i <= n - k; i++) {
      Lnn *= T1(n - i + 1) * T1(n - i) / T1(i);
    }

    return Lnn;
  }
}

template<typename Tp>
inline Tp
lah_number(unsigned int n, unsigned int k) {
  return lah_number_recurrence<Tp>(n, k);
}
}
