#pragma once

#include <cmath>

#include <ATen/native/special_functions/detail/bessel.h>
#include <ATen/native/special_functions/detail/modified_bessel.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, typename T2>
struct airy_t {
  T1 x;

  T2 ai;
  T2 ai_derivative;

  T2 bi;
  T2 bi_derivative;
};

template<typename T1>
airy_t<T1, T1>
airy(T1 z, bool exp = false) {
  using T2 = airy_t<T1, T1>;

  if (std::isnan(z)) {
    return {
        z,
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
    };
  } else if (z == std::numeric_limits<T1>::infinity()) {
    return {
        z,
        T1(0),
        T1(0),
        std::numeric_limits<T1>::infinity(),
        std::numeric_limits<T1>::infinity(),
    };
  } else if (z == -std::numeric_limits<T1>::infinity()) {
    return {
        z,
        T1(0),
        T1(0),
        T1(0),
        T1(0),
    };
  } else {
    const auto abs_z = std::abs(z);
    const auto sqrt_abs_z = std::sqrt(abs_z);
    const auto sqrt3_v = c10::numbers::sqrt3_v<T1>;
    const auto abs_z_2_sqrt_abs_z_3 = T1(2) * abs_z * sqrt_abs_z / T1(3);
    const auto exp_2_abs_z_sqrt_abs_z_3 = std::exp(abs_z_2_sqrt_abs_z_3);

    if (z > T1(0)) {
      const auto modified_bessel_2_3 = modified_bessel(T1(2) / T1(3), abs_z_2_sqrt_abs_z_3, exp);

      const auto modified_bessel_2_3_i = modified_bessel_2_3.i;
      const auto modified_bessel_2_3_k = modified_bessel_2_3.k;

      const auto modified_bessel_1_3 = modified_bessel(T1(1) / T1(3), abs_z_2_sqrt_abs_z_3, false);

      const auto modified_bessel_1_3_i = modified_bessel_1_3.i;
      const auto modified_bessel_1_3_k = modified_bessel_1_3.k;

      const auto pi_v = c10::numbers::pi_v<T1>;

      const auto sqrt3_v_pi_v = sqrt3_v * pi_v;

      const auto modified_bessel_2_3_i_sqrt3_v_2 = T1(2) * modified_bessel_2_3_i / sqrt3_v;

      const auto modified_bessel_1_3_k_pi_v = modified_bessel_1_3_k / pi_v;

      if (exp) {
        const auto ai_derivative = -z * modified_bessel_2_3_k / sqrt3_v_pi_v * exp_2_abs_z_sqrt_abs_z_3;
        const auto bi_derivative =
            +z * (modified_bessel_2_3_k / pi_v + modified_bessel_2_3_i_sqrt3_v_2) / exp_2_abs_z_sqrt_abs_z_3;

        const auto ai = sqrt_abs_z * modified_bessel_1_3_k / sqrt3_v_pi_v * exp_2_abs_z_sqrt_abs_z_3;
        const auto bi = sqrt_abs_z * (modified_bessel_1_3_k_pi_v + T1(2) * modified_bessel_1_3_i / sqrt3_v)
            / exp_2_abs_z_sqrt_abs_z_3;

        return {z, ai, ai_derivative, bi, bi_derivative};
      } else {
        const auto ai = sqrt_abs_z * modified_bessel_1_3_k / sqrt3_v_pi_v * T1(1);
        const auto bi = sqrt_abs_z * (modified_bessel_1_3_k_pi_v + T1(2) * modified_bessel_1_3_i / sqrt3_v) / T1(1);

        const auto ai_derivative = -z * modified_bessel_2_3_k / sqrt3_v_pi_v * T1(1);
        const auto bi_derivative = +z * (modified_bessel_2_3_k / pi_v + modified_bessel_2_3_i_sqrt3_v_2) / T1(1);

        return {z, ai, ai_derivative, bi, bi_derivative};
      }
    } else if (z < T1(0)) {
      const auto bessel_1_3 = bessel(abs_z_2_sqrt_abs_z_3, T1(1) / T1(3));
      const auto bessel_2_3 = bessel(abs_z_2_sqrt_abs_z_3, T1(2) / T1(3));

      const auto bessel_1_3_j = bessel_1_3.j;
      const auto bessel_1_3_y = bessel_1_3.y;

      const auto bessel_2_3_j = bessel_2_3.j;
      const auto bessel_2_3_y = bessel_2_3.y;

      const auto negative_bessel_1_3 = bessel_1_3_j - bessel_1_3_y / sqrt3_v;
      const auto positive_bessel_1_3 = bessel_1_3_y + bessel_1_3_j / sqrt3_v;

      if (exp) {
        const auto ai = +sqrt_abs_z * negative_bessel_1_3 / T1(2) * exp_2_abs_z_sqrt_abs_z_3;
        const auto bi = -sqrt_abs_z * positive_bessel_1_3 / T1(2) / exp_2_abs_z_sqrt_abs_z_3;

        const auto ai_derivative = abs_z * (bessel_2_3_y / sqrt3_v + bessel_2_3_j) / T1(2) * exp_2_abs_z_sqrt_abs_z_3;
        const auto bi_derivative = abs_z * (bessel_2_3_j / sqrt3_v - bessel_2_3_y) / T1(2) / exp_2_abs_z_sqrt_abs_z_3;

        return {z, ai, ai_derivative, bi, bi_derivative};
      } else {
        const auto ai = +sqrt_abs_z * negative_bessel_1_3 / T1(2) * T1(1);
        const auto bi = -sqrt_abs_z * positive_bessel_1_3 / T1(2) / T1(1);

        const auto ai_derivative = abs_z * (bessel_2_3_y / sqrt3_v + bessel_2_3_j) / T1(2) * T1(1);
        const auto bi_derivative = abs_z * (bessel_2_3_j / sqrt3_v - bessel_2_3_y) / T1(2) / T1(1);

        return {z, ai, ai_derivative, bi, bi_derivative};
      }
    } else {
      return {
          z,
          +T1(0.3550280538878172392600631860041831763979791741991772L),
          -T1(0.2588194037928067984051835601892039634790911383549300L),
          +T1(0.3550280538878172392600631860041831763979791741991772L) * sqrt3_v,
          +T1(0.2588194037928067984051835601892039634790911383549300L) * sqrt3_v,
      };
    }
  }
}
}
}
}
}
