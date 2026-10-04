#pragma once

#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace at::native {

namespace detail {

// Smallest representable value used as the identity element of the running
// max/min scan. For floating point types this is -infinity so that the scan
// behaves like jax.lax.associative_scan (NaN is propagated by combine).
template <typename scalar_t>
C10_HOST_DEVICE scalar_t scan_lowest() {
  if constexpr (std::is_floating_point_v<scalar_t>) {
    return -std::numeric_limits<scalar_t>::infinity();
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return c10::Half(-std::numeric_limits<float>::infinity());
  } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    return c10::BFloat16(-std::numeric_limits<float>::infinity());
  } else if constexpr (c10::is_complex<scalar_t>::value) {
    // max/min are not supported for complex types; the combine is never
    // invoked with this identity.
    return scalar_t(0);
  } else {
    return std::numeric_limits<scalar_t>::min();
  }
}

// Identity element of the running max/min scan. For floating point types this
// is +infinity so the min scan seeds with a neutral element.
template <typename scalar_t>
C10_HOST_DEVICE scalar_t scan_highest() {
  if constexpr (std::is_floating_point_v<scalar_t>) {
    return std::numeric_limits<scalar_t>::infinity();
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return c10::Half(std::numeric_limits<float>::infinity());
  } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    return c10::BFloat16(std::numeric_limits<float>::infinity());
  } else if constexpr (c10::is_complex<scalar_t>::value) {
    // max/min are not supported for complex types; the combine is never
    // invoked with this identity.
    return scalar_t(0);
  } else {
    return std::numeric_limits<scalar_t>::max();
  }
}

// NaN-propagating max/min, matching the behavior of jax.lax.max/min.
template <typename scalar_t>
C10_HOST_DEVICE scalar_t nan_max(scalar_t a, scalar_t b) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    // max/min are not supported for complex types; never reached.
    return a;
  } else {
    if (at::_isnan(a)) return a;
    if (at::_isnan(b)) return b;
    return a > b ? a : b;
  }
}

template <typename scalar_t>
C10_HOST_DEVICE scalar_t nan_min(scalar_t a, scalar_t b) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    // max/min are not supported for complex types; never reached.
    return a;
  } else {
    if (at::_isnan(a)) return a;
    if (at::_isnan(b)) return b;
    return a < b ? a : b;
  }
}

} // namespace detail

// A scan element with L scalar components. L == 1 for the pointwise combine
// modes (add/mul/max/min) and L == 2 for the linear recurrence mode, whose
// combine operates on (a, b) pairs.
template <typename scalar_t, int L>
struct ScanVec {
  scalar_t v[L];
};

template <typename scalar_t, int L>
struct CombineAdd {
  static C10_HOST_DEVICE ScanVec<scalar_t, L> identity() {
    ScanVec<scalar_t, L> res{};
    for (int i = 0; i < L; ++i) {
      res.v[i] = 0;
    }
    return res;
  }
  static C10_HOST_DEVICE ScanVec<scalar_t, L> combine(
      const ScanVec<scalar_t, L>& a,
      const ScanVec<scalar_t, L>& b) {
    ScanVec<scalar_t, L> res;
    for (int i = 0; i < L; ++i) {
      res.v[i] = a.v[i] + b.v[i];
    }
    return res;
  }
};

template <typename scalar_t, int L>
struct CombineMul {
  static C10_HOST_DEVICE ScanVec<scalar_t, L> identity() {
    ScanVec<scalar_t, L> res{};
    for (int i = 0; i < L; ++i) {
      res.v[i] = 1;
    }
    return res;
  }
  static C10_HOST_DEVICE ScanVec<scalar_t, L> combine(
      const ScanVec<scalar_t, L>& a,
      const ScanVec<scalar_t, L>& b) {
    ScanVec<scalar_t, L> res;
    for (int i = 0; i < L; ++i) {
      res.v[i] = a.v[i] * b.v[i];
    }
    return res;
  }
};

template <typename scalar_t>
struct CombineMax {
  using vec_t = ScanVec<scalar_t, 1>;
  static C10_HOST_DEVICE vec_t identity() {
    return vec_t{detail::scan_lowest<scalar_t>()};
  }
  static C10_HOST_DEVICE vec_t combine(const vec_t& a, const vec_t& b) {
    return vec_t{detail::nan_max(a.v[0], b.v[0])};
  }
};

template <typename scalar_t>
struct CombineMin {
  using vec_t = ScanVec<scalar_t, 1>;
  static C10_HOST_DEVICE vec_t identity() {
    return vec_t{detail::scan_highest<scalar_t>()};
  }
  static C10_HOST_DEVICE vec_t combine(const vec_t& a, const vec_t& b) {
    return vec_t{detail::nan_min(a.v[0], b.v[0])};
  }
};

// combine((a1, b1), (a2, b2)) = (a2 * a1, a2 * b1 + b2), the associative
// operation behind the Mamba-style SSM recurrence h_t = a_t * h_{t-1} + b_t.
// A segment's transform maps h_{l-1} -> h_r = A * h_{l-1} + H; concatenating
// left segment (a1, b1) then right segment (a2, b2) composes as a2 applied
// after a1, so the accumulated A is a2 * a1 and the accumulated H is
// a2 * b1 + b2.
template <typename scalar_t>
struct CombineLinearRecurrence {
  using vec_t = ScanVec<scalar_t, 2>;
  static C10_HOST_DEVICE vec_t identity() {
    return vec_t{static_cast<scalar_t>(1), static_cast<scalar_t>(0)};
  }
  static C10_HOST_DEVICE vec_t combine(const vec_t& a, const vec_t& b) {
    return vec_t{
        static_cast<scalar_t>(a.v[0] * b.v[0]),
        static_cast<scalar_t>(b.v[0] * a.v[1] + b.v[1]),
    };
  }
};

} // namespace at::native
