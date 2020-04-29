#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <limits>
#include <cstdint>
#include <cassert>

namespace at {

// Using DistAccumType in accumulate types for distributions.
// Note: Ideally we'd be using ATen/AccumulateType.h but looks
// like the there is some inconsistency in how accumulate types
// are mapped currently, e.g. for the cpu side, float is mapped
// to double.
template <typename T>
struct DistAccumType {  };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct DistAccumType<half> { using type = float; };
#endif
template <> struct DistAccumType<Half> { using type = float; };
template <> struct DistAccumType<float> { using type = float; };
template <> struct DistAccumType<double> { using type = double; };

template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

/**
 * A transformation function for `torch.Tensor.random_()`, when both `from` and `to` are specified.
 * `range` is `to - from`
 * `base` is `from`
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_from_to_transformation(V val, uint64_t range, int64_t base) {
  return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

/**
 * A transformation function for `torch.Tensor.random_()`, when `from=min_value(int64_t)` and to=None
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_full_range_transformation(V val) {
  return static_cast<T>(static_cast<int64_t>(val));
}

/**
 * A transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`.
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_transformation(V val) {
  if (std::is_same<T, bool>::value) {
    return static_cast<bool>(val & 1);
  } else if (std::is_same<T, double>::value) {
    return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
  } else if (std::is_same<T, int64_t>::value) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else if (std::is_floating_point<T>::value || std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value) {
    return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
  } else if (std::is_integral<T>::value) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else {
    assert(false);
    return 0;
  }
}

template <typename T, typename V>
C10_HOST_DEVICE inline dist_acctype<T> uniform_real_transformation(V val, T from, T to) {
  constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  dist_acctype<T> x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

} // namespace at
