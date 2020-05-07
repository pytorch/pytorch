#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <ATen/NumericUtils.h>
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

namespace transformation {

/**
 * A transformation function for `torch.Tensor.random_()`, when both `from` and `to` are specified.
 * `range` is `to - from`
 * `base` is `from`
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
  return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

/**
 * A transformation function for `torch.Tensor.random_()`, when `from=min_value(int64_t)` and to=None
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_full_range(V val) {
  return static_cast<T>(static_cast<int64_t>(val));
}

/**
 * A transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`.
 */
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int(V val) {
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
C10_HOST_DEVICE inline dist_acctype<T> uniform_real(V val, T from, T to) {
  constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  dist_acctype<T> x = (val & MASK) * DIVISOR;
  return (x * (to - from) + from);
}

/**
 * Transforms normally distributed `val` with mean 0.0 and standard deviation 1.0 to 
 * normally distributed with `mean` and standard deviation `std`.
 */
template <typename T>
C10_HOST_DEVICE inline T normal(T val, T mean, T std) {
  return val * std + mean;
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * Cauchy distribution with location parameter `median` and scale parameter `sigma`.
 */
template <typename T>
C10_HOST_DEVICE inline T cauchy(T val, T median, T sigma) {
  return median + sigma * at::tan(static_cast<T>(M_PI) * (val - static_cast<T>(0.5)));
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * exponentialy distributed with `lambda` parameter of the distribution.
 */
template <typename T>
C10_HOST_DEVICE __ubsan_ignore_float_divide_by_zero__ inline T exponential(T val, T lambda) {
  return static_cast<T>(-1.0) / lambda * at::log(static_cast<T>(1.0) - val);
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * geometricaly distributed with success probability `p`. 
 */
template <typename T>
C10_HOST_DEVICE inline T geometric(T val, T p) {
  return static_cast<T>(std::ceil(at::log(val) / at::log(static_cast<T>(1.0) - p)));
}

/**
 * Transforms normally distributed `val` with mean 0.0 and standard deviation 1.0 to 
 * log-normally distributed with `mean` and standard deviation `std`.
 */
template <typename T>
C10_HOST_DEVICE inline T log_normal(T val, T mean, T std) {
  return at::exp(val * std + mean);
}

}} // namespace at::transformation
