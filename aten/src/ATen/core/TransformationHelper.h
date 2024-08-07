#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/MathConstants.h>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <limits>
#include <type_traits>

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
template <> struct DistAccumType<BFloat16> { using type = float; };
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
 * In order to prevent compiler warnings reported in GitHub issue 46391, T can't be float or double
 * in this overloaded version
 */
template <typename T, typename V>
C10_HOST_DEVICE inline std::enable_if_t<!(std::is_floating_point_v<T>), T>uniform_int(V val) {
  if constexpr (std::is_same_v<T, bool>) {
    return static_cast<bool>(val & 1);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else if constexpr (std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>) {
    return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
  } else {
    assert(false);
    return 0;
  }
}

/**
 * An overloaded transformation function for `torch.Tensor.random_()`, when used without specifying `from` and `to`,
 * added to fix compiler warnings reported in GitHub issue 46391. T is either float or double in this version.
 */
template<typename T, typename V>
C10_HOST_DEVICE inline std::enable_if_t<std::is_floating_point_v<T>, T>uniform_int(V val) {
  return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
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
  // https://en.wikipedia.org/wiki/Cauchy_distribution#Cumulative_distribution_function
  // __tanf overflows and returns `inf/-inf` when (val > 1 - eps) or (val < 0 + eps),
  // thus we clip those values.
  constexpr T eps = std::numeric_limits<T>::epsilon();
  constexpr T one_minus_eps = 1 - eps;
  constexpr T zero_plus_eps = 0 + eps;
  val = (val > one_minus_eps ? one_minus_eps : val);
  val = (val < zero_plus_eps ? zero_plus_eps : val);
  return median + sigma * at::tan(c10::pi<T> * (val - static_cast<T>(0.5)));
}

template <>
C10_HOST_DEVICE inline double cauchy(double val, double median, double sigma) {
  // https://en.wikipedia.org/wiki/Cauchy_distribution#Cumulative_distribution_function
  return median + sigma * at::tan(c10::pi<double> * (val - static_cast<double>(0.5)));
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * exponentially distributed with `lambda` parameter of the distribution.
 */
template <typename T>
C10_HOST_DEVICE inline T exponential(T val, T lambda) {
  // https://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
  // Different implementations for CUDA and CPU to preserve original logic
  // TODO: must be investigated and unified!!!
  // https://github.com/pytorch/pytorch/issues/38662
#if defined(__CUDACC__) || defined(__HIPCC__)
      // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/16706
      // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
      // we need log to be not 0, and not underflow when converted to half
      // fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1 args
  auto log = val >= static_cast<T>(1.) - std::numeric_limits<T>::epsilon() / 2
      ? -std::numeric_limits<T>::epsilon() / 2
      : at::log(val);
  return static_cast<T>(-1.0) / lambda * log;
#else
  return static_cast<T>(-1.0) / lambda * at::log1p(-val);
#endif
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * geometrically distributed with success probability `p`.
 */
template <typename T>
C10_HOST_DEVICE inline T geometric(T val, T p) {
  // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
  return static_cast<T>(::ceil(at::log(val) / at::log1p(-p)));
}

/**
 * Transforms normally distributed `val` to log-normally distributed.
 */
template <typename T>
C10_HOST_DEVICE inline T log_normal(T val) {
  // https://en.wikipedia.org/wiki/Log-normal_distribution#Mode,_median,_quantiles
  return at::exp(val);
}

/**
 * Transforms uniformly distributed `val` between 0.0 and 1.0 to
 * bernoulli distributed with success probability `p`.
 */
template <typename T>
C10_HOST_DEVICE inline T bernoulli(T val, T p) {
  return val < p;
}

}} // namespace at::transformation
