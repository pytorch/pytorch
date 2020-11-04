#ifndef CAFFE2_CORE_DISTRIBUTIONS_STUBS_H_
#define CAFFE2_CORE_DISTRIBUTIONS_STUBS_H_

#include <c10/macros/Macros.h>

/**
 * This file provides distributions compatible with
 * ATen/core/DistributionsHelper.h but backed with the std RNG implementation
 * instead of the ATen one.
 *
 * Caffe2 mobile builds currently do not depend on all of ATen so this is
 * required to allow using the faster ATen RNG for normal builds but keep the
 * build size small on mobile. RNG performance typically doesn't matter on
 * mobile builds since the models are small and rarely using random
 * initialization.
 */

namespace at {
namespace {

template <typename R, typename T>
struct distribution_adapter {
  template <typename... Args>
  C10_HOST_DEVICE inline distribution_adapter(Args... args)
      : distribution_(std::forward<Args>(args)...) {}

  template <typename RNG>
  C10_HOST_DEVICE inline R operator()(RNG generator) {
    return distribution_(*generator);
  }

 private:
  T distribution_;
};

template <typename T>
struct uniform_int_from_to_distribution
    : distribution_adapter<T, std::uniform_int_distribution<T>> {
  C10_HOST_DEVICE inline uniform_int_from_to_distribution(
      uint64_t range,
      int64_t base)
      : distribution_adapter<T, std::uniform_int_distribution<T>>(
            base,
            // std is inclusive, at is exclusive
            base + range - 1) {}
};

template <typename T>
using uniform_real_distribution =
    distribution_adapter<T, std::uniform_real_distribution<T>>;

template <typename T>
using normal_distribution =
    distribution_adapter<T, std::normal_distribution<T>>;

template <typename T>
using bernoulli_distribution =
    distribution_adapter<T, std::bernoulli_distribution>;

template <typename T>
using exponential_distribution =
    distribution_adapter<T, std::exponential_distribution<T>>;

template <typename T>
using cauchy_distribution =
    distribution_adapter<T, std::cauchy_distribution<T>>;

template <typename T>
using lognormal_distribution =
    distribution_adapter<T, std::lognormal_distribution<T>>;

} // namespace
} // namespace at

#endif // CAFFE2_CORE_DISTRIBUTIONS_STUBS_H_
