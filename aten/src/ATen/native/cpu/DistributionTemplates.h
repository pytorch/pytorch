#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <mutex>

namespace at { namespace native { namespace templates {

template<typename RNG>
void cauchy_kernel(TensorIterator& iter, double median, double sigma, RNG* generator) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cauchy_cpu", [&]() {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    cpu_serial_kernel(iter, [median, sigma, generator]() -> scalar_t {
      at::cauchy_distribution<double> cauchy(median, sigma);
      return (scalar_t)cauchy(generator);
    });
  });
}

}}}
