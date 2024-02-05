#pragma once

#include <ATen/detail/FunctionTraits.h>

#include <sycl/sycl.hpp>

namespace sycl {

template <typename ker_t>
static inline void kernel_submit(
    int64_t global_range,
    int64_t local_range,
    sycl::queue q,
    ker_t ker) {
  auto cgf = [&](sycl::handler &cgh) {
    cgh.parallel_for<ker_t>(
        sycl::nd_range<1>(
            sycl::range<1>(global_range),
            sycl::range<1>(local_range)),
        ker);
  };
  // XXX: c10::xpu::getStreamFromPool().queue();
  q.submit(cgf);
}

// Call for kernels using shared memory. The current SYCL command group handler
// is required to create shared memory (SYCL local accessor).
// To use sycl::ker_creator_t to define a creator for kernel.
template <typename ker_t, typename ker_creator_t>
static inline void kernel_submit(
    int64_t global_range,
    int64_t local_range,
    sycl::queue q,
    ker_creator_t creator) {
  using traits = function_traits<ker_creator_t>;
  static_assert(
      std::is_same<ker_t, typename traits::result_type>::value,
      "Kernel type does not match with the return type of kernel creator ...");
  auto cgf = [&](sycl::handler &cgh) {
    ker_t ker = creator(cgh);
    cgh.parallel_for<ker_t>(
        sycl::nd_range<1>(
            sycl::range<1>(global_range),
            sycl::range<1>(local_range)),
        ker);
  };
  // XXX: c10::xpu::getStreamFromPool().queue();
  q.submit(cgf);
}

} // namespace sycl
