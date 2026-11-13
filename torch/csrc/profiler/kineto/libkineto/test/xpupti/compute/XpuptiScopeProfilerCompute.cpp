//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// This file is copied from Intel repository:
// https://github.com/intel/pti-gpu/blob/master/sdk/samples/metrics_scope/main_metrics_scope.cc
// and edited:
// - removed unnecessary code
// - changed formatting

// Mark that we are building the library so the header exports (not imports)
// ComputeOnXpu on Windows.
#define XPUPTI_COMPUTE_BUILDING
#include "XpuptiScopeProfilerCompute.h"

#include <sycl/sycl.hpp>
#include <cmath>

constexpr float A_VALUE = 0.128f;
constexpr float B_VALUE = 0.256f;

void GEMM(
    const float* a,
    const float* b,
    float* c,
    unsigned size,
    sycl::id<2> id) {
  int i = id.get(0);
  int j = id.get(1);
  float sum = 0.0f;
  for (unsigned k = 0; k < size; ++k) {
    sum += a[i * size + k] * b[k * size + j];
  }
  c[i * size + j] = sum;
}

static void Run(
    sycl::queue queue,
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    unsigned size) {
  sycl::buffer<float, 1> a_buf(a.data(), a.size());
  sycl::buffer<float, 1> b_buf(b.data(), b.size());
  sycl::buffer<float, 1> c_buf(c.data(), c.size());

  [[maybe_unused]] sycl::event event = queue.submit([&](sycl::handler& cgh) {
    auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
    auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
    auto c_acc = c_buf.get_access<sycl::access::mode::write>(cgh);

    cgh.parallel_for<class __GEMM>(
        sycl::range<2>(size, size), [=](sycl::id<2> id) {
          auto a_acc_ptr = a_acc.get_multi_ptr<sycl::access::decorated::no>();
          auto b_acc_ptr = b_acc.get_multi_ptr<sycl::access::decorated::no>();
          auto c_acc_ptr = c_acc.get_multi_ptr<sycl::access::decorated::no>();
          GEMM(a_acc_ptr.get(), b_acc_ptr.get(), c_acc_ptr.get(), size, id);
        });
  });
  queue.wait_and_throw();
}

XPUPTI_COMPUTE_API void ComputeOnXpu(unsigned size, unsigned repeatCount) {
  unsigned sizeSq = size * size;
  sycl::device dev = sycl::device(sycl::gpu_selector_v);
  sycl::property_list propList{sycl::property::queue::in_order()};
  sycl::queue queue(
      dev, sycl::async_handler{}, propList); // Main runandcheck kernel

  std::cout << "DPC++ Matrix Multiplication (matrix size: " << size << " x "
            << size << ", repeats " << repeatCount << " times)" << std::endl;
  std::cout << "Target device: "
            << queue.get_info<sycl::info::queue::device>()
                   .get_info<sycl::info::device::name>()
            << std::endl;

  std::vector<float> a(sizeSq, A_VALUE);
  std::vector<float> b(sizeSq, B_VALUE);
  std::vector<float> c(sizeSq, 0.0f);

  auto start = std::chrono::steady_clock::now();

  for (unsigned i = 0; i < repeatCount; ++i) {
    Run(queue, a, b, c, size);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  std::cout << "Total execution time: " << time.count() << " sec" << std::endl;
}
