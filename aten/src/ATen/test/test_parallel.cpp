#include "gtest/gtest.h"

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_seed.h"

using namespace at;

TEST(TestParallel, TestParallel) {
  manual_seed(123, at::kCPU);
  set_num_threads(1);

  Tensor a = rand({1, 3});
  a[0][0] = 1;
  a[0][1] = 0;
  a[0][2] = 0;
  Tensor as = rand({3});
  as[0] = 1;
  as[1] = 0;
  as[2] = 0;
  ASSERT_TRUE(a.sum(0).equal(as));
}

TEST(TestParallel, NestedParallel) {
  Tensor a = ones({1024, 1024});
  auto expected = a.sum();
  // check that calling sum() from within a parallel block computes the same result
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    if (begin == 0) {
      ASSERT_TRUE(a.sum().equal(expected));
    }
  });
}

TEST(TestParallel, Exceptions) {
  // parallel case
  ASSERT_THROW(
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception");
    }),
    std::runtime_error);

  // non-parallel case
  ASSERT_THROW(
    at::parallel_for(0, 1, 1000, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception");
    }),
    std::runtime_error);
}
