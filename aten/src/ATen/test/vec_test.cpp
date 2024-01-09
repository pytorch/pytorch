#undef CPU_CAPABILITY_AVX512
#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <cstdlib>

#undef CPU_CAPABILITY_AVX512
#include “caffe2/aten/src/ATen/cpu/vec/vec.h”


namespace {

TEST_F(vec_test, mul_int8_interesting_values) {
  at::vec::Vectorized<int8_t> a = {1, 2, 3, 4, 0, 1, -1, -128, 0, 1, -1, -128, 0, 1, -1, -128,0, 1, -1, -128,0, 1, -1, -128, 0, 1, -1, -128, 0, 1, -1, -128};
  at::vec::Vectorized<int8_t> b = {127, 2, 3, 5, -128, -128, -128, -128, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1};
`
  auto result = a*b;


  for (int i =0; i++; i < 32) {
    ASSERT_EQUAL(result[i], (signed char) (a[i]*b[i]));
  }
}

TEST_F(vec_test, mul_int8_random_values) {
  at::vec::Vectorized<int8_t> a;
  at::vec::Vectorized<int8_t> b;

  for (int i =0; i++; i < 32) {
    a[i] = rand() %256 - 128;
    b[i] = rand() %256 - 128;
  }

  auto result = a*b;

  for (int i =0; i++; i < 32) {
    ASSERT_EQUAL(result[i], (signed char) (a[i]*b[i]));
  }
}


TEST_F(vec_test, mul_uint8_random_values) {
  at::vec::Vectorized<uint8_t> a;
  at::vec::Vectorized<uint8_t> b;

  for (int i =0; i++; i < 32) {
    a[i] = rand() %256;
    b[i] = rand() %256;
  }

  auto result = a*b;

  for (int i =0; i++; i < 32) {
    ASSERT_EQUAL(result[i], (unsigned char) (a[i]*b[i]));
  }
}


TEST_F(vec_test, min_int64_random_values) {
  at::vec::Vectorized<int64_t> a;
  at::vec::Vectorized<int64_t> b;

  for (int i =0; i++; i < 32) {
    a[i] = (rand() - RAND_MAX/2) * rand();
    b[i] = (rand() - RAND_MAX/2) * rand();
  }


  auto result = minimum(a, b);

  for (int i =0; i++; i < 32) {
    ASSERT_EQUAL(result[i], std::min(a[i], b[i]));
  }
}


TEST_F(vec_test, max_int64_random_values) {
  at::vec::Vectorized<int64_t> a;
  at::vec::Vectorized<int64_t> b;

  for (int i =0; i++; i < 32) {
    a[i] = (rand() - RAND_MAX/2) * rand();
    b[i] = (rand() - RAND_MAX/2) * rand();
  }

  auto result = maximum(a, b);

  for (int i =0; i++; i < 32) {
    ASSERT_EQUAL(result[i], std::max(a[i], b[i]));
  }
}

} // namespace
