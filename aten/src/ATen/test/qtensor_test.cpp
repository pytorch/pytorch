#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <ATen/test/test_assert.h>

using namespace at;

TEST(TestQTensor, First) {
  Tensor r = at::ones({2});
  Tensor qr = r.to_quantize();
  Tensor rqr = qr.to_real();
  auto rqr_a = rqr.accessor<float, 1>();
  auto r_a = r.accessor<float, 1>();
  for (auto i = 0; i < 2; ++i) {
    ASSERT_EQ(r_a[i], rqr_a[i]);
  }
}
