#include <gtest/gtest.h>

#include <iostream>
#include <random>
// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

using std::cout;
using namespace at;

constexpr auto Float = ScalarType::Float;

template<typename scalar_type>
struct Foo {
  static void apply(Tensor a, Tensor b) {
    scalar_type s = 1;
    std::stringstream ss;
    ss << "hello, dispatch: " << a.toString() << s << "\n";
    auto data = (scalar_type*)a.data_ptr();
    (void)data;
  }
};
template<>
struct Foo<Half> {
  static void apply(Tensor a, Tensor b) {}
};

void test_overflow() {
  auto s1 = Scalar(M_PI);
  ASSERT_EQ(s1.toFloat(), static_cast<float>(M_PI));
  s1.toHalf();

  s1 = Scalar(100000);
  ASSERT_EQ(s1.toFloat(), 100000.0);
  ASSERT_EQ(s1.toInt(), 100000);

  ASSERT_THROW(s1.toHalf(), std::runtime_error);

  s1 = Scalar(NAN);
  ASSERT_TRUE(std::isnan(s1.toFloat()));
  ASSERT_THROW(s1.toInt(), std::runtime_error);

  s1 = Scalar(INFINITY);
  ASSERT_TRUE(std::isinf(s1.toFloat()));
  ASSERT_THROW(s1.toInt(), std::runtime_error);
}

TEST(TestScalar, TestScalar) {
  manual_seed(123);

  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;
  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " "
       << bar.toDouble() << " " << what.isIntegral(false) << "\n";
  auto gen = at::detail::getDefaultCPUGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    ASSERT_NO_THROW(gen.set_current_seed(std::random_device()()));
  }
  auto&& C = at::globalContext();
  if (at::hasCUDA()) {
    auto t2 = zeros({4, 4}, at::kCUDA);
    cout << &t2 << "\n";
  }
  auto t = ones({4, 4});

  auto wha2 = zeros({4, 4}).add(t).sum();
  ASSERT_EQ(wha2.item<double>(), 16.0);

  ASSERT_EQ(t.sizes()[0], 4);
  ASSERT_EQ(t.sizes()[1], 4);
  ASSERT_EQ(t.strides()[0], 4);
  ASSERT_EQ(t.strides()[1], 1);

  TensorOptions options = dtype(kFloat);
  Tensor x = randn({1, 10}, options);
  Tensor prev_h = randn({1, 20}, options);
  Tensor W_h = randn({20, 20}, options);
  Tensor W_x = randn({20, 10}, options);
  Tensor i2h = at::mm(W_x, x.t());
  Tensor h2h = at::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

  ASSERT_ANY_THROW(Tensor{}.item());

  test_overflow();

  if (at::hasCUDA()) {
    auto r = next_h.to(at::Device(kCUDA), kFloat, /*non_blocking=*/ false, /*copy=*/ true);
    ASSERT_TRUE(r.to(at::Device(kCPU), kFloat, /*non_blocking=*/ false, /*copy=*/ true).equal(next_h));
  }
  ASSERT_NO_THROW(randn({10, 10, 2}, options));

  // check Scalar.toTensor on Scalars backed by different data types
  ASSERT_EQ(scalar_to_tensor(bar).scalar_type(), kDouble);
  ASSERT_EQ(scalar_to_tensor(what).scalar_type(), kLong);
  ASSERT_EQ(scalar_to_tensor(ones({}).item()).scalar_type(), kDouble);

  if (x.scalar_type() != ScalarType::Half) {
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "foo", [&] {
      scalar_t s = 1;
      std::stringstream ss;
      ASSERT_NO_THROW(
          ss << "hello, dispatch" << x.toString() << s << "\n");
      auto data = (scalar_t*)x.data_ptr();
      (void)data;
    });
  }

  // test direct C-scalar type conversions
  {
    auto x = ones({1, 2}, options);
    ASSERT_ANY_THROW(x.item<float>());
  }
  auto float_one = ones({}, options);
  ASSERT_EQ(float_one.item<float>(), 1);
  ASSERT_EQ(float_one.item<int32_t>(), 1);
  ASSERT_EQ(float_one.item<at::Half>(), 1);
}

TEST(TestScalar, TestConj) {
  Scalar int_scalar = 257;
  Scalar float_scalar = 3.0;
  Scalar complex_scalar = c10::complex<double>(2.3, 3.5);

  ASSERT_EQ(int_scalar.conj().toInt(), 257);
  ASSERT_EQ(float_scalar.conj().toDouble(), 3.0);
  ASSERT_EQ(complex_scalar.conj().toComplexDouble(), c10::complex<double>(2.3, -3.5));
}

TEST(TestScalar, TestEqual) {
  ASSERT_FALSE(Scalar(1.0).equal(false));
  ASSERT_FALSE(Scalar(1.0).equal(true));
  ASSERT_FALSE(Scalar(true).equal(1.0));
  ASSERT_TRUE(Scalar(true).equal(true));

  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 5.0}).equal(c10::complex<double>{2.0, 5.0}));
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 0}).equal(2.0));
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 0}).equal(2));

  ASSERT_TRUE(Scalar(2.0).equal(c10::complex<double>{2.0, 0.0}));
  ASSERT_FALSE(Scalar(2.0).equal(c10::complex<double>{2.0, 4.0}));
  ASSERT_FALSE(Scalar(2.0).equal(3.0));
  ASSERT_TRUE(Scalar(2.0).equal(2));

  ASSERT_TRUE(Scalar(2).equal(c10::complex<double>{2.0, 0}));
  ASSERT_TRUE(Scalar(2).equal(2));
  ASSERT_TRUE(Scalar(2).equal(2.0));
}
