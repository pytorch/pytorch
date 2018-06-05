#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "test_seed.h"

using std::cout;
using namespace at;

constexpr auto Float = ScalarType::Float;

template<typename scalar_type>
struct Foo {
  static void apply(Tensor a, Tensor b) {
    scalar_type s = 1;
    std::stringstream ss;
    ss << "hello, dispatch: " << a.type().toString() << s << "\n";
    auto data = (scalar_type*)a.data_ptr();
    (void)data;
  }
};
template<>
struct Foo<Half> {
  static void apply(Tensor a, Tensor b) {}
};

void test_ctors() {
  // create scalars backed by tensors
  auto s1 = Scalar(CPU(kFloat).scalarTensor(1));
  auto s2 = Scalar(CPU(kFloat).scalarTensor(2));
  Scalar{s1};
  Scalar{std::move(s2)};
  REQUIRE(s2.isBackedByTensor());
  REQUIRE(!s2.toTensor().defined());
  s2 = s1;
  REQUIRE(s2.isBackedByTensor());
  REQUIRE(s2.toFloat() == 1.0);
  Scalar s3;
  s3 = std::move(s2);
  REQUIRE(s2.isBackedByTensor());
  REQUIRE(!s2.toTensor().defined());
  REQUIRE(s3.isBackedByTensor());
  REQUIRE(s3.toFloat() == 1.0);
}

void test_overflow() {
  auto s1 = Scalar(M_PI);
  REQUIRE(s1.toFloat() == static_cast<float>(M_PI));
  s1.toHalf();

  s1 = Scalar(100000);
  REQUIRE(s1.toFloat() == 100000.0);
  REQUIRE(s1.toInt() == 100000);

  REQUIRE_THROWS_AS(s1.toHalf(), std::domain_error);

  s1 = Scalar(NAN);
  REQUIRE(std::isnan(s1.toFloat()));
  REQUIRE_THROWS_AS(s1.toInt(), std::domain_error);

  s1 = Scalar(INFINITY);
  REQUIRE(std::isinf(s1.toFloat()));
  REQUIRE_THROWS_AS(s1.toInt(), std::domain_error);
}

TEST_CASE( "scalar test", "[]" ) {

  manual_seed(123, at::Backend::CPU);
  manual_seed(123, at::Backend::CUDA);

  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;
  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " " << bar.toDouble() << " " << what.isIntegral() <<  "\n";
  Generator & gen = at::globalContext().defaultGenerator(Backend::CPU);
  REQUIRE_NOTHROW(gen.seed());
  auto && C = at::globalContext();
  if(at::hasCUDA()) {
    auto & CUDAFloat = C.getType(Backend::CUDA,ScalarType::Float);
    auto t2 = zeros(CUDAFloat, {4,4});
    cout << &t2 << "\n";
    cout << "AFTER GET TYPE " << &CUDAFloat << "\n";
    auto s = CUDAFloat.storage(4);
    REQUIRE( s->get(3).toFloat() == 0.0 );
    s->fill(7);
    REQUIRE( s->get(3).toFloat() == 7.0 );
  }
  auto t = ones(CPU(Float), {4,4});

  auto wha2 = zeros(CPU(Float), {4,4}).add(t).sum();
  REQUIRE( wha2.toCDouble() == 16.0 );

  REQUIRE( t.sizes()[0] == 4 );
  REQUIRE( t.sizes()[1] == 4 );
  REQUIRE( t.strides()[0] == 4 );
  REQUIRE( t.strides()[1] == 1 );

  Type & T = CPU(Float);
  Tensor x = randn(T, {1,10});
  Tensor prev_h = randn(T, {1,20});
  Tensor W_h = randn(T, {20,20});
  Tensor W_x = randn(T, {20,10});
  Tensor i2h = at::mm(W_x, x.t());
  Tensor h2h = at::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

  REQUIRE_THROWS(Scalar{Tensor{}});

  test_ctors();
  test_overflow();

  if(at::hasCUDA()) {
    auto r = CUDA(Float).copy(next_h);
    REQUIRE(CPU(Float).copy(r).equal(next_h));
  }
  REQUIRE_NOTHROW(randn(T, {10,10,2}));

  // check Scalar.toTensor on Scalars backed by different data types
  REQUIRE(bar.toTensor().type().scalarType() == kDouble);
  REQUIRE(what.toTensor().type().scalarType() == kLong);
  REQUIRE(Scalar(ones(CPU(kFloat), {})).toTensor().type().scalarType() == kFloat);

  if (x.type().scalarType() != ScalarType::Half) {
    AT_DISPATCH_ALL_TYPES(x.type(), "foo", [&] {
      scalar_t s = 1;
      std::stringstream ss;
      REQUIRE_NOTHROW(ss << "hello, dispatch" << x.type().toString() << s << "\n");
      auto data = (scalar_t*)x.data_ptr();
      (void)data;
    });
  }

  // test direct C-scalar type conversions
  {
    auto x = ones(T, {1,2});
    REQUIRE_THROWS(x.toCFloat());
  }
  auto float_one = ones(T, {});
  REQUIRE(float_one.toCFloat() == 1);
  REQUIRE(float_one.toCInt() == 1);
  REQUIRE((float_one.toCHalf() == 1));
}
