#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

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

void test_overflow() {
  auto s1 = Scalar(M_PI);
  CATCH_REQUIRE(s1.toFloat() == static_cast<float>(M_PI));
  s1.toHalf();

  s1 = Scalar(100000);
  CATCH_REQUIRE(s1.toFloat() == 100000.0);
  CATCH_REQUIRE(s1.toInt() == 100000);

  CATCH_REQUIRE_THROWS_AS(s1.toHalf(), std::domain_error);

  s1 = Scalar(NAN);
  CATCH_REQUIRE(std::isnan(s1.toFloat()));
  CATCH_REQUIRE_THROWS_AS(s1.toInt(), std::domain_error);

  s1 = Scalar(INFINITY);
  CATCH_REQUIRE(std::isinf(s1.toFloat()));
  CATCH_REQUIRE_THROWS_AS(s1.toInt(), std::domain_error);
}

CATCH_TEST_CASE( "scalar test", "[]" ) {

  manual_seed(123, at::kCPU);
  manual_seed(123, at::kCUDA);

  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;
  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " " << bar.toDouble() << " " << what.isIntegral() <<  "\n";
  Generator & gen = at::globalContext().defaultGenerator(at::kCPU);
  CATCH_REQUIRE_NOTHROW(gen.seed());
  auto && C = at::globalContext();
  if(at::hasCUDA()) {
    auto t2 = zeros({4,4}, at::kCUDA);
    cout << &t2 << "\n";
  }
  auto t = ones({4,4});

  auto wha2 = zeros({4,4}).add(t).sum();
  CATCH_REQUIRE( wha2.toCDouble() == 16.0 );

  CATCH_REQUIRE( t.sizes()[0] == 4 );
  CATCH_REQUIRE( t.sizes()[1] == 4 );
  CATCH_REQUIRE( t.strides()[0] == 4 );
  CATCH_REQUIRE( t.strides()[1] == 1 );

  Type & T = CPU(Float);
  Tensor x = randn({1,10}, T);
  Tensor prev_h = randn({1,20}, T);
  Tensor W_h = randn({20,20}, T);
  Tensor W_x = randn({20,10}, T);
  Tensor i2h = at::mm(W_x, x.t());
  Tensor h2h = at::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

  _CATCH_REQUIRE_THROWS(at::_local_scalar(Tensor{}));

  test_overflow();

  if(at::hasCUDA()) {
    auto r = CUDA(Float).copy(next_h);
    CATCH_REQUIRE(CPU(Float).copy(r).equal(next_h));
  }
  CATCH_REQUIRE_NOTHROW(randn({10,10,2}, T));

  // check Scalar.toTensor on Scalars backed by different data types
  CATCH_REQUIRE(scalar_to_tensor(bar).type().scalarType() == kDouble);
  CATCH_REQUIRE(scalar_to_tensor(what).type().scalarType() == kLong);
  CATCH_REQUIRE(scalar_to_tensor(ones({})._local_scalar()).type().scalarType() == kDouble);

  if (x.type().scalarType() != ScalarType::Half) {
    AT_DISPATCH_ALL_TYPES(x.type(), "foo", [&] {
      scalar_t s = 1;
      std::stringstream ss;
      CATCH_REQUIRE_NOTHROW(ss << "hello, dispatch" << x.type().toString() << s << "\n");
      auto data = (scalar_t*)x.data_ptr();
      (void)data;
    });
  }

  // test direct C-scalar type conversions
  {
    auto x = ones({1,2}, T);
    _CATCH_REQUIRE_THROWS(x.toCFloat());
  }
  auto float_one = ones({}, T);
  CATCH_REQUIRE(float_one.toCFloat() == 1);
  CATCH_REQUIRE(float_one.toCInt() == 1);
  CATCH_REQUIRE((float_one.toCHalf() == 1));
}
