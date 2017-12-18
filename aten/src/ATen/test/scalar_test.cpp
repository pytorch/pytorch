#include <iostream>
// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "test_assert.h"

using std::cout;
using namespace at;

constexpr auto Float = ScalarType::Float;

template<typename scalar_type>
struct Foo {
  static void apply(Tensor a, Tensor b) {
    scalar_type s = 1;
    cout << "hello, dispatch: " << a.type().toString() << s << "\n";
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
  ASSERT(s2.isBackedByTensor() && !s2.toTensor().defined());
  s2 = s1;
  ASSERT(s2.isBackedByTensor() && s2.toFloat() == 1.0);
  Scalar s3;
  s3 = std::move(s2);
  ASSERT(s2.isBackedByTensor() && !s2.toTensor().defined());
  ASSERT(s3.isBackedByTensor() && s3.toFloat() == 1.0);
}

void test_overflow() {
  auto s1 = Scalar(M_PI);
  ASSERT(s1.toFloat() == static_cast<float>(M_PI));
  s1.toHalf();

  s1 = Scalar(100000);
  ASSERT(s1.toFloat() == 100000.0);
  ASSERT(s1.toInt() == 100000);

  bool threw = false;
  try {
    s1.toHalf();
  } catch (std::domain_error& e) {
    threw = true;
  }
  ASSERT(threw);

  s1 = Scalar(NAN);
  ASSERT(std::isnan(s1.toFloat()));
  threw = false;
  try {
    s1.toInt();
  } catch (std::domain_error& e) {
    threw = true;
  }
  ASSERT(threw);

  s1 = Scalar(INFINITY);
  ASSERT(std::isinf(s1.toFloat()));
  threw = false;
  try {
    s1.toInt();
  } catch (std::domain_error& e) {
    threw = true;
  }
  ASSERT(threw);
}

int main() {
  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;
  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " " << bar.toDouble() << " " << what.isIntegral() <<  "\n";
  Generator & gen = at::globalContext().defaultGenerator(Backend::CPU);
  cout << gen.seed() << "\n";
  auto && C = at::globalContext();
  if(at::hasCUDA()) {
    auto & CUDAFloat = C.getType(Backend::CPU,ScalarType::Float);
    auto t2 = CUDAFloat.zeros({4,4});
    cout << &t2 << "\n";
    cout << "AFTER GET TYPE " << &CUDAFloat << "\n";
    cout << "STORAGE: " << CUDAFloat.storage(4).get() << "\n";
    auto s = CUDAFloat.storage(4);
    s->fill(7);
    cout << "GET " << s->get(3).toFloat() << "\n";
  }
  auto t = CPU(Float).ones({4,4});

  auto wha2 = CPU(Float).zeros({4,4}).add(t).sum();
  cout << wha2.toCDouble() << " <-ndim\n";

  cout << t.sizes() << " " << t.strides() << "\n";

  Type & T = CPU(Float);
  Tensor x = T.randn({1,10});
  Tensor prev_h = T.randn({1,20});
  Tensor W_h = T.randn({20,20});
  Tensor W_x = T.randn({20,10});
  Tensor i2h = at::mm(W_x, x.t());
  Tensor h2h = at::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

  bool threw = false;
  try {
    Scalar{Tensor{}};
  } catch (std::runtime_error& e) {
    threw = true;
  }
  ASSERT(threw);

  test_ctors();
  test_overflow();

  if(at::hasCUDA()) {
    auto r = CUDA(Float).copy(next_h);

    cout << r << "\n";
  }
  cout << T.randn({10,10,2}) << "\n";

  // check Scalar.toTensor on Scalars backed by different data types
  ASSERT(bar.toTensor().type().scalarType() == kDouble);
  ASSERT(what.toTensor().type().scalarType() == kLong);
  ASSERT(Scalar(CPU(kFloat).ones({})).toTensor().type().scalarType() == kFloat);

  dispatch_all<Foo>(x.type(),"foo",x,prev_h);

  // test direct C-scalar type conversions
  try {
    auto x = T.ones({1,2});
    x.toCFloat();
    ASSERT(false);
  } catch (std::runtime_error &e) {}
  auto float_one = T.ones({});
  ASSERT(float_one.toCFloat() == 1);
  ASSERT(float_one.toCInt() == 1);
  ASSERT(float_one.toCHalf() == 1);

  return 0;

}
