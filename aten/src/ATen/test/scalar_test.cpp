#include <iostream>
#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "test_assert.h"

using std::cout;
using namespace at;

constexpr auto Float = ScalarType::Float;

template<typename scalar_type>
struct Foo {
  static void CPU(const Type & t, Tensor a, Tensor b) {
    scalar_type s = 1;
    cout << "hello, dispatch: " << t.toString() << s << "\n";
    auto data = (scalar_type*)a.data_ptr();
    (void)data;
  }
  static void CUDA(const Type & t, Tensor a, Tensor b) {
  }
};
template<>
struct Foo<Half> {
  static void CPU(const Type & t, Tensor a, Tensor b) {}
  static void CUDA(const Type & t, Tensor a, Tensor b) {}
};

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
  cout << wha2.toDouble() << " <-ndim\n";

  cout << t.sizes() << " " << t.strides() << "\n";

  auto output = CPU(Float).ones(3);
  at::Abs_updateOutput(t,output);
  Type & T = CPU(Float);
  Tensor x = T.randn({1,10});
  Tensor prev_h = T.randn({1,20});
  Tensor W_h = T.randn({20,20});
  Tensor W_x = T.randn({20,10});
  Tensor i2h = at::mm(W_x, x.t());
  Tensor h2h = at::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

  if(at::hasCUDA()) {
    auto r = CUDA(Float).copy(next_h);

    cout << r << "\n";
  }
  cout << T.randn({10,10,2}) << "\n";

  // check Scalar.toTensor on Scalars backed by different data types
  ASSERT(bar.toTensor().type().scalarType() == kDouble);
  ASSERT(what.toTensor().type().scalarType() == kLong);
  ASSERT(Scalar(CPU(kFloat).ones({})).toTensor().type().scalarType() == kFloat);

  dispatch<Foo>(x.type(),x,prev_h);
  return 0;

}
