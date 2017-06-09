#include <iostream>
#include "TensorLib/TensorLib.h"

using std::cout;
using namespace tlib;

constexpr auto Float = ScalarType::Float;
constexpr auto Double = ScalarType::Float;

int main() {
  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;

  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " " << bar.toDouble() << " " << what.isIntegral() <<  "\n";
  Generator & gen = tlib::globalContext()->defaultGenerator(Processor::CPU);
  cout << gen.seed() << "\n";
  auto C = tlib::globalContext();
  auto & CUDAFloat = C->getType(Processor::CPU,ScalarType::Float);
  auto t2 = CUDAFloat.zeros({4,4});
  cout << &t2 << "\n";
  cout << "AFTER GET TYPE " << &CUDAFloat << "\n";
  cout << "STORAGE: " << CUDAFloat.newStorage(4).get() << "\n";
  auto s = CUDAFloat.newStorage(4);
  s->fill(7);

  cout << "GET " << s->get(3).toFloat() << "\n";

  auto t = CUDA(Float).ones({4,4});

  auto wha2 = CUDA(Float).zeros({4,4}).add(t).sum();
  cout << wha2.toDouble() << " <-ndim\n";

  cout << t.sizes() << " " << t.strides() << "\n";


  Tensor x = tlib::randn({1,10});
  Tensor prev_h = tlib::randn({1,20});
  Tensor W_h = tlib::randn({20,20});
  Tensor W_x = tlib::randn({20,10});
  Tensor i2h = tlib::mm(W_x, x.t());
  Tensor h2h = tlib::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

}
