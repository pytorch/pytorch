#include <iostream>
#include "TensorLib.h"

using std::cout;
using namespace tlib;

int main() {
  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;

  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " " << bar.toDouble() << " " << what.isIntegral() <<  "\n";
  CUDAGenerator gen(tlib::globalContext());
  cout << gen.seed() << "\n";
  auto C = tlib::globalContext();
  auto & CUDAFloat = C->getType(Processor::CUDA,ScalarType::Float);
  cout << "AFTER GET TYPE " << &CUDAFloat << "\n";
  cout << "STORAGE: " << CUDAFloat.newStorage(4) << "\n";
  std::unique_ptr<Storage> s(CUDAFloat.newStorage(4));
  s->fill(7);

  cout << "GET " << s->get(3).toFloat() << "\n";

}
