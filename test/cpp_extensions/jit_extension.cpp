#include <torch/torch.h>

using namespace at;

Tensor tanh_add(Tensor x, Tensor y) {
  return x.tanh() + y.tanh();
}

PYBIND11_MODULE(jit_extension, m) {
  m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
}
