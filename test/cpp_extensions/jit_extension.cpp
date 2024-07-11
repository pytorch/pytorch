#include <torch/extension.h>

#include "doubler.h"

using namespace at;

Tensor exp_add(Tensor x, Tensor y);

Tensor tanh_add(Tensor x, Tensor y) {
  return x.tanh() + y.tanh();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
  m.def("exp_add", &exp_add, "exp(x) + exp(y)");
  py::class_<Doubler>(m, "Doubler")
  .def(py::init<int, int>())
  .def("forward", &Doubler::forward)
  .def("get", &Doubler::get);
}
