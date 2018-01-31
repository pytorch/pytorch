#include <torch/torch.h>

#include "doubler.h"

using namespace at;

Tensor exp_add(Tensor x, Tensor y) {
  return x.exp() + y.sigmoid();
}

PYBIND11_MODULE(torch_test_cpp_extensions, m) {
  m.def("exp_add", &exp_add, "exp(x) + exp(y)");
  py::class_<Doubler>(m, "Doubler")
      .def(py::init<int, int>())
      .def("forward", &Doubler::forward);
}
