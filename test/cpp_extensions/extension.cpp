#include <torch/torch.h>

using namespace at;

Tensor sigmoid_add(Tensor x, Tensor y) {
  return x.sigmoid() + y.sigmoid();
}

struct MatrixMultiplier {
  MatrixMultiplier(int A, int B) {
     tensor_ = CPU(kDouble).ones({A, B});
  }
  Tensor forward(Tensor weights) {
    return tensor_.mm(weights);
  }
  Tensor get() const {
    return tensor_;
  }

private:
  Tensor tensor_;
};

PYBIND11_MODULE(torch_test_cpp_extensions, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  py::class_<MatrixMultiplier>(m, "MatrixMultiplier")
      .def(py::init<int, int>())
      .def("forward", &MatrixMultiplier::forward)
      .def("get", &MatrixMultiplier::get);
}
