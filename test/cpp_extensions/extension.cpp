#include <torch/extension.h>

at::Tensor sigmoid_add(at::Tensor x, at::Tensor y) {
  return x.sigmoid() + y.sigmoid();
}

struct MatrixMultiplier {
  MatrixMultiplier(int A, int B) {
    tensor_ = at::ones({A, B}, torch::CPU(at::kDouble));
    torch::set_requires_grad(tensor_, true);
  }
  at::Tensor forward(at::Tensor weights) {
    return tensor_.mm(weights);
  }
  at::Tensor get() const {
    return tensor_;
  }

 private:
  at::Tensor tensor_;
};

bool function_taking_optional(c10::optional<at::Tensor> tensor) {
  return tensor.has_value();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  m.def(
      "function_taking_optional",
      &function_taking_optional,
      "function_taking_optional");
  py::class_<MatrixMultiplier>(m, "MatrixMultiplier")
      .def(py::init<int, int>())
      .def("forward", &MatrixMultiplier::forward)
      .def("get", &MatrixMultiplier::get);
}
