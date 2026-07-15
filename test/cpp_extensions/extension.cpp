#include <torch/extension.h>

// test include_dirs in setuptools.setup with relative path
#include <tmp.h>
#include <ATen/OpMathType.h>

torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  return x.sigmoid() + y.sigmoid();
}

struct MatrixMultiplier {
  MatrixMultiplier(int A, int B) {
    tensor_ =
        torch::ones({A, B}, torch::dtype(torch::kFloat64).requires_grad(true));
  }
  torch::Tensor forward(torch::Tensor weights) {
    return tensor_.mm(weights);
  }
  torch::Tensor get() const {
    return tensor_;
  }

 private:
  torch::Tensor tensor_;
};

bool function_taking_optional(std::optional<torch::Tensor> tensor) {
  return tensor.has_value();
}

torch::Tensor random_tensor() {
  return torch::randn({1});
}

at::ScalarType get_math_type(at::ScalarType other) {
  return at::toOpMathType(other);
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

  m.def("get_complex", []() { return c10::complex<double>(1.0, 2.0); });
  m.def("get_device", []() { return at::device_of(random_tensor()).value(); });
  m.def("get_generator", []() { return at::detail::getDefaultCPUGenerator(); });
  m.def("get_intarrayref", []() { return at::IntArrayRef({1, 2, 3}); });
  m.def("get_memory_format", []() { return c10::get_contiguous_memory_format(); });
  m.def("get_storage", []() { return random_tensor().storage(); });
  m.def("get_symfloat", []() { return c10::SymFloat(1.0); });
  m.def("get_symint", []() { return c10::SymInt(1); });
  m.def("get_symintarrayref", []() { return at::SymIntArrayRef({1, 2, 3}); });
  m.def("get_tensor", []() { return random_tensor(); });
  m.def("get_math_type", &get_math_type);
}
