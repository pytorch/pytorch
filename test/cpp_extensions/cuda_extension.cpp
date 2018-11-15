#include <torch/extension.h>

// Declare the function from cuda_extension.cu. It will be compiled
// separately with nvcc and linked with the object file of cuda_extension.cpp
// into one shared library.
void sigmoid_add_cuda(const float* x, const float* y, float* output, int size);

torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  AT_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
  AT_CHECK(y.type().is_cuda(), "y must be a CUDA tensor");
  auto output = torch::zeros_like(x);
  sigmoid_add_cuda(
      x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
}
