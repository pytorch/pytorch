#include <torch/extension.h>

// Declare the function from cuda_dlink_extension.cu.
// output = a * b + c
void addmm_cuda(const float* a, const float* b, const float* c, float* output, int size);

at::Tensor addmm(at::Tensor a, at::Tensor b, at::Tensor c) {
  TORCH_CHECK(a.device().is_cuda(), "a is a cuda tensor");
  TORCH_CHECK(b.device().is_cuda(), "b is a cuda tensor");
  TORCH_CHECK(c.device().is_cuda(), "c is a cuda tensor");
  TORCH_CHECK(a.dtype() == at::kFloat, "a is a float tensor");
  TORCH_CHECK(b.dtype() == at::kFloat, "b is a float tensor");
  TORCH_CHECK(c.dtype() == at::kFloat, "c is a float tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b should have same size");
  TORCH_CHECK(a.sizes() == c.sizes(), "a and c should have same size");

  at::Tensor output = at::empty_like(a);
  addmm_cuda(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), output.data_ptr<float>(), a.numel());

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("addmm", &addmm, "a * b + c");
}
