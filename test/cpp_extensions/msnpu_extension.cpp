#include <torch/extension.h>

#include <ATen/ExtensionBackendRegistration.h>

using namespace at;

static int test_int;

Tensor get_dummy_tensor() {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 1)), nullptr, false),
      MSNPUTensorId(),
      false);
  return Tensor(std::move(tensor_impl));
}

Tensor zeros_override(IntArrayRef size, const TensorOptions & options) {
  test_int = 0;
  return get_dummy_tensor();
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 1;
  return get_dummy_tensor();
}

Tensor sum_override(const Tensor & self) {
  test_int = 2;
  return get_dummy_tensor();
}

// needed for sum backwards
Tensor expand_override(const Tensor & self, IntArrayRef size, bool implicit) {
  return get_dummy_tensor();
}


Tensor kl_div_override(
    const Tensor & self, const Tensor & target, int64_t reduction) {
  test_int = 3;
  return get_dummy_tensor();
}

Tensor kl_div_backward_override(
    const Tensor & grad_output,
    const Tensor & self,
    const Tensor & target,
    int64_t reduction) {
  test_int = 4;
  return get_dummy_tensor();
}

// numel and ones_like are needed for autograd backwards
int64_t numel_override(const Tensor & self) {
  return 1;
}

Tensor ones_like_override(const Tensor & self, const TensorOptions & options) {
  return get_dummy_tensor();
}

void init_msnpu_extension() {
  register_extension_backend_op(
    Backend::MSNPU,
    "zeros(IntArrayRef size, TensorOptions options) -> Tensor", &zeros_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "add(Tensor self, Tensor other, Scalar alpha) -> Tensor", &add_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "sum(Tensor self) -> Tensor", &sum_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "expand(Tensor self, IntArrayRef size, bool implicit) -> Tensor",
    &expand_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "kl_div(Tensor self, Tensor target, int64_t reduction) -> Tensor",
    &kl_div_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor",
    &kl_div_backward_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "numel(Tensor self) -> int64_t", &numel_override);
  register_extension_backend_op(
    Backend::MSNPU,
    "ones_like(Tensor self, TensorOptions options) -> Tensor",
    &ones_like_override);
}

int get_test_int() {
  return test_int;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_msnpu_extension", &init_msnpu_extension);
  m.def("get_test_int", &get_test_int);
}
