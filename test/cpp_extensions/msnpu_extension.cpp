#include <torch/extension.h>

#include <ATen/ExtensionBackendRegistration.h>

using namespace at;

static int test_int;

Tensor get_dtype_tensor(caffe2::TypeMeta dtype) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          dtype, 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 0)), nullptr, false),
      MSNPUTensorId());
  return Tensor(std::move(tensor_impl));
}

Tensor zeros_override(IntArrayRef size, const TensorOptions & options) {
  test_int = 0;
  return get_dtype_tensor(options.dtype());
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 1;
  return get_dtype_tensor(a.dtype());
}

Tensor sum_override(const Tensor & self) {
  test_int = 2;
  return get_dtype_tensor(self.dtype());
}

// needed for sum backwards
Tensor expand_override(const Tensor & self, IntArrayRef size, bool implicit) {
  return get_dtype_tensor(self.dtype());
}


Tensor kl_div_override(
    const Tensor & self, const Tensor & target, int64_t reduction) {
  test_int = 3;
  return get_dtype_tensor(self.dtype());
}

Tensor kl_div_backward_override(
    const Tensor & grad_output,
    const Tensor & self,
    const Tensor & target,
    int64_t reduction) {
  test_int = 4;
  return get_dtype_tensor(self.dtype());
}

// numel and ones_like are needed for autograd backwards
int64_t numel_override(const Tensor & self) {
  return 1;
}

Tensor ones_like_override(const Tensor & self, const TensorOptions & options) {
  return get_dtype_tensor(options.dtype());
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

// TODO: Extend this to exercise multi-device setting.  In that case,
// we need to add a thread local variable to track the current device.
struct MSNPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::MSNPU;
  MSNPUGuardImpl() {}
  MSNPUGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::MSNPU);
  }
  DeviceType type() const override {
    return DeviceType::MSNPU;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MSNPU);
    AT_ASSERT(d.index() == 0);
    return d;
  }
  Device getDevice() const override {
    return Device(DeviceType::MSNPU, 0);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MSNPU);
    AT_ASSERT(d.index() == 0);
  }
  void uncheckedSetDevice(Device d) const noexcept override {
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MSNPU, 0));
  }
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MSNPU, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
};

constexpr DeviceType MSNPUGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL(MSNPU, MSNPUGuardImpl);

int get_test_int() {
  return test_int;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_msnpu_extension", &init_msnpu_extension);
  m.def("get_test_int", &get_test_int);
}
