#include <torch/extension.h>

#include <ATen/core/op_registration/op_registration.h>

using namespace at;

static int test_int;

Tensor get_tensor(caffe2::TypeMeta dtype, IntArrayRef size) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          dtype, 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 0)), nullptr, false),
      DispatchKey::MSNPUTensorId);
  // This is a hack to workaround the shape checks in _convolution.
  tensor_impl->set_sizes_contiguous(size);
  return Tensor(std::move(tensor_impl));
}

Tensor empty_override(IntArrayRef size, const TensorOptions & options) {
  test_int = 0;
  return get_tensor(options.dtype(), size);
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 1;
  return get_tensor(a.dtype(), a.sizes());
}

Tensor fake_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  test_int = 2;
  // Only the first 2 dimension of output shape is correct.
  return get_tensor(input.dtype(), {input.size(0), weight.size(0), input.size(2), input.size(3)});
}

std::tuple<Tensor,Tensor,Tensor> fake_convolution_backward(
        const Tensor & grad_output, const Tensor & input, const Tensor & weight,
        IntArrayRef stride, IntArrayRef padding,
        IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
        int64_t groups, std::array<bool,3> output_mask) {
    test_int = 3;
    return std::tuple<Tensor, Tensor, Tensor>(
            get_tensor(input.dtype(), input.sizes()),
            get_tensor(weight.dtype(), weight.sizes()),
            get_tensor(input.dtype(), {}));
}

void init_msnpu_extension() {
  static auto registry = torch::import("aten")
    .impl("empty.memory_format",                torch::dispatch(DispatchKey::MSNPUTensorId, CppFunction::makeUnboxedOnly(empty_override)))
    .impl("add.Tensor",                         torch::dispatch(DispatchKey::MSNPUTensorId, CppFunction::makeUnboxedOnly(add_override)))
    .impl("convolution_overrideable",           torch::dispatch(DispatchKey::MSNPUTensorId, CppFunction::makeUnboxedOnly(fake_convolution)))
    .impl("convolution_backward_overrideable",  torch::dispatch(DispatchKey::MSNPUTensorId, CppFunction::makeUnboxedOnly(fake_convolution_backward)))
    ;
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

  // Event-related functions
  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(false, "MSNPU backend doesn't support events.");
  }
  void block(
    void* event,
    const Stream& stream) const override {
    TORCH_CHECK(false, "MSNPU backend doesn't support events.");
  }
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "MSNPU backend doesn't support events.");
  }
  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override { }
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
