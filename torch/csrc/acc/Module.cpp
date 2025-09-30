#include <torch/csrc/acc/Module.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch::acc {

// python hook interface
struct PythonHooks final : public at::PrivateUse1HooksInterface {
  using at::PrivateUse1HooksInterface::PrivateUse1HooksInterface;
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    PYBIND11_OVERRIDE_PURE_NAME(
        bool,
        at::PrivateUse1HooksInterface,
        "has_primary_context",
        hasPrimaryContext,
        device_index);
  }

  bool isBuilt() const override {
    PYBIND11_OVERRIDE_PURE_NAME(
        bool, at::PrivateUse1HooksInterface, "is_built", isBuilt, );
  }

  bool isAvailable() const override {
    PYBIND11_OVERRIDE_PURE_NAME(
        bool, at::PrivateUse1HooksInterface, "is_available", isBuilt, );
  }

  // TODO(qihqi): these is not supported from python yet
  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    return at::PrivateUse1HooksInterface::getDefaultGenerator(device_index);
  }

  at::Generator getNewGenerator(
      c10::DeviceIndex device_index = -1) const override {
    return at::PrivateUse1HooksInterface::getNewGenerator(device_index);
  }

  at::Device getDeviceFromPtr(void* data) const override {
    return at::PrivateUse1HooksInterface::getDeviceFromPtr(data);
  }

  bool isPinnedPtr(const void* data) const override {
    return at::PrivateUse1HooksInterface::isPinnedPtr(data);
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return at::PrivateUse1HooksInterface::getPinnedMemoryAllocator();
  }
};

struct PythonDeviceGuard final : public c10::impl::DeviceGuardImplInterface {
  using c10::impl::DeviceGuardImplInterface::DeviceGuardImplInterface;

  c10::DeviceType type() const override {
    PYBIND11_OVERRIDE_PURE_NAME(
        c10::DeviceType, c10::impl::DeviceGuardImplInterface, "type_", type, );
  }

  // TODO(qihqi): figure out if those are even useful
  // to python or not
  c10::Device exchangeDevice(c10::Device device) const override {
    return getDevice();
  }
  c10::Device getDevice() const override {
    return c10::Device(type(), 0);
  }
  void setDevice(c10::Device device) const override {}
  void uncheckedSetDevice(c10::Device device) const noexcept override {}
  c10::Stream getStream(c10::Device) const noexcept override {
    // no-op
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }

  c10::Stream getNewStream(c10::Device, int priority = 0) const override {
    // no-op
    (void)priority;
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }

  c10::Stream exchangeStream(c10::Stream) const noexcept override {
    // no-op
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // TODO(qihqi): support Event-related functions
  void record(
      void** /*event*/,
      const c10::Stream& /*stream*/,
      const c10::DeviceIndex /*device_index*/,
      const c10::EventFlag /*flag*/) const override {}
  void block(void* /*event*/, const c10::Stream& /*stream*/) const override {}
  bool queryEvent(void* /*event*/) const override {
    return true;
  }
  void destroyEvent(void* /*event*/, const c10::DeviceIndex /*device_index*/)
      const noexcept override {}

  // Stream-related functions
  bool queryStream(const c10::Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const c10::Stream& /*stream*/) const override {}
};

namespace {

bool registerPythonPrivateUse1Hook(const py::object& hook) {
  if (at::isPrivateUse1HooksRegistered()) {
    return false;
  }
  hook.inc_ref();
  at::RegisterPrivateUse1HooksInterface(
      hook.cast<PrivateUse1HooksInterface*>());
  return true;
}

bool registerPythonPrivateUse1DeviceGuard(const py::object& guard) {
  if (c10::impl::hasDeviceGuardImpl(c10::DeviceType::PrivateUse1)) {
    return false;
  }
  guard.inc_ref();
  c10::impl::registerDeviceGuard(
      c10::DeviceType::PrivateUse1,
      guard.cast<c10::impl::DeviceGuardImplInterface*>());
  return true;
}

at::Tensor createEmptyTensor(
    const std::vector<int64_t>& shape,
    c10::ScalarType dtype) {
  c10::Storage storage{
      c10::Storage::use_byte_size_t{},
      0,
      c10::GetAllocator(c10::kMeta),
      true,
  };

  c10::Device device(c10::DeviceType::PrivateUse1, 0);
  storage.set_data_ptr_noswap(at::DataPtr{nullptr, device});
  c10::DispatchKeySet key_set({c10::DispatchKey::PrivateUse1});
  at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), key_set, c10::scalarTypeToTypeMeta(dtype));

  std::vector<int64_t> strides(shape.size());
  int64_t size = 1;
  for (auto i = strides.size(); i > 0; --i) {
    strides[i - 1] = size;
    size *= shape[i - 1];
  }

  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(shape, strides, 0);
  return tensor;
}
} // namespace

void initModule(PyObject* module) {
  auto py_module = py::reinterpret_borrow<py::module>(module);
  auto _acc =
      py_module.def_submodule("_acc", "classes related to custom accelerators");

  py::class_<at::PrivateUse1HooksInterface, PythonHooks>(
      _acc.ptr(), "PrivateUse1Hooks")
      .def(py::init<>())
      .def(
          "has_primary_context",
          &at::PrivateUse1HooksInterface::hasPrimaryContext)
      .def("is_built", &at::PrivateUse1HooksInterface::isBuilt)
      .def("is_available", &at::PrivateUse1HooksInterface::isAvailable);

  py::class_<c10::impl::DeviceGuardImplInterface, PythonDeviceGuard>(
      _acc.ptr(), "DeviceGuard")
      .def(py::init<>())
      .def("type_", &c10::impl::DeviceGuardImplInterface::type);

  _acc.def(
      "register_python_privateuseone_hook", &registerPythonPrivateUse1Hook);
  _acc.def(
      "register_python_privateuseone_device_guard",
      &registerPythonPrivateUse1DeviceGuard);
  _acc.def("create_empty_tensor", &createEmptyTensor);
}

} // namespace torch::acc
