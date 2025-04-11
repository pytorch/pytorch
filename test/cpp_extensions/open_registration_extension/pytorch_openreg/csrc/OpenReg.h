#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/utils/pybind.h>

namespace openreg {

using openreg_ptr_t = uint64_t;

void set_impl_factory(PyObject* factory);
py::function get_method(const char* name);

static constexpr char kFreeMethod[] = "free";
static constexpr char kHostFreeMethod[] = "hostFree";

template <const char* name>
static void ReportAndDelete(void* ptr) {
  if (!ptr || !Py_IsInitialized()) {
    return;
  }

  py::gil_scoped_acquire acquire;

  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  // Always stash, this will be a no-op if there is no error
  PyErr_Fetch(&type, &value, &traceback);

  TORCH_CHECK(
      get_method(name)(reinterpret_cast<openreg_ptr_t>(ptr)).cast<bool>(),
      "Failed to free memory pointer at ",
      ptr);

  // If that user code raised an error, just print it without raising it
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  // Restore the original error
  PyErr_Restore(type, value, traceback);
}

struct HostAllocator final : at::Allocator {
  HostAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("hostMalloc")(nbytes).cast<openreg_ptr_t>());
      TORCH_CHECK(data, "Failed to allocator ", nbytes, " bytes on host.");
    }
    return {data, data, &ReportAndDelete<kHostFreeMethod>, at::Device(at::kCPU)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kHostFreeMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("hostCopyData")(
        reinterpret_cast<openreg_ptr_t>(dest),
        reinterpret_cast<openreg_ptr_t>(src),
        count);
  }
};

struct OpenRegAllocator final : at::Allocator {
  OpenRegAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("malloc")(nbytes).cast<openreg_ptr_t>());
      TORCH_CHECK(
          data, "Failed to allocator ", nbytes, " bytes on openreg device.");
    }
    return {data, data, &ReportAndDelete<kFreeMethod>, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kFreeMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("copy_data")(
        reinterpret_cast<openreg_ptr_t>(dest),
        reinterpret_cast<openreg_ptr_t>(src),
        count);
  }
};
} // namespace openreg
