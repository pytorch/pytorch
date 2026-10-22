#include <torch/csrc/PyObjectConversion.h>

#include <c10/util/Exception.h>

namespace torch::detail {

namespace {

// Error message shared by all no-op entry points.
constexpr const char* kNoImplMsg =
    "Using APIs relating to conversion between a Python object and its "
    "torch::stable equivalent requires libtorch_python to be loaded (e.g. "
    "`import torch` in the running process). This process linked only "
    "libtorch.";

// Default implementation used until libtorch_python registers the real one.
// Mirrors NoopPyInterpreterVTable: calling a method is a hard error rather than
// silent misbehavior.
struct NoopPyObjectConversion final : PyObjectConversionInterface {
  bool is_tensor_pyobject(PyObject* /*obj*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  at::Tensor tensor_from_pyobject(PyObject* /*obj*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  PyObject* tensor_to_pyobject(const at::Tensor& /*t*/, PyObject* /*py_type*/)
      const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  at::ScalarType dtype_from_pyobject(PyObject* /*obj*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  PyObject* dtype_to_pyobject(at::ScalarType /*dtype*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  at::Device device_from_pyobject(PyObject* /*obj*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  PyObject* device_to_pyobject(const at::Device& /*device*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
};

const NoopPyObjectConversion noop_impl;
const PyObjectConversionInterface* g_impl = &noop_impl;

} // namespace

void setPyObjectConversionImpl(const PyObjectConversionInterface* impl) {
  g_impl = (impl != nullptr) ? impl : &noop_impl;
}

const PyObjectConversionInterface& getPyObjectConversionImpl() {
  return *g_impl;
}

} // namespace torch::detail
