#include <torch/csrc/PyObjectConversion.h>

#include <c10/util/Exception.h>

namespace torch::detail {

namespace {

// Error message shared by both no-op entry points.
constexpr const char* kNoImplMsg =
    "Converting between a Python object and a torch::stable::Tensor requires "
    "libtorch_python to be loaded (e.g. `import torch` in the running "
    "process). This process linked only libtorch.";

// Default implementation used until libtorch_python registers the real one.
// Mirrors NoopPyInterpreterVTable: calling a method is a hard error rather than
// silent misbehavior.
struct NoopPyObjectConversion final : PyObjectConversionInterface {
  at::Tensor tensor_from_pyobject(PyObject* /*obj*/) const override {
    TORCH_CHECK(false, kNoImplMsg);
  }
  PyObject* tensor_to_pyobject(const at::Tensor& /*t*/, PyObject* /*py_type*/)
      const override {
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
