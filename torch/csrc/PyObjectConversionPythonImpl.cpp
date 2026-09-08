// Concrete PyObjectConversionInterface, compiled into libtorch_python (it uses
// THPVariable_* and the CPython C API). Registered with libtorch at load time
// so the libtorch-only stable shims (torch_tensor_{from,to}_pyobject) can reach
// it through torch::detail::getPyObjectConversionImpl().

#include <torch/csrc/PyObjectConversion.h>

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>

namespace torch::detail {

namespace {

struct ConcretePyObjectConversion final : PyObjectConversionInterface {
  at::Tensor tensor_from_pyobject(PyObject* obj) const override {
    // The GIL guards the THPVariable access below; a boxed STABLE_TORCH_LIBRARY
    // kernel may run with the GIL released, so assert rather than race.
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPVariable_Check(obj),
        "torch_tensor_from_pyobject: expected torch.Tensor, got ",
        Py_TYPE(obj)->tp_name);
    return THPVariable_Unpack(obj);
  }

  PyObject* tensor_to_pyobject(const at::Tensor& t, PyObject* py_type)
      const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_to_pyobject requires the GIL to be held");
    // Guard before reinterpret_cast: THPVariable_Wrap calls type->tp_alloc, so
    // a non-type py_type would dereference garbage. THPVariable_Wrap itself
    // then checks it is actually a torch.Tensor subclass.
    TORCH_CHECK(
        py_type == nullptr || PyType_Check(py_type),
        "torch_tensor_to_pyobject: py_type must be a Python type object, got ",
        Py_TYPE(py_type)->tp_name);
    PyObject* py = (py_type != nullptr)
        ? THPVariable_Wrap(t, reinterpret_cast<PyTypeObject*>(py_type))
        : THPVariable_Wrap(t);
    // THPVariable_Wrap throws (a c10::Error, surfaced by the shim's error-code
    // macro) on failure rather than returning null; guard defensively so we
    // never hand back a null PyObject.
    TORCH_CHECK(
        py != nullptr,
        "torch_tensor_to_pyobject: THPVariable_Wrap returned null");
    return py;
  }
};

// Registered once when libtorch_python is loaded (static init), analogous to
// how ConcretePyInterpreterVTable installs itself.
struct RegisterPyObjectConversion {
  ConcretePyObjectConversion impl;
  RegisterPyObjectConversion() {
    setPyObjectConversionImpl(&impl);
  }
  ~RegisterPyObjectConversion() {
    // On libtorch_python teardown, reset g_impl to the libtorch-resident no-op
    // so a late conversion errors cleanly instead of using this destroyed impl
    // (cf. PyInterpreterHolder::~PyInterpreterHolder calling disarm()).
    setPyObjectConversionImpl(nullptr);
  }
};

const RegisterPyObjectConversion register_py_object_conversion;

} // namespace

} // namespace torch::detail
