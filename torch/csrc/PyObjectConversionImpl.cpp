// Concrete PyObjectConversionInterface, compiled into libtorch_python (it uses
// THPVariable_* and the CPython C API). Registered with libtorch at load time
// so the libtorch-only stable shims (torch_tensor_{from,to}_pyobject) can reach
// it through torch::detail::getPyObjectConversionImpl().

#include <torch/csrc/PyObjectConversion.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/python_headers.h>

using torch::aot_inductor::new_tensor_handle;
using torch::aot_inductor::tensor_handle_to_tensor_pointer;

namespace torch::detail {

namespace {

struct ConcretePyObjectConversion final : PyObjectConversionInterface {
  AtenTensorHandle from_pyobject(PyObject* obj) const override {
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
    return new_tensor_handle(at::Tensor(THPVariable_Unpack(obj)));
  }

  PyObject* to_pyobject(AtenTensorHandle ath, PyObject* py_type)
      const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_to_pyobject requires the GIL to be held");
    at::Tensor* t = tensor_handle_to_tensor_pointer(ath);
    PyObject* py = (py_type != nullptr)
        ? THPVariable_Wrap(*t, reinterpret_cast<PyTypeObject*>(py_type))
        : THPVariable_Wrap(*t);
    if (py == nullptr) {
      // Forward the Python error left set by THPVariable_Wrap.
      throw python_error();
    }
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
};

const RegisterPyObjectConversion register_py_object_conversion;

} // namespace

} // namespace torch::detail
