#include <torch/csrc/stable/c/python_shim.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/python_headers.h>

using torch::aot_inductor::new_tensor_handle;
using torch::aot_inductor::tensor_handle_to_tensor_pointer;

extern "C" {

AOTITorchError torch_tensor_from_pyobject(
    void* py_obj,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(py_obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(ret != nullptr, "ret must not be null");

    PyObject* obj = static_cast<PyObject*>(py_obj);
    TORCH_CHECK(
        THPVariable_Check(obj),
        "torch_tensor_from_pyobject: expected torch.Tensor, got ",
        Py_TYPE(obj)->tp_name);

    // THPVariable_Unpack returns a const reference to the at::Tensor stored
    // inside THPVariable::cdata. Copy-construct to bump the intrusive_ptr
    // refcount on the underlying TensorImpl, then heap-allocate via
    // new_tensor_handle so the resulting handle has independent ownership.
    *ret = new_tensor_handle(at::Tensor(THPVariable_Unpack(obj)));
  });
}

AOTITorchError torch_tensor_to_pyobject(
    AtenTensorHandle ath,
    void* py_type,
    void** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(ath != nullptr, "ath must not be null");
    TORCH_CHECK(ret != nullptr, "ret must not be null");

    at::Tensor* t = tensor_handle_to_tensor_pointer(ath);

    PyObject* py = (py_type != nullptr)
        ? THPVariable_Wrap(*t, static_cast<PyTypeObject*>(py_type))
        : THPVariable_Wrap(*t);
    if (py == nullptr) {
      // THPVariable_Wrap left the Python error indicator set; convert to a
      // C++ exception so AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE turns it
      // into AOTI_TORCH_FAILURE while the Python error remains set for the
      // caller to observe via PyErr_Occurred().
      throw python_error();
    }
    *ret = py;
  });
}

} // extern "C"
