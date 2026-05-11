#ifndef STABLE_TORCH_PYTHON_SHIM
#define STABLE_TORCH_PYTHON_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/version.h>

// Stable C entry points for converting between a Python torch.Tensor
// (PyObject*) and an AtenTensorHandle. PyObject* values cross the ABI as
// opaque void* so this header does not depend on Python.h. The
// implementation lives in libtorch_python.so since it uses THPVariable_*,
// so consumers must also link against libtorch_python. The GIL must be held
// by the caller.

#ifdef __cplusplus
extern "C" {
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

// Wrap a Python torch.Tensor as a new AtenTensorHandle that shares the
// underlying TensorImpl with the input.
AOTI_TORCH_EXPORT AOTITorchError torch_tensor_from_pyobject(
    void* py_obj,
    AtenTensorHandle* ret); // returns new reference

// Wrap an AtenTensorHandle as a Python torch.Tensor. If py_type is
// non-null, it is used as the result's exact PyTypeObject* (e.g.
// torch.nn.Parameter); nullptr means the default torch.Tensor type. On
// failure the Python error indicator is left set.
AOTI_TORCH_EXPORT AOTITorchError torch_tensor_to_pyobject(
    AtenTensorHandle ath,
    void* py_type,
    void** ret); // returns new reference

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_PYTHON_SHIM
