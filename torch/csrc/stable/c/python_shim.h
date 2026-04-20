#ifndef STABLE_TORCH_PYTHON_SHIM
#define STABLE_TORCH_PYTHON_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/version.h>

// This header declares ABI-stable C entry points for converting between a
// Python torch.Tensor (PyObject*) and an AtenTensorHandle. PyObject* values
// cross the boundary as opaque void* so this header does not depend on
// Python.h (and consequently neither does any extension that includes it
// transitively via torch/csrc/stable headers).
//
// Implementation lives in torch/csrc/stable/python_shim.cpp and is linked
// into libtorch_python.so (NOT libtorch.so), because it needs THPVariable_*
// symbols. Extensions using these helpers must therefore also link against
// libtorch_python.

#ifdef __cplusplus
extern "C" {
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

// Convert a Python torch.Tensor (PyObject*) into a new owning
// AtenTensorHandle. The returned handle shares the underlying TensorImpl
// with the input Python tensor (refcount on TensorImpl is incremented).
//
// Args:
//   py_obj: borrowed reference to a torch.Tensor (or a subclass thereof),
//           passed as void* so this header is free of Python.h.
//   ret:    output. On success, set to a new AtenTensorHandle that the
//           caller is responsible for releasing via
//           aoti_torch_delete_tensor_object.
//
// Returns AOTI_TORCH_FAILURE if py_obj is null, is not a torch.Tensor, or
// any underlying call throws.
//
// GIL: must be held by the caller.
AOTI_TORCH_EXPORT AOTITorchError torch_tensor_from_pyobject(
    void* py_obj,
    AtenTensorHandle* ret);

// Convert an AtenTensorHandle to a Python torch.Tensor (new reference).
// The returned PyObject* shares storage with the input tensor.
//
// Args:
//   ath:     borrowed AtenTensorHandle.
//   py_type: optional PyTypeObject* (passed as void*) controlling the
//            exact Python type the result should have. Used to preserve
//            subclasses such as torch.nn.Parameter. Pass nullptr to wrap
//            as the default torch.Tensor type.
//   ret:     output. On success, set to a new reference PyObject*. The
//            caller is responsible for Py_DECREF (or wrapping in a smart
//            handle from their binding framework).
//
// Returns AOTI_TORCH_FAILURE on null inputs or if THPVariable_Wrap fails;
// in the latter case the Python error indicator is left set.
//
// GIL: must be held by the caller.
AOTI_TORCH_EXPORT AOTITorchError torch_tensor_to_pyobject(
    AtenTensorHandle ath,
    void* py_type,
    void** ret);

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_PYTHON_SHIM
