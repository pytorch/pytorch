#pragma once

#include <torch/csrc/stable/c/python_shim.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

// Header-only helpers for converting between a Python torch.Tensor (passed
// as a raw PyObject* / void*) and torch::stable::Tensor. Binding-framework
// specific casters (pybind11, nanobind, ...) live in separate opt-in
// headers that build on top of these. The GIL must be held by the caller.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

// Wrap a Python torch.Tensor (PyObject* passed as void*) as a stable Tensor
// that shares its underlying TensorImpl.
inline Tensor from_pyobject(void* py_obj) {
  AtenTensorHandle ath{};
  TORCH_ERROR_CODE_CHECK(torch_tensor_from_pyobject(py_obj, &ath));
  return Tensor(ath);
}

// Wrap a stable Tensor as a new-reference Python torch.Tensor. py_type is
// an optional PyTypeObject* (passed as void*) used as the result's exact
// Python type; nullptr means default torch.Tensor.
inline void* to_pyobject(const Tensor& t, void* py_type = nullptr) {
  void* raw = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_tensor_to_pyobject(t.get(), py_type, &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

HIDDEN_NAMESPACE_END(torch, stable)
