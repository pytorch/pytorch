#pragma once

#include <torch/csrc/stable/c/python_shim.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

// Header-only, framework-agnostic helpers for converting between a Python
// torch.Tensor (passed as a raw PyObject* / void*) and torch::stable::Tensor.
// Binding-framework specific casters (pybind11, nanobind, ...) are expected
// to live in separate opt-in headers that build on top of these.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

/**
 * @brief Convert a Python torch.Tensor to a torch::stable::Tensor.
 *
 * The returned Tensor shares its underlying TensorImpl with the input. The
 * caller MUST hold the GIL.
 *
 * @param py_obj  PyObject* pointing at a torch.Tensor (passed as void* to
 *                avoid pulling Python.h into stable headers). May be a
 *                subclass such as torch.nn.Parameter.
 * @return        A new torch::stable::Tensor owning the wrapped handle.
 *
 * @throws std::runtime_error if py_obj is not a torch.Tensor.
 *
 * Minimum compatible version: PyTorch 2.13.
 */
inline Tensor from_pyobject(void* py_obj) {
  AtenTensorHandle ath{};
  TORCH_ERROR_CODE_CHECK(torch_tensor_from_pyobject(py_obj, &ath));
  return Tensor(ath);
}

/**
 * @brief Convert a torch::stable::Tensor to a Python torch.Tensor.
 *
 * The returned PyObject* is a NEW reference; the caller is responsible for
 * Py_DECREF (or wrapping it in a smart handle from their binding framework,
 * e.g. pybind11::reinterpret_steal or nanobind::steal).
 *
 * If @p py_type is non-null, the result will have that exact Python type.
 * Use this to round-trip subclasses such as torch.nn.Parameter.
 *
 * The caller MUST hold the GIL.
 *
 * @param t        Stable Tensor to wrap.
 * @param py_type  Optional PyTypeObject* (passed as void*); nullptr ->
 *                 default torch.Tensor type.
 * @return         New reference PyObject*, returned as void* to keep this
 *                 header free of Python.h.
 *
 * Minimum compatible version: PyTorch 2.13.
 */
inline void* to_pyobject(const Tensor& t, void* py_type = nullptr) {
  void* raw = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_tensor_to_pyobject(t.get(), py_type, &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

HIDDEN_NAMESPACE_END(torch, stable)
