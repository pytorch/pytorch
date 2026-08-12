#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device_struct.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

// Header-only helpers converting between Python objects (passed as raw
// PyObject* / void*) and their torch::stable equivalents. These are
// libtorch-only to link against, but require libtorch_python to be loaded at
// runtime (see the Python interop shims section in torch/csrc/stable/c/shim.h).
// The GIL must be held by the caller.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// Wrap a Python torch.Tensor (PyObject* passed as void*) as a stable Tensor
// that shares its underlying TensorImpl.
inline Tensor tensor_from_pyobject(void* py_obj) {
  AtenTensorHandle ath = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_tensor_from_pyobject(py_obj, &ath));
  return Tensor(ath);
}

// Wrap a stable Tensor as a new-reference Python torch.Tensor. py_type is an
// optional PyTypeObject* (passed as void*) used as the result's exact Python
// type; nullptr means default torch.Tensor.
inline void* tensor_to_pyobject(const Tensor& t, void* py_type = nullptr) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_tensor_to_pyobject(t.get(), py_type, &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_15_0

// Whether py_obj is a Python torch.Tensor (or a subclass). A probe for callers
// that want to type-check before tensor_from_pyobject (which errors on
// non-tensors).
inline bool is_tensor_pyobject(void* py_obj) {
  bool ret = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_is_tensor_pyobject(py_obj, &ret));
  return ret;
}

// The dtype/device helpers below translate codes through the stable enum
// mappings (torch::stable::detail::from/to), which cover a subset of all
// ScalarTypes / DeviceTypes; a valid torch.dtype / torch.device outside that
// subset errors even though the C shim itself can represent it.

// Read the ScalarType out of a Python torch.dtype (PyObject* passed as void*).
inline torch::headeronly::ScalarType dtype_from_pyobject(void* py_obj) {
  int32_t dtype = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_dtype_from_pyobject(py_obj, &dtype));
  return torch::stable::detail::to<torch::headeronly::ScalarType>(
      torch::stable::detail::from(dtype));
}

// Wrap a ScalarType as a new-reference Python torch.dtype.
inline void* dtype_to_pyobject(torch::headeronly::ScalarType dtype) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_dtype_to_pyobject(
      torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype)),
      &raw));
  return raw;
}

// Read the Device out of a Python torch.device (PyObject* passed as void*).
inline Device device_from_pyobject(void* py_obj) {
  int32_t device_type = 0;
  int32_t device_index = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_device_from_pyobject(py_obj, &device_type, &device_index));
  DeviceType extension_device_type = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  return Device(extension_device_type, static_cast<DeviceIndex>(device_index));
}

// Wrap a Device as a new-reference Python torch.device.
inline void* device_to_pyobject(const Device& device) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_device_to_pyobject(
      torch::stable::detail::to<int32_t>(
          torch::stable::detail::from(device.type())),
      static_cast<int32_t>(device.index()),
      &raw));
  return raw;
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_15_0

HIDDEN_NAMESPACE_END(torch, stable)
