#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/nested.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

namespace torch {
namespace utils {

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
c10::TensorOptions typeIdWithDefault(
    PythonArgs& r,
    int device_idx,
    c10::DispatchKey dispatch_key) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  if (!r.isNone(device_idx)) {
    options = options.device(r.device(device_idx));
  }
  return options;
}

at::Tensor nested_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    torch::PythonArgs& r) {
  TORCH_CHECK(r.idx == 0, "nested_tensor(): invalid arguments");

  PyObject* data = r.pyobject(0);
  // Check if data is a list: Only List[Tensor] and List[List...[Scalar]] are
  // accepted for now
  TORCH_CHECK_TYPE(
      PyList_Check(data),
      "Only lists (List[Tensor] and List[List...[Scalar]]) are accepted in nested_tensor");

  auto dtype_val = r.scalartypeWithDefault(1, scalar_type);
  auto tensor_options = typeIdWithDefault(r, 2, dispatch_key);
  bool pin_memory = r.toBool(3);
  bool args_requires_grad = r.toBool(4);

  TORCH_CHECK(
      PyList_Size(data) >= 0,
      "Something went really wrong and your list has negative size");

  // Check whether we are dealing with lists of tensors or not
  std::vector<at::Tensor> new_list(PyList_Size(data));
  for (const auto i : c10::irange(PyList_Size(data))) {
    PyObject* elem = PyList_GetItem(data, i);
    if (THPVariable_Check(elem)) {
      new_list[i] = THPVariable_Unpack(PyList_GetItem(data, i)).detach();
      TORCH_CHECK(
          !new_list[i].is_nested(),
          "We do not accept nested tensors as input to nested tensors");
      TORCH_CHECK(
          new_list[i].layout() == kStrided,
          "We do not accept non-strided layouts as input to nested tensors");
    } else {
      PythonArgs elem_r(r);
      std::array<PyObject*, 6> elem_args = {
          elem, // data
          r.args[1], // dtpye
          nullptr, // device (cpu)
          nullptr, // no pinned memory
          r.args[4], // requires grad
          nullptr // names
      };
      elem_r.args = elem_args.data();
      new_list[i] = tensor_ctor(dispatch_key, scalar_type, elem_r);
    }
  }

  at::ScalarType final_dtype = dtype_val;
  if (r.isNone(1) && new_list.size() > 0) {
    final_dtype = c10::typeMetaToScalarType(new_list[0].dtype());
  }
  at::Device final_device = tensor_options.device();
  if (r.isNone(2) && new_list.size() > 0) {
    final_device = new_list[0].device();
  }
  auto out = at::_nested_tensor_from_tensor_list(
      new_list, final_dtype, c10::nullopt, final_device, pin_memory);
  out.requires_grad_(args_requires_grad);
  return out;
}

} // namespace utils
} // namespace torch
