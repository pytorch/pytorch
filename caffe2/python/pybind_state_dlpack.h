#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/python/dlpack.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace caffe2 {
namespace python {

namespace py = pybind11;

const DLDeviceType* CaffeToDLDeviceType(int device_type);

const DLDataType* CaffeToDLType(const TypeMeta meta);

const TypeMeta DLTypeToCaffe(const DLDataType& dl_type);

// TODO: remove context
template <class Context>
class DLPackWrapper {
 public:
  DLPackWrapper(Tensor* tensor, DeviceOption device_option)
      : tensor(tensor), device_option(device_option) {}

  py::object data() {
    DLDevice tensor_context;
    auto device_type_ptr = CaffeToDLDeviceType(device_option.device_type());
    CAFFE_ENFORCE(
        device_type_ptr,
        "Unsupported device type: ",
        device_option.device_type());
    tensor_context.device_type = *device_type_ptr;
    tensor_context.device_id = device_option.device_id();

    if (tensor->numel() <= 0) {
      tensor->Resize(0);
    }
    if (tensor->dtype() == ScalarType::Undefined) {
      // treat uninitialized tensor as float tensor
      tensor->template mutable_data<float>();
    }
    CAFFE_ENFORCE_GT(tensor->dim(), 0);

    auto type_ptr = CaffeToDLType(tensor->dtype());
    CAFFE_ENFORCE(
        type_ptr,
        "Tensor type is not supported in DLPack: ",
        tensor->dtype().name());
    DLDataType tensor_type = *type_ptr;

    DLTensor dlTensor;
    dlTensor.data = const_cast<void*>(tensor->raw_data());
    dlTensor.device = tensor_context;
    dlTensor.ndim = tensor->dim();
    dlTensor.dtype = tensor_type;
    dlTensor.shape = const_cast<int64_t*>(&(tensor->sizes()[0]));
    dlTensor.strides = nullptr;
    dlTensor.byte_offset = 0;

    managed_tensor.dl_tensor = dlTensor;
    // C2 Tensor memory is managed by C2
    managed_tensor.manager_ctx = nullptr;
    managed_tensor.deleter = [](DLManagedTensor*) {};

    return py::reinterpret_steal<py::object>(
        PyCapsule_New(&managed_tensor, "dltensor", nullptr));
  }

  void feed(py::object obj) {
    CAFFE_ENFORCE(PyCapsule_CheckExact(obj.ptr()), "Expected DLPack capsule");
    DLManagedTensor* dlMTensor =
        (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    CAFFE_ENFORCE(dlMTensor, "Invalid DLPack capsule");
    DLTensor* dlTensor = &dlMTensor->dl_tensor;
    auto device_type_ptr = CaffeToDLDeviceType(device_option.device_type());
    CAFFE_ENFORCE(
        device_type_ptr,
        "Unsupported device type: ",
        device_option.device_type());
    CAFFE_ENFORCE(
        dlTensor->device.device_type == *device_type_ptr,
        "DLPack tensor device type mismatch");
    int dlpack_device_id = dlTensor->device.device_id;
    CAFFE_ENFORCE_EQ(
        dlpack_device_id,
        device_option.device_id(),
        "Expected same device id for DLPack and C2 tensors");

    std::vector<int64_t> dims;
    dims.reserve(dlTensor->ndim);
    for (int idx = 0; idx < dlTensor->ndim; ++idx) {
      dims.push_back(dlTensor->shape[idx]);
    }

    if (dlTensor->strides) {
      int64_t stride = 1;
      for (int idx = dims.size() - 1; idx >= 0; --idx) {
        CAFFE_ENFORCE_EQ(
            stride,
            dlTensor->strides[idx],
            "Tensors with non-standard strides are not supported");
        stride *= dims[idx];
      }
    }

    tensor->Resize(dims);
    caffe2::TypeMeta meta = DLTypeToCaffe(dlTensor->dtype);
    at::Device device = at::Device(tensor->GetDeviceType());
    tensor->ShareExternalPointer(
        at::DataPtr(
            (void*)(((int8_t*)dlTensor->data) + dlTensor->byte_offset),
            static_cast<void*>(dlMTensor),
            [](void* t_ptr) -> void {
              DLManagedTensor* mt_ptr = static_cast<DLManagedTensor*>(t_ptr);
              if (mt_ptr->deleter) {
                mt_ptr->deleter(mt_ptr);
              }
            },
            device),
        meta,
        0);
  }

  Tensor* tensor;
  DeviceOption device_option;
  DLManagedTensor managed_tensor;
};

} // namespace python
} // namespace caffe2
