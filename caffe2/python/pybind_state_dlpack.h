#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/python/dlpack.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace caffe2 {
namespace python {

namespace py = pybind11;

const DLDeviceType* CaffeToDLDeviceType(int device_type);

const DLDataType* CaffeToDLType(const TypeMeta& meta);

const TypeMeta& DLTypeToCaffe(const DLDataType& dl_type);

template <class Context>
class DLPackWrapper {
 public:
  DLPackWrapper(Tensor<Context>* tensor, DeviceOption device_option)
      : tensor(tensor), device_option(device_option) {}

  py::object data() {
    DLContext tensor_context;
    auto device_type_ptr = CaffeToDLDeviceType(device_option.device_type());
    CAFFE_ENFORCE(
        device_type_ptr,
        "Unsupported device type: ",
        device_option.device_type());
    tensor_context.device_type = *device_type_ptr;
    tensor_context.device_id = device_option.cuda_gpu_id();

    if (tensor->size() <= 0) {
      tensor->Resize(0);
    }
    if (tensor->meta().id() == 0) {
      // treat uninitialized tensor as float tensor
      tensor->template mutable_data<float>();
    }
    CAFFE_ENFORCE_GT(tensor->ndim(), 0);

    auto type_ptr = CaffeToDLType(tensor->meta());
    CAFFE_ENFORCE(
        type_ptr,
        "Tensor type is not supported in DLPack: ",
        tensor->meta().name());
    DLDataType tensor_type = *type_ptr;

    DLTensor dlTensor;
    dlTensor.data = const_cast<void*>(tensor->raw_data());
    dlTensor.ctx = tensor_context;
    dlTensor.ndim = tensor->ndim();
    dlTensor.dtype = tensor_type;
    dlTensor.shape = const_cast<int64_t*>(&(tensor->dims()[0]));
    dlTensor.strides = nullptr;
    dlTensor.byte_offset = 0;

    managed_tensor.dlTensor = dlTensor;
    // C2 Tensor memory is managed by C2
    managed_tensor.ctx = nullptr;
    managed_tensor.destructor = [](DLManagedTensor*) {};

    return py::reinterpret_steal<py::object>(
        PyCapsule_New(&managed_tensor, "dltensor", nullptr));
  }

  void feed(py::object obj) {
    CAFFE_ENFORCE(PyCapsule_CheckExact(obj.ptr()), "Expected DLPack capsule");
    DLManagedTensor* dlMTensor =
        (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    CAFFE_ENFORCE(dlMTensor, "Invalid DLPack capsule");
    DLTensor* dlTensor = &dlMTensor->dlTensor;
    auto device_type_ptr = CaffeToDLDeviceType(device_option.device_type());
    CAFFE_ENFORCE(
        device_type_ptr,
        "Unsupported device type: ",
        device_option.device_type());
    CAFFE_ENFORCE(
        dlTensor->ctx.device_type == *device_type_ptr,
        "DLPack tensor device type mismatch");
    int dlpack_device_id = dlTensor->ctx.device_id;
    CAFFE_ENFORCE_EQ(
        dlpack_device_id,
        device_option.cuda_gpu_id(),
        "Expected same device id for DLPack and C2 tensors");

    std::vector<TIndex> dims;
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
    const auto& meta = DLTypeToCaffe(dlTensor->dtype);
    tensor->ShareExternalPointer(
        ((int8_t*)dlTensor->data) + dlTensor->byte_offset,
        meta,
        0,
        [dlMTensor](void*) {
          if (dlMTensor->destructor) {
            dlMTensor->destructor(dlMTensor);
          }
        });
  }

  Tensor<Context>* tensor;
  DeviceOption device_option;
  DLManagedTensor managed_tensor;
};

} // namespace python
} // namespace caffe2
