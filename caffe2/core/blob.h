#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include "caffe2/core/common.h"

#include <ATen/core/blob.h>
#include <ATen/core/typeid.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

inline bool BlobIsTensorType(const Blob& blob, DeviceType device_type) {
  bool is_match = blob.meta().Match<Tensor>();
  if (!is_match) {
    return false;
  }
  const Tensor* tensor = &blob.Get<Tensor>();
  return tensor && *tensor && tensor->GetDeviceType() == device_type;
}

inline Tensor* BlobSetTensor(Blob* blob, const Tensor& tensor) {
  return blob->Reset<Tensor>(new Tensor(tensor));
}

inline Tensor*
BlobGetMutableTensor(Blob* blob, at::IntList dims, at::TensorOptions options) {
  if (blob->IsType<Tensor>()) {
    Tensor* tensor = blob->GetMutable<Tensor>();
    if (*tensor) {
      if (tensor->GetDevice() == options.device()) {
        if (tensor->sizes() != dims) {
          // Resize when the dims doesn't match
          tensor->Resize(dims);
        }
        if (tensor->dtype() == options.dtype()) {
          tensor->raw_mutable_data();
        } else {
          // create a new Tensor when the data_type doesn't match
          return BlobSetTensor(blob, caffe2::empty(dims, options));
        }
        return tensor;
      }
      // create a new Tensor when device doesn't match
    }
  }

  VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
          << " dims: " << dims;
  // << " options: " << options; (operator<< for Options is in at:: now)
  return BlobSetTensor(blob, caffe2::empty(dims, options));
}

inline Tensor* BlobGetMutableTensor(Blob* blob, DeviceType device_type) {
  if (blob->IsType<Tensor>()) {
    Tensor* tensor = blob->GetMutable<Tensor>();
    if (*tensor && tensor->GetDeviceType() == device_type) {
      return tensor;
    }
  }

  // if we're here, then either Blob didn't hold a Tensor
  // or that Tensor had the wrong DeviceType.
  VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
          << " DeviceType:" << device_type;

  return BlobSetTensor(blob, Tensor(device_type));
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
