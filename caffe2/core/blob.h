#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include "caffe2/core/common.h"

#include <ATen/core/blob.h>
#include <c10/util/typeid.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/tensor_int8.h"

namespace caffe2 {

inline bool BlobIsInt8TensorCPUType(const Blob& blob) {
  return blob.meta().Match<int8::Int8TensorCPU>();
}

inline bool BlobIsTensorType(const Blob& blob, DeviceType device_type) {
  bool is_match = blob.meta().Match<Tensor>();
  if (!is_match) {
    return false;
  }
  const Tensor* tensor = &blob.Get<Tensor>();
  return tensor && *tensor && tensor->GetDeviceType() == device_type;
}

inline Tensor* BlobSetTensor(Blob* blob, Tensor&& tensor) {
  return blob->Reset<Tensor>(new Tensor(std::move(tensor)));
}

inline Tensor GetSizedTensorWithOptions(
    Tensor&& previous_tensor,
    at::IntArrayRef dims,
    at::TensorOptions options) {
  Tensor tensor = std::move(previous_tensor);
  if (!tensor.defined()) {
    return caffe2::empty(dims, options);
  }
  if (tensor.GetDevice() == options.device() ||
      (!tensor.GetDevice().has_index() &&
       tensor.GetDeviceType() == options.device().type())) {
    if (tensor.sizes() != dims) {
      // Resize when the dims doesn't match
      tensor.Resize(dims);
    }
    if (tensor.dtype() == options.dtype()) {
      tensor.raw_mutable_data();
    } else {
      // create a new Tensor when the data_type doesn't match
      return caffe2::empty(dims, options);
    }
    return tensor;
  }
  return caffe2::empty(dims, options);
}

// need to keep both functions that returns Tensor* and the one
// returns Tensor for clangr codemod
inline Tensor*
BlobGetMutableTensor(Blob* blob, at::IntArrayRef dims, at::TensorOptions options) {
  if (blob->IsType<Tensor>()) {
    Tensor* tensor = blob->GetMutable<Tensor>();
    if (*tensor) {
      // We only compare device_type if the index is not set since there are Tensors
      // TODO: remove the extra check when all the Tensors are properly initialized
      if (tensor->GetDevice() == options.device() || (!tensor->GetDevice().has_index() && tensor->GetDeviceType() == options.device().type())) {
        if (tensor->sizes() != dims) {
          // Resize when the dims doesn't match
          tensor->Resize(dims);
        }
        if (tensor->dtype() == options.dtype()) {
          tensor->raw_mutable_data();
        } else {
          tensor->raw_mutable_data(options.dtype());
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

inline Tensor
XBlobGetMutableTensor(Blob* blob, at::IntArrayRef dims, at::TensorOptions options) {
  return BlobGetMutableTensor(blob, dims, options)->UnsafeSharedInstance();
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

inline const Tensor& BlobGetTensor(const Blob& blob, DeviceType device_type) {
  if (blob.IsType<Tensor>()) {
    const auto& tensor = blob.Get<Tensor>();
    if (tensor.GetDeviceType() == device_type) {
      return tensor;
    }
  }
  CAFFE_THROW("Blob didn't contain a Tensor or the device_type doesn't match");
}

inline Tensor BlobGetTensorOrUndefined(const Blob& blob) {
  if (blob.IsType<Tensor>()) {
    return blob.Get<Tensor>().UnsafeSharedInstance();
  } else {
    return Tensor();
  }
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
