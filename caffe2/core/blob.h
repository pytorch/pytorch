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
#include <ATen/core/intrusive_ptr.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

using TensorImplPtr = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;

inline bool BlobIsTensorType(const Blob& blob, DeviceType device_type) {
  bool is_match = blob.meta().Match<Tensor>();
  if (!is_match) {
    return false;
  }
  const Tensor* tensor = &blob.Get<Tensor>();
  return tensor && *tensor && tensor->GetDeviceType() == device_type;
}

inline bool XBlobIsTensorType(const Blob& blob, DeviceType device_type) {
  if (!blob.meta().Match<TensorImplPtr>()) {
    return false;
  }
  const auto& tensor_impl_ptr = blob.Get<TensorImplPtr>();
  return tensor_impl_ptr && tensor_impl_ptr->device_type() == device_type;
}

inline Tensor* BlobSetTensor(Blob* blob, const Tensor& tensor) {
  return blob->Reset<Tensor>(new Tensor(tensor));
}

inline Tensor
XBlobGetMutableTensor(Blob* blob, at::IntList dims, at::TensorOptions options) {
  auto* tensor_impl_ptr = blob->GetMutableOrNull<TensorImplPtr>();
  // Create a new Tensor(TensorImpl) when either the stored object is not TensorImplPtr
  // or data type does not match or device type does not match
  if (!tensor_impl_ptr || (*tensor_impl_ptr)->dtype() != options.dtype()
      || (*tensor_impl_ptr).get()->GetDevice() != options.device()) {
    VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<TensorImplPtr>()
            << " dims: " << dims  << " options: " << options;
    return Tensor(*blob->Reset<TensorImplPtr>(new TensorImplPtr(caffe2::empty(dims, options).getIntrusivePtr())));
  } else {
    auto& tensor_impl = *tensor_impl_ptr;
    if (tensor_impl->sizes() != dims) {
      // Resize when the dims doesn't match
      tensor_impl->Resize(dims);
    }
    tensor_impl.get()->raw_mutable_data(tensor_impl->dtype());
  }
  return Tensor(*tensor_impl_ptr);
}

// need to keep both for clangr codemod
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
        } else {          // create a new Tensor when the data_type doesn't match
          return blob->Reset<Tensor>(new Tensor(caffe2::empty(dims, options)));
        }
        return tensor;
      }
      // create a new Tensor when device doesn't match
    }
  }

  VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
          << " dims: " << dims;
  // << " options: " << options; (operator<< for Options is in at:: now)
  // TODO: Blob store Tensor directly?
  return blob->Reset<Tensor>(new Tensor(caffe2::empty(dims, options)));
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
  return blob->Reset<Tensor>(new Tensor(device_type));
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
