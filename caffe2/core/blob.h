#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include <ATen/core/blob.h>

namespace caffe2 {

inline bool BlobIsTensorType(const Blob& blob, DeviceType device_type) {
  bool is_match = blob.meta().Match<Tensor>();
  if (!is_match) {
    return false;
  }
  const Tensor* tensor = &blob.Get<Tensor>();
  return tensor && tensor->GetDeviceType() == device_type;
}

inline Tensor* BlobGetMutableTensor(Blob* blob, DeviceType device_type) {
  if (BlobIsTensorType(*blob, device_type)) {
    return blob->GetMutable<Tensor>();
  } else {
    VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
            << " DeviceType:" << device_type;
    return blob->Reset<Tensor>(new Tensor(device_type));
  }
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
