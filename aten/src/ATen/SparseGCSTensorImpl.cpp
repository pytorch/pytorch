#include <ATen/ATen.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

namespace {
  DeviceType sparseGCSTensorSetToDeviceType(DispatchKeySet key_set) {
    if (key_set.has(DispatchKey::SparseCPU)) {
      return kCPU;
    } else if (key_set.has(DispatchKey::SparseCUDA)) {
      return kCUDA;
    } else {
      AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", key_set);
    }
  }
}


SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type)
  :   SparseGCSTensorImpl(key_set, data_type
      , at::empty({1, 0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type))) {
  
}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                                         at::Tensor pointers, at::Tensor indices, at::Tensor values)
  :
  TensorImpl(key_set, data_type, values.device()) {
}

}
