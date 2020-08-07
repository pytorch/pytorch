#include <ATen/ATen.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

namespace {
  DeviceType sparseGCSTensorSetToDeviceType(DispatchKeySet key_set) {
    if (key_set.has(DispatchKey::SparseGCS_CPU)) {
      return kCPU;
    } else if (key_set.has(DispatchKey::SparseGCS_CUDA)) {
      return kCUDA;
    } else {
      AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", key_set);
    }
  }
}


SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type)
  :   SparseGCSTensorImpl(key_set, data_type
      , at::empty({1, 0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type))
      , Scalar()  ) {}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type,
                                         at::Tensor pointers, at::Tensor indices, at::Tensor values, at::Tensor reduction,
                                         Scalar fill_value)
  : TensorImpl(key_set, data_type, values.device()),
    pointers_(std::move(pointers)),
    indices_(std::move(indices)),
    values_(std::move(values)),
    reduction_(std::move(reduction)),
    fill_value_(std::move(fill_value)) {}
}
