#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at { namespace {

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta& data_type)
  :   SparseTensorImpl(key_set, data_type
      , at::empty({1, 0}, at::initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(data_type))) {}

}}
