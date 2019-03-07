#include "c10/util/Exception.h"
#include "MKLDNNTensorImpl.h"

namespace c10 { namespace mkldnn {

MKLDNNTensorImpl::MKLDNNTensorImpl(c10::TensorTypeId type_id, const caffe2::TypeMeta& data_type)
  : c10::TensorImpl(type_id, data_type, nullptr, false) {}

IntArrayRef MKLDNNTensorImpl::sizes() const {
  return sizes_;
}
IntArrayRef MKLDNNTensorImpl::strides() const {
  AT_ERROR("mkldnn tensors do not have strides");
}
bool MKLDNNTensorImpl::is_contiguous() const {
  AT_ERROR("mkldnn tensors do not have is_contiguous");
}
int64_t MKLDNNTensorImpl::size(int64_t d) const {
  return it_.get_dims()[d];
}
int64_t MKLDNNTensorImpl::stride(int64_t d) const {
  AT_ERROR("mkldnn tensors do not have strides");
}
void MKLDNNTensorImpl::resize_dim(int64_t ndim) {
  AT_ERROR("mkldnn tensors do not have resize_dim");
}
void MKLDNNTensorImpl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("mkldnn tensors do not have set_size");
}
void MKLDNNTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("mkldnn tensors do not have set_stride");
}
void MKLDNNTensorImpl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("mkldnn tensors do not have set_storage_offset");
}
int64_t MKLDNNTensorImpl::dim() const {
  return it_.get_dims().size();
}
TensorImpl* MKLDNNTensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  AT_CHECK(condition_when_zero_dim == (dim() == 0),
           "Attempted to maybe_zero_dim on a MKLDNNTensorImpl to ", condition_when_zero_dim,
           " but the MKLDNNTensor's dim() is ", dim(), " and MKLDNNTensors do not support"
           " changing dimensionality via maybe_zero_dim");
  return this;
}
bool MKLDNNTensorImpl::has_storage() const {
  return false;
}
const Storage& MKLDNNTensorImpl::storage() const {
  AT_ERROR("mkldnn tensors do not have storage");
}
int64_t MKLDNNTensorImpl::storage_offset() const {
  AT_ERROR("mkldnn tensors do not have storage");
}

}} // namespace at::native
