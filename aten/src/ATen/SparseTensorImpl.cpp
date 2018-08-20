#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>

namespace at {

namespace {
  Backend sparseTensorIdToDenseBackend(TensorTypeId type_id) {
    if (type_id == SparseCPUTensorId()) {
      return Backend::CPU;
    } else if (type_id == SparseCUDATensorId()) {
      return Backend::CUDA;
    } else {
      AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", type_id);
    }
  }
}


// An empty dense tensor defaults to a 1-dimensional tensor of size [0]
// (recall, it is not a 0-dimensional tensor, because such a tensor would
// a scalar and have one element)
//
// Thus, an empty sparse tensor should be a 1-dimensional tensor of size [0].
// Furthermore, we have dim == sparseDims + denseDims; since this is a sparse
// tensor, let us say that an empty sparse tensor has sparseDims == 1 and
// denseDims == 0.  (There is a degree of freedom here, but given that this
// is a sparse dimension, it seems reasonable to demand that sparseDims > 0).
//
// In an ideal world, this would then mean we allocate a [1,0] size indices
// tensor and a [0] size values tensor for such an empty tensor.  However,
// we don't currently support zero-size dimensions, so we can't actually
// do this; so we just allocate zero-size tensors for everything.
SparseTensorImpl::SparseTensorImpl(at::TensorTypeId type_id, at::ScalarType scalar_type)
    : TensorImpl(type_id, scalar_type, false)
    , size_{0}
    , sparseDims_(1)
    , denseDims_(0)
    , indices_(globalContext().getTypeOpt(sparseTensorIdToDenseBackend(type_id), ScalarType::Long)->tensor())
    , values_(globalContext().getTypeOpt(sparseTensorIdToDenseBackend(type_id), scalar_type)->tensor()) {}

IntList SparseTensorImpl::sizes() const {
  return size_;
}
IntList SparseTensorImpl::strides() const {
  AT_ERROR("sparse tensors do not have strides");
}
int64_t SparseTensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return size_[d];
}
int64_t SparseTensorImpl::stride(int64_t d) const {
  AT_ERROR("sparse tensors do not have strides");
}

int64_t SparseTensorImpl::dim() const {
  return sparseDims_ + denseDims_;
}
TensorImpl* SparseTensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  AT_CHECK(condition_when_zero_dim == (dim() == 0),
           "Attempted to maybe_zero_dim on a SparseTensorImpl to ", condition_when_zero_dim,
           " but the SparseTensor's dim() is ", dim(), " and SparseTensors do not support"
           " changing dimensionality via maybe_zero_dim");
  return this;
}
std::unique_ptr<Storage> SparseTensorImpl::storage() {
  AT_ERROR("sparse tensors do not have storage");
}
at::StorageImpl* SparseTensorImpl::storageImpl() const {
  AT_ERROR("sparse tensors do not have storage");
}
int64_t SparseTensorImpl::storage_offset() const {
  AT_ERROR("sparse tensors do not have storage");
}
void SparseTensorImpl::set_indices_and_values(const Tensor& indices, const Tensor& values) {
  // TODO: Explicit empty test is needed because we don't handle size zero
  // dimensions at the moment
  bool empty = values.numel() == 0;
  AT_CHECK(values.type().toSparse() == type(), "values type must match sparse tensor type");
  AT_CHECK(indices.type().scalarType() == kLong, "indices must be an int64 tensor");
  AT_CHECK(indices.type().backend() == values.type().backend(), "backend of indices (", indices.type().backend(), ") must match backend of values (", values.type().backend(), ")");
  AT_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), "device of indices (", indices.get_device(), ") must match device of values (", values.get_device(), ")");
  if (!empty) {
    AT_CHECK(indices.dim() == 2, "indices must be nDim x nnz");
    AT_CHECK(indices.size(1) == values.size(0), "indices and values must have same nnz");
    AT_CHECK(indices.size(0) == sparseDims_, "indices has incorrect first dimension, expected ", sparseDims_, ", got ", indices.size(0));
    AT_CHECK(values.dim() == denseDims_ + 1, "values has incorrect number of dimensions, expected ", denseDims_ + 1, ", got ", values.dim());
  } else {
    AT_CHECK(indices.numel() == 0, "if values is empty, indices must be empty too");
  }
  indices_ = indices;
  values_ = values;
  // TODO: Eliminate this ternary when we handle size zero dimensions.
  // (Actually, this will "accidentally" work today because all zero-size
  // tensors have size [0], and so you'll get 0 when empty is zero; but it's
  // more explicit this way.)
  nnz_ = empty ? 0 : values.size(0);
  coalesced_ = false;
}


} // namespace at
