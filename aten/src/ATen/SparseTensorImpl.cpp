#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/InitialTensorOptions.h>

namespace at {

namespace {
  DeviceType sparseTensorIdToDeviceType(TensorTypeId type_id) {
    if (type_id == SparseCPUTensorId()) {
      return kCPU;
    } else if (type_id == SparseCUDATensorId()) {
      return kCUDA;
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
// Furthermore, we have dim == sparse_dim + dense_dim; since this is a sparse
// tensor, let us say that an empty sparse tensor has sparse_dim == 1 and
// dense_dim == 0.  (There is a degree of freedom here, but given that this
// is a sparse dimension, it seems reasonable to demand that sparse_dim > 0).
//
// This means that we allocate a [1,0] size indices tensor and a [0] size
// values tensor for such an empty tensor.
SparseTensorImpl::SparseTensorImpl(at::TensorTypeId type_id, const caffe2::TypeMeta& data_type)
    : TensorImpl(type_id, data_type, nullptr, false)
    , size_{0}
    , sparse_dim_(1)
    , dense_dim_(0)
    , indices_(at::empty({1, 0}, at::initialTensorOptions().device(sparseTensorIdToDeviceType(type_id)).dtype(ScalarType::Long)))
    , values_(at::empty({0}, at::initialTensorOptions().device(sparseTensorIdToDeviceType(type_id)).dtype(data_type))) {}

IntList SparseTensorImpl::sizes() const {
  return size_;
}
IntList SparseTensorImpl::strides() const {
  AT_ERROR("sparse tensors do not have strides");
}
bool SparseTensorImpl::is_contiguous() const {
  AT_ERROR("sparse tensors do not have is_contiguous");
}
int64_t SparseTensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return size_[d];
}
int64_t SparseTensorImpl::stride(int64_t d) const {
  AT_ERROR("sparse tensors do not have strides");
}
void SparseTensorImpl::resize_dim(int64_t ndim) {
  AT_ERROR("sparse tensors do not have resize_dim");
}
void SparseTensorImpl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("sparse tensors do not have set_size");
}
void SparseTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("sparse tensors do not have set_stride");
}
void SparseTensorImpl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("sparse tensors do not have set_storage_offset");
}

int64_t SparseTensorImpl::dim() const {
  return sparse_dim_ + dense_dim_;
}
TensorImpl* SparseTensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  AT_CHECK(condition_when_zero_dim == (dim() == 0),
           "Attempted to maybe_zero_dim on a SparseTensorImpl to ", condition_when_zero_dim,
           " but the SparseTensor's dim() is ", dim(), " and SparseTensors do not support"
           " changing dimensionality via maybe_zero_dim");
  return this;
}
const Storage& SparseTensorImpl::storage() const {
  AT_ERROR("sparse tensors do not have storage");
}
int64_t SparseTensorImpl::storage_offset() const {
  AT_ERROR("sparse tensors do not have storage");
}
void SparseTensorImpl::set_indices_and_values_unsafe(const Tensor& indices, const Tensor& values) {
  AT_ASSERT(!indices.is_variable() && !values.is_variable());  // They should be plain tensors!

  AT_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
  AT_CHECK(!values.is_sparse(), "expected values to be a dense tensor, but got values of layout ", values.layout());

  AT_CHECK(values.type().toSparse() == type(), "values type must match sparse tensor type");
  AT_CHECK(indices.type().scalarType() == kLong, "indices must be an int64 tensor");
  AT_CHECK(indices.type().backend() == values.type().backend(), "backend of indices (", indices.type().backend(), ") must match backend of values (", values.type().backend(), ")");
  AT_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), "device of indices (", indices.get_device(), ") must match device of values (", values.get_device(), ")");

  AT_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sizes());
  AT_CHECK(indices.size(1) == values.size(0), "indices and values must have same nnz, but got nnz from indices: ", indices.size(1), ", nnz from values: ", values.size(0));
  AT_CHECK(indices.size(0) == sparse_dim_, "indices has incorrect first dimension, expected ", sparse_dim_, ", got ", indices.size(0));
  AT_CHECK(values.dim() == dense_dim_ + 1, "values has incorrect number of dimensions, expected ", dense_dim_ + 1, ", got ", values.dim());

  auto dense_size_original = sizes().slice(sparse_dim_);
  std::vector<int64_t> expected_values_size_vec = {values.size(0)};
  expected_values_size_vec.insert(expected_values_size_vec.end(), dense_size_original.begin(), dense_size_original.end());
  IntList expected_values_size(expected_values_size_vec);
  auto new_values_size = values.sizes();
  AT_CHECK(
    std::equal(expected_values_size.begin(), expected_values_size.end(), new_values_size.begin()),
    "values has incorrect size, expected ", expected_values_size, ", got ", new_values_size
  );

  indices_ = indices;
  values_ = values;

  coalesced_ = false;
}


} // namespace at
