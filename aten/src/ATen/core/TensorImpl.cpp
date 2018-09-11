#include <ATen/core/TensorImpl.h>

#include <ATen/core/optional.h>
#include <ATen/core/Backend.h>
#include <ATen/core/WrapDimMinimal.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <ATen/core/VariableHooksInterface.h>

namespace at {

Tensor& TensorImpl::grad() {
  AT_ERROR("grad is not implemented for Tensor");
}

const Tensor& TensorImpl::grad() const {
  AT_ERROR("grad is not implemented for Tensor");
}

TensorImpl::TensorImpl(TensorTypeId type_id, ScalarType scalar_type, Allocator *allocator, TensorImplOptions options)
    : TensorImpl({}, type_id, scalar_type, options) {
  // UndefinedTensors and SparseTensors don't have storages.
  if (type_id != UndefinedTensorId() && scalar_type != ScalarType::Undefined
      && type_id != SparseCPUTensorId() && type_id != SparseCUDATensorId()) {
    storage_ = Storage(scalar_type, 0, allocator, true);
  }
}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, TensorImplOptions options)
    : TensorImpl(std::move(storage), type_id, dataTypeToScalarType(storage.dtype()), options) {}

TensorImpl::TensorImpl(Storage&& storage, TensorTypeId type_id, ScalarType scalar_type, TensorImplOptions options)
    : storage_(std::move(storage)),
      storage_offset_(0),
      sizes_{0},
      strides_{1},
      is_contiguous_(true),
      numel_(0),
      type_id_(type_id),
      scalar_type_(scalar_type),
      options_(options) {}

IntList TensorImpl::sizes() const {
  return sizes_;
}

IntList TensorImpl::strides() const {
  if (!options_.has_strides_) {
    AT_ERROR("This type of tensor does not have strides");
  }
  return strides_;
}

bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = dim() - 1; d >= 0; d--) {
    if (size(d) != 1) {
      if (stride(d) == z) {
        z *= size(d);
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

void TensorImpl::release_resources() {
  if (storage_) {
    storage_ = {};
  }
}

int64_t TensorImpl::dim() const {
  return sizes_.size();
}

int64_t TensorImpl::size(int64_t d) const {
  d = at::maybe_wrap_dim(d, dim(), false);
  return sizes_[d];
}

int64_t TensorImpl::stride(int64_t d) const {
  if (!options_.has_strides_) {
    AT_ERROR("This type of tensor does not have strides");
  }
  d = at::maybe_wrap_dim(d, dim(), false);
  return strides_[d];
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  if (!options_.support_resize_by_maybe_zero_dim_) {
    AT_CHECK(condition_when_zero_dim == (dim() == 0),
           "This type of tensor does not support changing dimensionality via maybe_zero_dim");
  }
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    resize_dim(0);
  }
  return this;
}

const Storage& TensorImpl::storage() const {
  if (!options_.has_storage_) {
    AT_ERROR("This type of tensor does not have storage");
  }
  return storage_;
}

} // namespace at
