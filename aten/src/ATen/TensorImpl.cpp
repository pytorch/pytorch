#include <ATen/TensorImpl.h>

#include "ATen/Context.h"
#include <ATen/Tensor.h>
#include <ATen/core/optional.h>
#include <ATen/Context.h>
#include <ATen/Backend.h>

#include <ATen/detail/VariableHooksInterface.h>

#include <TH/THTensor.hpp>

namespace at {

Type& TensorImpl::type() const {
  // Select backend from the hard-coded ones that the legacy ATen dispatcher
  // knows about
  Backend backend = tensorTypeIdToBackend(type_id_);
  Type* base_type = &globalContext().getType(backend, scalar_type_);
  if (is_variable_) {
    return detail::getVariableHooks().getVariableType(*base_type);
  } else {
    return *base_type;
  }
}

Tensor& TensorImpl::grad() {
  AT_ERROR("grad is not implemented for Tensor");
}

const Tensor& TensorImpl::grad() const {
  AT_ERROR("grad is not implemented for Tensor");
}

Tensor TensorImpl::detach() const {
  AT_ERROR("detach is not implemented for Tensor");
}

const char* TensorImpl::toString() const {
  // This matches behavior with VariableImpl
  return type().toString();
}

void TensorImpl::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  AT_ERROR("backward is not implemented for Tensor");
}

void TensorImpl::set_data(Tensor new_data) {
  AT_ERROR("set_type is not implemented for Tensor");
}

void Tensor::backward(
    at::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  pImpl->backward(std::move(gradient), keep_graph, create_graph);
}

TensorImpl::TensorImpl(TensorTypeId type_id, ScalarType scalar_type, bool is_variable)
    : TensorImpl(nullptr, type_id, scalar_type, is_variable) {
  // UndefinedTensors and SparseTensors don't have storages.
  if (type_id != UndefinedTensorId() && scalar_type != ScalarType::Undefined
      && type_id != SparseCPUTensorId() && type_id != SparseCUDATensorId()) {
    auto type = &globalContext().getType(tensorTypeIdToBackend(type_id), scalar_type);
    auto storage = type->storage(true);
    storage_ = storage->pImpl();
    storage_->retain();
  }
}

TensorImpl::TensorImpl(StorageImpl* storage, TensorTypeId type_id, bool is_variable)
    : TensorImpl(storage, type_id, storage->scalar_type(), is_variable) {}

TensorImpl::TensorImpl(StorageImpl* storage, TensorTypeId type_id, ScalarType scalar_type, bool is_variable)
    : storage_(storage),
      storage_offset_(0),
      sizes_{0},
      strides_{1},
      type_id_(type_id),
      scalar_type_(scalar_type),
      is_variable_(is_variable) {}

TensorImpl::~TensorImpl() {
  if (storage_) {
    storage_->release();
    storage_ = nullptr;
  }
}

IntList TensorImpl::sizes() const {
  return sizes_;
}

IntList TensorImpl::strides() const {
  return strides_;
}

void TensorImpl::release_resources() {
  if (storage_) {
    storage_->release();
    storage_ = nullptr;
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
  d = at::maybe_wrap_dim(d, dim(), false);
  return strides_[d];
}

TensorImpl* TensorImpl::maybe_zero_dim(bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && this->sizes().size() == 1 && this->size(0) == 1;
  if (set_zero_dim) {
    THTensor_resizeDim(this, 0);
  }
  return this;
}

std::unique_ptr<Storage> TensorImpl::storage() {
  storage_->retain();
  return std::unique_ptr<Storage>(new Storage(storage_));
}

} // namespace at
