#include "ATen/WrapDimUtils.h"

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

TensorImpl::TensorImpl(
    TensorTypeId type_id,
    ScalarType scalar_type,
    bool is_variable)
    : type_id_(type_id), scalar_type_(scalar_type) {
  auto type = &globalContext().getType(tensorTypeIdToBackend(type_id), scalar_type);
  try {
    Storage* storage = type->storage(true).release();
    storage_ = storage->pImpl();
  } catch (const at::Error& e) {
  }
}

IntList TensorImpl::sizes() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(sizes_.data(), dim());
}

IntList TensorImpl::strides() const {
  // NB: dim in tensor is not synchronized with THTensor, so it's
  // important to apply dim here
  return IntList(strides_.data(), dim());
}

void TensorImpl::release_resources() {
  if (storage_) {
    storage_->release();
  }
}

int64_t TensorImpl::dim() const {
  return is_zero_dim_ ? 0 : sizes_.size();
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
  bool is_zero_dim = (condition_when_zero_dim && sizes().size() == 1 && size(0) == 1) || dim() == 0;
  is_zero_dim_ = is_zero_dim;
  return this;
}

void * TensorImpl::unsafeGetTH(bool retain) {
  if (retain) {
    this->retain();
  }
  return this;
}

std::unique_ptr<Storage> TensorImpl::storage() {
  storage_->retain();
  return std::unique_ptr<Storage>(new Storage(storage_));
}

} // namespace at
