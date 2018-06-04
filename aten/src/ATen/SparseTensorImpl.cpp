#include <ATen/SparseTensorImpl.h>
#include <ATen/Type.h>

namespace at {

SparseTensorImpl::SparseTensorImpl(Type * type)
    : TensorImpl(type)
    , indices_(type->toScalarType(ScalarType::Long).tensor())
    , values_(type->tensor()) {}

const char * SparseTensorImpl::toString() const {
  // TODO: also give back type information
  return "SparseTensor";
}
IntList SparseTensorImpl::sizes() const {
  return size_;
}
IntList SparseTensorImpl::strides() const {
  AT_ERROR("sparse tensors do not have strides");
}
int64_t SparseTensorImpl::dim() const {
  return dimI_ + dimV_;
}
Scalar SparseTensorImpl::localScalar() {
  AT_ERROR("sparse tensors cannot be scalars");
}
void * SparseTensorImpl::unsafeGetTH(bool retain) {
  AT_ERROR("unsafeGetTH not supported for new style TensorImpl");
}
std::unique_ptr<Storage> SparseTensorImpl::storage() {
  AT_ERROR("sparse tensors do not have storage");
}

} // namespace at
