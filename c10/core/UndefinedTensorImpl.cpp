#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
: TensorImpl(DispatchKey::Undefined, caffe2::TypeMeta(), c10::nullopt) {
}

IntArrayRef UndefinedTensorImpl::sizes() const {
  AT_ERROR("sizes() called on undefined Tensor");
}

int64_t UndefinedTensorImpl::size(int64_t d) const {
  AT_ERROR("size(dim) called on an undefined Tensor");
}

int64_t UndefinedTensorImpl::stride(int64_t d) const {
  AT_ERROR("stride(dim) called on an undefined Tensor");
}

int64_t UndefinedTensorImpl::dim() const {
  AT_ERROR("dim() called on undefined Tensor");
}

bool UndefinedTensorImpl::has_storage() const {
  AT_ERROR("has_storage() called on undefined Tensor");
}

const Storage& UndefinedTensorImpl::storage() const {
  AT_ERROR("storage() called on undefined Tensor");
}

int64_t UndefinedTensorImpl::storage_offset() const {
  AT_ERROR("storage_offset() called on an undefined Tensor");
}

IntArrayRef UndefinedTensorImpl::strides() const {
  AT_ERROR("strides() called on undefined Tensor");
}
UndefinedTensorImpl UndefinedTensorImpl::_singleton;

}
