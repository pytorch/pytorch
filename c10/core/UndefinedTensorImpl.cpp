#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
: TensorImpl(DispatchKey::Undefined, caffe2::TypeMeta(), c10::nullopt) {
}

int64_t UndefinedTensorImpl::size(int64_t d) const {
  AT_ERROR("size(dim) called on an undefined Tensor");
}

int64_t UndefinedTensorImpl::stride(int64_t d) const {
  AT_ERROR("stride(dim) called on an undefined Tensor");
}

bool UndefinedTensorImpl::has_storage() const {
  AT_ERROR("has_storage() called on undefined Tensor");
}

const Storage& UndefinedTensorImpl::storage() const {
  AT_ERROR("storage() called on undefined Tensor");
}

void UndefinedTensorImpl::set_storage_offset(int64_t) {
  AT_ERROR("set_storage_offset() called on an undefined Tensor");
}

IntArrayRef UndefinedTensorImpl::strides() const {
  AT_ERROR("strides() called on undefined Tensor");
}
UndefinedTensorImpl UndefinedTensorImpl::_singleton;

}
