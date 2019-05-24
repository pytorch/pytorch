#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
: TensorImpl(UndefinedTensorId(), caffe2::TypeMeta(), c10::nullopt) {
}

bool UndefinedTensorImpl::has_storage() const {
  AT_ERROR("has_storage() called on undefined Tensor");
}

UndefinedTensorImpl UndefinedTensorImpl::_singleton;

}
