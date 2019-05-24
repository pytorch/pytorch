#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
: TensorImpl(UndefinedTensorId(), caffe2::TypeMeta(), c10::nullopt) {
}

UndefinedTensorImpl UndefinedTensorImpl::_singleton;

}
