#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
    : TensorImpl(DispatchKey::Undefined, caffe2::TypeMeta(), c10::nullopt) {
  set_storage_access_should_throw();
  set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
}

#ifdef DEBUG
bool UndefinedTensorImpl::has_storage() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !storage_, "UndefinedTensorImpl assumes that storage_ is never set");
  return false;
}
#endif

void UndefinedTensorImpl::set_storage_offset(int64_t) {
  TORCH_CHECK(false, "set_storage_offset() called on an undefined Tensor");
}

const char* UndefinedTensorImpl::tensorimpl_type_name() const {
  return "UndefinedTensorImpl";
}

UndefinedTensorImpl UndefinedTensorImpl::_singleton;

} // namespace c10
