#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensorImpl::UndefinedTensorImpl()
    : TensorImpl(DispatchKey::Undefined, caffe2::TypeMeta(), std::nullopt) {
  set_storage_access_should_throw();
  // TODO: accessing the sizes on an undefined tensor is not meaningful
  // and should error too, but empirically it does not!
  set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
}

c10::SymBool UndefinedTensorImpl::sym_is_contiguous_custom(
    MemoryFormat format) const {
  return is_contiguous_default(format);
}
IntArrayRef UndefinedTensorImpl::strides_custom() const {
  TORCH_CHECK(false, "strides() called on an undefined Tensor");
}
SymIntArrayRef UndefinedTensorImpl::sym_strides_custom() const {
  TORCH_CHECK(false, "sym_strides() called on an undefined Tensor");
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

#ifdef _WIN32
UndefinedTensorImpl& UndefinedTensorImpl::getInstance() {
  static UndefinedTensorImpl instance;
  return instance;
}
#else
UndefinedTensorImpl UndefinedTensorImpl::_singleton;
#endif

} // namespace c10
