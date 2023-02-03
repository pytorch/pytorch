#pragma once

#include <c10/core/TensorImpl.h>

namespace c10 {

struct C10_API UndefinedTensorImpl final : public TensorImpl {
 public:
  // Without this, we get:
  //  error: identifier "at::UndefinedTensorImpl::_singleton" is undefined in
  //  device code
  // (ostensibly because the constexpr tricks MSVC into trying to compile this
  // function for device as well).
#ifdef _WIN32
  static inline TensorImpl* singleton() {
#else
  static constexpr inline TensorImpl* singleton() {
#endif
    return &_singleton;
  }
#ifdef DEBUG
  bool has_storage() const override;
#endif
  void set_storage_offset(int64_t offset) override;

 protected:
  bool is_contiguous_custom(MemoryFormat format) const override;
  IntArrayRef strides_custom() const override;
  SymIntArrayRef sym_strides_custom() const override;

 private:
  UndefinedTensorImpl();
  static UndefinedTensorImpl _singleton;
  const char* tensorimpl_type_name() const override;
};

} // namespace c10
