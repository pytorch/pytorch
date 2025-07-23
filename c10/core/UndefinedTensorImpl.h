#pragma once

#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <cstdint>

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
    return &getInstance();
  }
#else
  static constexpr inline TensorImpl* singleton() {
    return &_singleton;
  }
#endif

#ifdef DEBUG
  bool has_storage() const override;
#endif
  void set_storage_offset(int64_t offset) override;

 protected:
  c10::SymBool sym_is_contiguous_custom(MemoryFormat format) const override;
  IntArrayRef strides_custom() const override;
  SymIntArrayRef sym_strides_custom() const override;

 private:
  UndefinedTensorImpl();
#ifdef _WIN32
  static UndefinedTensorImpl& getInstance();
#else
  static UndefinedTensorImpl _singleton;
#endif
  const char* tensorimpl_type_name() const override;
};

} // namespace c10
