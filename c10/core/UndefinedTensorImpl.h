#pragma once

#include <c10/core/TensorImpl.h>

namespace c10 {

struct C10_API UndefinedTensorImpl final : public TensorImpl {
 public:
  // Without this, we get:
  //  error: identifier "at::UndefinedTensorImpl::_singleton" is undefined in device code
  // (ostensibly because the constexpr tricks MSVC into trying to compile this
  // function for device as well).
#ifdef _WIN32
  static inline TensorImpl * singleton() {
#else
  static constexpr inline TensorImpl * singleton() {
#endif
    return &_singleton;
  }
  IntArrayRef sizes() const override;
  IntArrayRef strides() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  int64_t dim() const override;
  bool has_storage() const override;
  const Storage& storage() const override;
  int64_t storage_offset() const override;
private:
  UndefinedTensorImpl();
  static UndefinedTensorImpl _singleton;
public:
  friend struct UndefinedType;
};

} // namespace c10
