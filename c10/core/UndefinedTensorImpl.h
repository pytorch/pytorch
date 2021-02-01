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
  IntArrayRef strides() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  bool has_storage() const override;
  const Storage& storage() const override;
  void set_storage_offset(int64_t offset) override;
private:
  UndefinedTensorImpl();
  static UndefinedTensorImpl _singleton;
};

} // namespace c10
