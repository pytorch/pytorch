#pragma once

#include "ATen/core/TensorImpl.h"

namespace at {

struct AT_API UndefinedTensor final : public TensorImpl {
public:
  // Without this, we get:
  //  error: identifier "at::UndefinedTensor::_singleton" is undefined in device code
  // (ostensibly because the constexpr tricks MSVC into trying to compile this
  // function for device as well).
#ifdef _WIN32
  static inline TensorImpl * singleton() {
#else
  static constexpr inline TensorImpl * singleton() {
#endif
    return &_singleton;
  }
  IntList sizes() const override;
  IntList strides() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  int64_t dim() const override;
  const Storage& storage() const override;
  int64_t storage_offset() const override;
private:
  UndefinedTensor();
  static UndefinedTensor _singleton;
public:
  friend struct UndefinedType;
};

} // namespace at
