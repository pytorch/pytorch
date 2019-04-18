#pragma once

#include <c10/core/TensorImpl.h>

namespace c10 {

struct C10_API UndefinedTensorImpl final : public TensorImpl {
 public:
  static TensorImpl * singleton();

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
public:
  friend struct UndefinedType;
};

} // namespace c10
