#pragma once

#include "ATen/TensorImpl.h"

namespace at {

struct AT_API UndefinedTensor final : public TensorImpl {
public:
  static inline UndefinedTensor * singleton() {
    return &_singleton;
  }
  IntList sizes() const override;
  IntList strides() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  int64_t dim() const override;
  std::unique_ptr<Storage> storage() override;
  at::StorageImpl* storageImpl() const override;
  int64_t storage_offset() const override;
private:
  UndefinedTensor();
  static UndefinedTensor _singleton;
public:
  friend struct UndefinedType;
};

} // namespace at
