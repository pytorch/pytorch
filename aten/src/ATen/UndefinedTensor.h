#pragma once

#include "ATen/TensorImpl.h"

namespace at {

struct AT_API UndefinedTensor final : public TensorImpl {
public:
  static inline UndefinedTensor * singleton() {
    return &_singleton;
  }
  const char * toString() const override;
  IntList sizes() const override;
  IntList strides() const override;
  int64_t dim() const override;
  Scalar localScalar() override;
  void * unsafeGetTH(bool retain) override;
  std::unique_ptr<Storage> storage() override;
  static const char * typeString();
private:
  UndefinedTensor();
  static UndefinedTensor _singleton;
public:
  friend struct UndefinedType;
};

} // namespace at
