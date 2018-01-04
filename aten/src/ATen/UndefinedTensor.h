#pragma once

#include "ATen/TensorImpl.h"

namespace at {

struct AT_API UndefinedTensor final : public TensorImpl {
public:
  static inline UndefinedTensor * singleton() {
    return &_singleton;
  }
  virtual ~UndefinedTensor() {}
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void * unsafeGetTH(bool retain) override;
  virtual std::unique_ptr<Storage> storage() override;
  static const char * typeString();
private:
  UndefinedTensor();
  static UndefinedTensor _singleton;
public:
  friend struct UndefinedType;
};

} // namespace at
