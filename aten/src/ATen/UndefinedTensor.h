#pragma once

#include "ATen/TensorImpl.h"

namespace at {

struct UndefinedTensor final : public TensorImpl {
public:
  static UndefinedTensor * singleton();
  virtual ~UndefinedTensor();
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void assign_(Scalar s) override;
  virtual void * unsafeGetTH(bool retain) override;
  static const char * typeString();
private:
  UndefinedTensor();
public:
  friend struct UndefinedType;
};

} // namespace at
