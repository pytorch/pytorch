#pragma once

#include "TensorLib/Context.h"

namespace tlib {

struct ${Type} : public Type {
  ${Type}(Context* context);
  virtual ScalarType scalarType() override;
  virtual Processor processor() override;
  virtual bool isSparse() override;
  virtual bool isDistributed() override;
  virtual Storage * newStorage() override;
  virtual Storage * newStorage(size_t size) override;

  // example
  virtual Tensor * add(Tensor & a, Tensor & b) override;

private:
  Context* context;
};

} // namespace tlib
