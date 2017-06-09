#pragma once
#include "TensorLib/Type.h"
#include "TensorLib/Context.h"

namespace tlib {

struct ${Type} : public Type {
  ${Type}(Context* context);
  virtual ScalarType scalarType() override;
  virtual Processor processor() override;
  virtual bool isSparse() override;
  virtual bool isDistributed() override;
  virtual std::unique_ptr<Storage> newStorage() override;
  virtual std::unique_ptr<Storage> newStorage(size_t size) override;
  virtual std::unique_ptr<Generator> newGenerator() override;
  virtual const char * toString() const override;
  virtual int ID() const override;
  static const char * typeString();

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;

  ${type_derived_method_declarations}

private:
  Context* context;
};

} // namespace tlib
