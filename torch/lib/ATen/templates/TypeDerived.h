#pragma once
#include "ATen/Type.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"
#include "ATen/CheckGenerator.h"

namespace at {

struct ${Type} : public Type {
  explicit ${Type}(Context* context);
  virtual ScalarType scalarType() override;
  virtual Backend backend() override;
  virtual bool isCuda() override;
  virtual bool isSparse() override;
  virtual bool isDistributed() override;
  virtual std::unique_ptr<Storage> storage() override;
  virtual std::unique_ptr<Storage> storage(size_t size) override;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size) override;
  virtual std::unique_ptr<Generator> generator() override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
  static const char * typeString();
  Tensor unsafeTensorFromTH(void * th_pointer, bool retain) override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;

  virtual void copy(const Tensor & src, Tensor & dst) override;
  ${type_derived_method_declarations}
};

} // namespace at
