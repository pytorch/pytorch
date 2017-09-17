#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

namespace torch { namespace autograd {

using at::Context;
using at::Generator;
using at::IntList;
using at::Scalar;
using at::SparseTensor;
using at::Storage;
using at::Tensor;
using at::TensorList;
using at::Type;

struct VariableType : public at::Type {
  VariableType(Context* context, at::Type* baseType);
  virtual at::ScalarType scalarType() override;
  virtual at::Backend backend() override;
  virtual bool isCuda() override;
  virtual bool isSparse() override;
  virtual bool isDistributed() override;
  virtual std::unique_ptr<at::Storage> storage() override;
  virtual std::unique_ptr<at::Storage> storage(size_t size) override;
  virtual std::unique_ptr<at::Storage> storageFromBlob(void * data, int64_t size) override;
  virtual std::unique_ptr<at::Generator> generator() override;
  virtual const char * toString() const override;
  virtual at::TypeID ID() const override;
  virtual size_t elementSizeInBytes() const override;
  static const char * typeString();
  at::Tensor unsafeTensorFromTH(void * th_pointer, bool retain) override;

  virtual void copy(const at::Tensor & src, at::Tensor & dst) override;
  ${type_derived_method_declarations}

private:
  at::Tensor & checked_unpack(const Tensor & t, const char * name, int pos) const;

private:
  at::Type* baseType;
};

}} // namespace torch::autograd
