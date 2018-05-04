#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

#include <cstdint> // for size_t
#include <functional> // for function
#include <memory> // for unique_ptr
#include <string>
#include <vector>

namespace torch { namespace autograd {

struct Variable;
using at::Context;
using at::Generator;
using at::IntList;
using at::Scalar;
using at::SparseTensor;
using at::Storage;
using at::Tensor;
using at::TensorList;
using at::Type;
using at::ScalarType;
using at::optional;

struct VariableType final : public at::Type {
  VariableType(Context* context, at::Type* baseType);
  virtual at::ScalarType scalarType() const override;
  virtual at::Backend backend() const override;
  virtual bool is_cuda() const override;
  virtual bool is_sparse() const override;
  virtual bool is_distributed() const override;
  virtual std::unique_ptr<at::Storage> storage() const override;
  virtual std::unique_ptr<at::Storage> storage(size_t size) const override;
  virtual std::unique_ptr<at::Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
  virtual std::unique_ptr<Storage> storageWithAllocator(int64_t size, std::unique_ptr<at::Allocator> allocator) const override;
  virtual std::unique_ptr<at::Generator> generator() const override;
  virtual const char * toString() const override;
  virtual at::TypeID ID() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual at::Type & toBackend(at::Backend b) const override;
  virtual at::Type & toScalarType(at::ScalarType s) const override;
  static const char * typeString();
  virtual std::unique_ptr<at::Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual at::Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;

  static at::Type* getType(const at::Type& baseType);
  static at::Type* getType(const at::Tensor& tensor);
  static bool isVariableType(const at::Type& type);
  static std::vector<at::Type*> allTypes();

  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
  virtual Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const override;
  ${type_derived_method_declarations}

private:
  // checks that t is actually a Variable
  static Variable & checked_cast_variable(const Tensor & t, const char * name, int pos);
  static at::Tensor & unpack(const Tensor & t, const char * name, int pos);
  static at::SparseTensor unpack(SparseTensor t, const char * name, int pos);
  static at::Tensor unpack_opt(const Tensor & t, const char * name, int pos);
  static std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos);

  at::Type* baseType;
  std::string str;
};

}} // namespace torch::autograd
