#pragma once
#include <torch/torch.h>
#include "BatchTensor.h"
#include <iostream>

struct BatchTensor;

struct BatchTensorType final : public at::Type {
  explicit BatchTensorType(at::Context* context);
  virtual at::ScalarType scalarType() const override;
  virtual at::Backend backend() const override;
  virtual bool is_cuda() const override;
  virtual bool is_sparse() const override;
  virtual bool is_distributed() const override;
  virtual std::unique_ptr<at::Storage> storage() const override;
  virtual std::unique_ptr<at::Storage> storage(size_t size) const override;
  virtual std::unique_ptr<at::Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
  virtual std::unique_ptr<at::Storage> storageWithAllocator(int64_t size, std::unique_ptr<at::Allocator> allocator) const override;
  virtual std::unique_ptr<at::Generator> generator() const override;
  virtual const char * toString() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual Type & toBackend(at::Backend b) const override;
  virtual Type & toScalarType(at::ScalarType s) const override;
  virtual at::TypeID ID() const override;
  static const char * typeString();
  virtual std::unique_ptr<at::Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual at::Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;

  virtual at::Tensor & s_copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking) const override;
  virtual at::Tensor & _s_copy_from(const at::Tensor & self, at::Tensor & dst, bool non_blocking) const override;

  virtual at::Tensor sigmoid(const at::Tensor & self) const override;
  virtual at::Tensor tanh(const at::Tensor & self) const override;
  virtual at::Tensor relu(const at::Tensor & self) const override;
  virtual at::Tensor s_add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha=1) const override;
  virtual at::Tensor add(const at::Tensor & self, at::Scalar other, at::Scalar alpha=1) const override;
  virtual at::Tensor matmul(const at::Tensor & self, const at::Tensor & other) const override;
  virtual at::Tensor contiguous(const at::Tensor & self) const override;
  virtual at::Tensor view(const at::Tensor & self, at::IntList size) const override;
private:
  // at::Tensor elementwise_unary_ops_helper(const at::Tensor & self) const {}
};
