#pragma once
#include "ATen/Type.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"
#include "ATen/CheckGenerator.h"

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct CPUHalfType final : public Type {
  explicit CPUHalfType(Context* context);
  virtual ScalarType scalarType() const override;
  virtual Backend backend() const override;
  virtual bool is_cuda() const override;
  virtual bool is_sparse() const override;
  virtual bool is_distributed() const override;
  virtual std::unique_ptr<Storage> storage() const override;
  virtual std::unique_ptr<Storage> storage(size_t size) const override;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
  virtual std::unique_ptr<Storage> storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const override;
  virtual std::unique_ptr<Generator> generator() const override;
  virtual const char * toString() const override;
  virtual std::size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;
  static const char * typeString();
  virtual std::unique_ptr<Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;

  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
  virtual int64_t storage_offset(const Tensor & self) const override;
  virtual Tensor & resize_(Tensor & self, IntList size) const override;
  virtual int64_t numel(const Tensor & self) const override;
  virtual Tensor & set_(Tensor & self, Storage & storage) const override;
  virtual Tensor & set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const override;
  virtual Tensor & set_(Tensor & self, const Tensor & source) const override;
  virtual Tensor & set_(Tensor & self) const override;
  virtual bool is_contiguous(const Tensor & self) const override;
  virtual bool is_set_to(const Tensor & self, const Tensor & tensor) const override;
  virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) const override;
  virtual Tensor t(const Tensor & self) const override;
  virtual Tensor clone(const Tensor & self) const override;
  virtual Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const override;
  virtual void* data_ptr(const Tensor & self) const override;
  virtual Tensor tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const override;
  virtual Tensor tensor(IntList size) const override;
  virtual Tensor tensor(IntList size, IntList stride) const override;
  virtual Tensor tensor() const override;
  virtual Tensor alias(const Tensor & self) const override;
  virtual Tensor & as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const override;
  virtual Tensor as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const override;
  virtual Tensor & as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const override;
};

} // namespace at
