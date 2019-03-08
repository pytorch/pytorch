#pragma once

#include <ATen/TypeDefault.h>
#include <ATen/CheckGenerator.h>

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct UndefinedType final : public TypeDefault {
  explicit UndefinedType();
  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual Backend backend() const override;
  virtual Allocator* allocator() const override;
  virtual Device getDeviceFromPtr(void* data) const override;
  virtual Storage storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
  virtual Storage storageWithAllocator(int64_t size, Allocator* allocator) const override;
  virtual std::unique_ptr<Generator> generator() const override;
  virtual const char * toString() const override;
  virtual Type & toBackend(Backend b) const override;
  virtual Type & toScalarType(ScalarType s) const override;
  virtual TypeID ID() const override;
  virtual Storage unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;
};

} // namespace at
