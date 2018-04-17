#pragma once

#include <THC/THC.h>
#include <THCUNN/THCUNN.h>
#undef THNN_
#undef THCIndexTensor_
#include <THCS/THCS.h>
#undef THCIndexTensor_

#include "ATen/Storage.h"
#include "ATen/Context.h"

#include <memory>

namespace at {

struct Allocator;

struct CUDAByteStorage final : public Storage {
public:
  explicit CUDAByteStorage(Context* context);
  CUDAByteStorage(Context* context, THCudaByteStorage *wrapped);
  CUDAByteStorage(Context* context, std::size_t size);
  CUDAByteStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CUDAByteStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDAByteStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CUDAByteStorage& retain() override;
  virtual CUDAByteStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CUDAByteStorage& resize(int64_t new_size) override;
  virtual CUDAByteStorage& fill(Scalar value) override;
  virtual CUDAByteStorage& set(std::size_t ind, Scalar value) override;
  virtual CUDAByteStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CUDAByteType;
  THCudaByteStorage *storage;
  Context* context;
};

} // namespace at
