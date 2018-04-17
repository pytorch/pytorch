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

struct CUDAHalfStorage final : public Storage {
public:
  explicit CUDAHalfStorage(Context* context);
  CUDAHalfStorage(Context* context, THCudaHalfStorage *wrapped);
  CUDAHalfStorage(Context* context, std::size_t size);
  CUDAHalfStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CUDAHalfStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDAHalfStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CUDAHalfStorage& retain() override;
  virtual CUDAHalfStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CUDAHalfStorage& resize(int64_t new_size) override;
  virtual CUDAHalfStorage& fill(Scalar value) override;
  virtual CUDAHalfStorage& set(std::size_t ind, Scalar value) override;
  virtual CUDAHalfStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CUDAHalfType;
  THCudaHalfStorage *storage;
  Context* context;
};

} // namespace at
