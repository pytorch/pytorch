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

struct CUDADoubleStorage final : public Storage {
public:
  explicit CUDADoubleStorage(Context* context);
  CUDADoubleStorage(Context* context, THCudaDoubleStorage *wrapped);
  CUDADoubleStorage(Context* context, std::size_t size);
  CUDADoubleStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CUDADoubleStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDADoubleStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CUDADoubleStorage& retain() override;
  virtual CUDADoubleStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CUDADoubleStorage& resize(int64_t new_size) override;
  virtual CUDADoubleStorage& fill(Scalar value) override;
  virtual CUDADoubleStorage& set(std::size_t ind, Scalar value) override;
  virtual CUDADoubleStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CUDADoubleType;
  THCudaDoubleStorage *storage;
  Context* context;
};

} // namespace at
