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

struct CUDACharStorage final : public Storage {
public:
  explicit CUDACharStorage(Context* context);
  CUDACharStorage(Context* context, THCudaCharStorage *wrapped);
  CUDACharStorage(Context* context, std::size_t size);
  CUDACharStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CUDACharStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDACharStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CUDACharStorage& retain() override;
  virtual CUDACharStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CUDACharStorage& resize(int64_t new_size) override;
  virtual CUDACharStorage& fill(Scalar value) override;
  virtual CUDACharStorage& set(std::size_t ind, Scalar value) override;
  virtual CUDACharStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CUDACharType;
  THCudaCharStorage *storage;
  Context* context;
};

} // namespace at
