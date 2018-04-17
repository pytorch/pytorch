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

struct CUDALongStorage final : public Storage {
public:
  explicit CUDALongStorage(Context* context);
  CUDALongStorage(Context* context, THCudaLongStorage *wrapped);
  CUDALongStorage(Context* context, std::size_t size);
  CUDALongStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CUDALongStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDALongStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CUDALongStorage& retain() override;
  virtual CUDALongStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CUDALongStorage& resize(int64_t new_size) override;
  virtual CUDALongStorage& fill(Scalar value) override;
  virtual CUDALongStorage& set(std::size_t ind, Scalar value) override;
  virtual CUDALongStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CUDALongType;
  THCudaLongStorage *storage;
  Context* context;
};

} // namespace at
