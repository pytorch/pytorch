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

struct CUDAShortStorage final : public Storage {
public:
  explicit CUDAShortStorage(Context* context);
  CUDAShortStorage(Context* context, THCudaShortStorage *wrapped);
  CUDAShortStorage(Context* context, std::size_t size);
  CUDAShortStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CUDAShortStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDAShortStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CUDAShortStorage& retain() override;
  virtual CUDAShortStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CUDAShortStorage& resize(int64_t new_size) override;
  virtual CUDAShortStorage& fill(Scalar value) override;
  virtual CUDAShortStorage& set(std::size_t ind, Scalar value) override;
  virtual CUDAShortStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CUDAShortType;
  THCudaShortStorage *storage;
  Context* context;
};

} // namespace at
