#pragma once

#include <TH/TH.h>
#include <THNN/THNN.h>
#undef THNN_
#include <THS/THS.h>

#include "ATen/Storage.h"
#include "ATen/Context.h"

#include <memory>

namespace at {

struct Allocator;

struct CPUFloatStorage final : public Storage {
public:
  explicit CPUFloatStorage(Context* context);
  CPUFloatStorage(Context* context, THFloatStorage *wrapped);
  CPUFloatStorage(Context* context, std::size_t size);
  CPUFloatStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPUFloatStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUFloatStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPUFloatStorage& retain() override;
  virtual CPUFloatStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPUFloatStorage& resize(int64_t new_size) override;
  virtual CPUFloatStorage& fill(Scalar value) override;
  virtual CPUFloatStorage& set(std::size_t ind, Scalar value) override;
  virtual CPUFloatStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPUFloatType;
  THFloatStorage *storage;
  Context* context;
};

} // namespace at
