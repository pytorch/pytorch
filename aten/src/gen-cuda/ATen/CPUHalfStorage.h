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

struct CPUHalfStorage final : public Storage {
public:
  explicit CPUHalfStorage(Context* context);
  CPUHalfStorage(Context* context, THHalfStorage *wrapped);
  CPUHalfStorage(Context* context, std::size_t size);
  CPUHalfStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPUHalfStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUHalfStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPUHalfStorage& retain() override;
  virtual CPUHalfStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPUHalfStorage& resize(int64_t new_size) override;
  virtual CPUHalfStorage& fill(Scalar value) override;
  virtual CPUHalfStorage& set(std::size_t ind, Scalar value) override;
  virtual CPUHalfStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPUHalfType;
  THHalfStorage *storage;
  Context* context;
};

} // namespace at
