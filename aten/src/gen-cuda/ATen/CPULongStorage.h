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

struct CPULongStorage final : public Storage {
public:
  explicit CPULongStorage(Context* context);
  CPULongStorage(Context* context, THLongStorage *wrapped);
  CPULongStorage(Context* context, std::size_t size);
  CPULongStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPULongStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPULongStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPULongStorage& retain() override;
  virtual CPULongStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPULongStorage& resize(int64_t new_size) override;
  virtual CPULongStorage& fill(Scalar value) override;
  virtual CPULongStorage& set(std::size_t ind, Scalar value) override;
  virtual CPULongStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPULongType;
  THLongStorage *storage;
  Context* context;
};

} // namespace at
