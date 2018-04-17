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

struct CPUShortStorage final : public Storage {
public:
  explicit CPUShortStorage(Context* context);
  CPUShortStorage(Context* context, THShortStorage *wrapped);
  CPUShortStorage(Context* context, std::size_t size);
  CPUShortStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPUShortStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUShortStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPUShortStorage& retain() override;
  virtual CPUShortStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPUShortStorage& resize(int64_t new_size) override;
  virtual CPUShortStorage& fill(Scalar value) override;
  virtual CPUShortStorage& set(std::size_t ind, Scalar value) override;
  virtual CPUShortStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPUShortType;
  THShortStorage *storage;
  Context* context;
};

} // namespace at
