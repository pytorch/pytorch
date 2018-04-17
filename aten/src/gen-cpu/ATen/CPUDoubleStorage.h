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

struct CPUDoubleStorage final : public Storage {
public:
  explicit CPUDoubleStorage(Context* context);
  CPUDoubleStorage(Context* context, THDoubleStorage *wrapped);
  CPUDoubleStorage(Context* context, std::size_t size);
  CPUDoubleStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPUDoubleStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUDoubleStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPUDoubleStorage& retain() override;
  virtual CPUDoubleStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPUDoubleStorage& resize(int64_t new_size) override;
  virtual CPUDoubleStorage& fill(Scalar value) override;
  virtual CPUDoubleStorage& set(std::size_t ind, Scalar value) override;
  virtual CPUDoubleStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPUDoubleType;
  THDoubleStorage *storage;
  Context* context;
};

} // namespace at
