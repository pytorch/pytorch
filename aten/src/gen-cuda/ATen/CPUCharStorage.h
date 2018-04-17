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

struct CPUCharStorage final : public Storage {
public:
  explicit CPUCharStorage(Context* context);
  CPUCharStorage(Context* context, THCharStorage *wrapped);
  CPUCharStorage(Context* context, std::size_t size);
  CPUCharStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPUCharStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUCharStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPUCharStorage& retain() override;
  virtual CPUCharStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPUCharStorage& resize(int64_t new_size) override;
  virtual CPUCharStorage& fill(Scalar value) override;
  virtual CPUCharStorage& set(std::size_t ind, Scalar value) override;
  virtual CPUCharStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPUCharType;
  THCharStorage *storage;
  Context* context;
};

} // namespace at
