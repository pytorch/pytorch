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

struct CPUByteStorage final : public Storage {
public:
  explicit CPUByteStorage(Context* context);
  CPUByteStorage(Context* context, THByteStorage *wrapped);
  CPUByteStorage(Context* context, std::size_t size);
  CPUByteStorage(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  CPUByteStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUByteStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual CPUByteStorage& retain() override;
  virtual CPUByteStorage& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual CPUByteStorage& resize(int64_t new_size) override;
  virtual CPUByteStorage& fill(Scalar value) override;
  virtual CPUByteStorage& set(std::size_t ind, Scalar value) override;
  virtual CPUByteStorage& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct CPUByteType;
  THByteStorage *storage;
  Context* context;
};

} // namespace at
