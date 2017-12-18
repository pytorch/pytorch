#pragma once

$th_headers

#include "ATen/Storage.h"
#include "ATen/Context.h"

#include <memory>

namespace at {

struct Allocator;

struct ${Storage} final : public Storage {
public:
  explicit ${Storage}(Context* context);
  ${Storage}(Context* context, ${THStorage} *wrapped);
  ${Storage}(Context* context, std::size_t size);
  ${Storage}(Context* context, std::size_t size, std::unique_ptr<Allocator> allocator);
  ${Storage}(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~${Storage}();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual ${Storage}& retain() override;
  virtual ${Storage}& free() override;
  virtual void * unsafeGetTH(bool retain) const override;

  virtual ${Storage}& resize(int64_t new_size) override;
  virtual ${Storage}& fill(Scalar value) override;
  virtual ${Storage}& set(std::size_t ind, Scalar value) override;
  virtual ${Storage}& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

  virtual void set_flag(char flag) override;
  virtual void clear_flag(char flag) override;

  virtual Type& type() const override;
  virtual int getDevice() const override;
  virtual const char * toString() const override;

  static const char * typeString();


protected:
  friend struct ${Type};
  ${THStorage} *storage;
  Context* context;
};

} // namespace at
