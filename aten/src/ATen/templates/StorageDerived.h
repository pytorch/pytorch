#pragma once

$th_headers

#include "ATen/Storage.h"
#include "ATen/Context.h"

namespace at {

struct ${Storage} : public Storage {
public:
  ${Storage}(Context* context);
  ${Storage}(Context* context, ${THStorage} *wrapped);
  ${Storage}(Context* context, std::size_t size);
  ${Storage}(Context* context,
    void * data, std::size_t size);
  virtual ~${Storage}();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual ${Storage}& retain() override;
  virtual ${Storage}& free() override;

  virtual ${Storage}& resize(long new_size) override;
  virtual ${Storage}& fill(Scalar value) override;
  virtual ${Storage}& set(std::size_t ind, Scalar value) override;
  virtual ${Storage}& fast_set(std::size_t ind, Scalar value) override;
  virtual Scalar get(std::size_t ind) override;
  virtual Scalar fast_get(std::size_t ind) override;

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
