#pragma once

// ${generated_comment}

$th_headers

#include "ATen/Storage.h"
#include "ATen/Context.h"

#include <memory>

namespace at {

struct Allocator;

struct ${Storage} final : public Storage {
public:
  explicit ${Storage}(Context* context);
  ${Storage}(Context* context, THStorage *wrapped);
  ${Storage}(Context* context, size_t size);
  ${Storage}(Context* context, size_t size, Allocator* allocator);
  ${Storage}(Context* context,
    void * data, size_t size, const std::function<void(void*)> & deleter);
  ~${Storage}();

  size_t elementSize() const override;
  size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  void * unsafeGetTH(bool retain) const override;

  void clear_flag(char flag) override;

  Type& type() const override;
  int getDevice() const override;
  static const char * typeString();


protected:
  friend struct ${Type};
  THStorage *storage;
  Context* context;
};

} // namespace at
