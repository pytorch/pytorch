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

  size_t elementSize() const final;

  Type& type() const final;
  static const char * typeString();


protected:
  friend struct ${Type};
  Context* context;
};

} // namespace at
