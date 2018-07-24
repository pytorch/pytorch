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
  ${Storage}();
  ${Storage}(THStorage *wrapped);
  ${Storage}(size_t size);
  ${Storage}(size_t size, Allocator* allocator);
  ${Storage}(
    void * data, size_t size, const std::function<void(void*)> & deleter);
  ~${Storage}();

  size_t elementSize() const final;

  Type& type() const final;
  static const char * typeString();


protected:
  friend struct ${Type};
};

} // namespace at
