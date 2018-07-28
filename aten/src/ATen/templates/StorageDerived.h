#pragma once

// ${generated_comment}

$th_headers

#include "ATen/Storage.h"
#include "ATen/Context.h"

#include <memory>

namespace at {

struct Allocator;

struct ${Storage} final : public Storage {
  ${Storage}();
  ${Storage}(StorageImpl* storage_impl) : Storage(storage_impl){};
  ${Storage}(size_t size);
  ${Storage}(size_t size, Allocator* allocator);
  ${Storage}(
      void* data,
      size_t size,
      const std::function<void(void*)>& deleter);
  StorageImpl* storage_impl_;

 protected:
  friend struct ${Type};
};

} // namespace at
