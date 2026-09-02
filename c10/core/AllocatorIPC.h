#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace c10::allocator::ipc {

class C10_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

struct C10_API ShareableHandle {
  std::ptrdiff_t offset;
  std::string handle;
  std::optional<std::shared_ptr<void>> owner;
};

} // namespace c10::allocator::ipc
