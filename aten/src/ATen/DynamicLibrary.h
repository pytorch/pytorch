#pragma once

#include <c10/macros/Export.h>
#include <ATen/Utils.h>

namespace at {

struct DynamicLibrary {
  AT_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  TORCH_API DynamicLibrary(const char* name, const char* alt_name = nullptr, bool allow_fail=false);

  TORCH_API void* sym(const char* name);

  bool available() { return handle; }

  TORCH_API ~DynamicLibrary();

 private:
  void* handle = nullptr;
};

} // namespace at
