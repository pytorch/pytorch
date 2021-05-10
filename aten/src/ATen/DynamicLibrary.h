#pragma once

#include <ATen/Utils.h>
#include <c10/macros/Export.h>

namespace at {

struct DynamicLibrary {
  AT_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  TORCH_API DynamicLibrary(const char* name, const char* alt_name = nullptr);

  TORCH_API void* sym(const char* name);

  TORCH_API ~DynamicLibrary();

 private:
  void* handle = nullptr;
};

} // namespace at
