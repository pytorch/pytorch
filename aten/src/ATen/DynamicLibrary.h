#pragma once

#include <ATen/Utils.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

namespace c10 {

class DynamicLibraryError : public Error {
  using Error::Error;
};

} // namespace c10

namespace at {

struct DynamicLibrary {
  AT_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  TORCH_API DynamicLibrary(
      const char* name,
      const char* alt_name = nullptr,
      bool leak_handle = false);

  TORCH_API void* sym(const char* name);

  TORCH_API ~DynamicLibrary();

 private:
  bool leak_handle;
  void* handle = nullptr;
};

} // namespace at
