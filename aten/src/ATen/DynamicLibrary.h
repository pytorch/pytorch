#pragma once

#include <c10/macros/Export.h>
#include <ATen/Utils.h>

namespace at {

struct DynamicLibrary {
  AT_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);

  CAFFE2_API DynamicLibrary(const char* name);

  CAFFE2_API void* sym(const char* name);

  CAFFE2_API ~DynamicLibrary();

 private:
  void* handle = nullptr;
};

} // namespace at
