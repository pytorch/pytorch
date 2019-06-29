#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

#include <memory>
#include <string>

namespace at {

class CAFFE2_API ThreadLocalDebugInfoBase {
 public:
  ThreadLocalDebugInfoBase() {}
  virtual ~ThreadLocalDebugInfoBase() {}

  virtual std::string getString(const char* name) = 0;
  virtual int getInt(const char* name) = 0;
};

CAFFE2_API std::shared_ptr<ThreadLocalDebugInfoBase> getThreadLocalDebugInfo();

CAFFE2_API void setThreadLocalDebugInfo(
    std::shared_ptr<ThreadLocalDebugInfoBase> debug_info);

} // namespace at
