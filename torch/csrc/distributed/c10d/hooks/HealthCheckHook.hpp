// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

namespace c10d {

class ProcessGroup;

class TORCH_API HealthCheckHook {
 public:
  static void attach(const c10::intrusive_ptr<ProcessGroup>& pg);

 private:
  explicit HealthCheckHook(std::string backend_name);
  void onAbort() const;

  std::string backend_name_;
};

} // namespace c10d
