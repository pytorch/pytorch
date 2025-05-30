#pragma once

// This is directly synchronized with caffe2/proto/caffe2.proto, but
// doesn't require me to figure out how to get Protobuf headers into
// ATen/core (which would require a lot more build system hacking.)
// If you modify me, keep me synchronized with that file.

#include <c10/macros/Export.h>
#include <torch/standalone/core/DeviceType.h>

namespace c10 {
C10_API void register_privateuse1_backend(const std::string& backend_name);
C10_API bool is_privateuse1_backend_registered();
} // namespace c10

namespace torch {
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DeviceType;
} // namespace torch
