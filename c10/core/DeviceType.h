#pragma once

// This is directly synchronized with caffe2/proto/caffe2.proto, but
// doesn't require me to figure out how to get Protobuf headers into
// ATen/core (which would require a lot more build system hacking.)
// If you modify me, keep me synchronized with that file.

#include <c10/macros/Export.h>
#include <torch/standalone/header_only/core/DeviceType.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

namespace c10 {
using torch::standalone::DeviceType;
// clang-format off
// Turn off clang-format for this section to avoid reordering
using torch::standalone::kCPU;
using torch::standalone::kCUDA;
using torch::standalone::kMKLDNN;
using torch::standalone::kOPENGL;
using torch::standalone::kOPENCL;
using torch::standalone::kIDEEP;
using torch::standalone::kHIP;
using torch::standalone::kFPGA;
using torch::standalone::kMAIA;
using torch::standalone::kXLA;
using torch::standalone::kVulkan;
using torch::standalone::kMetal;
using torch::standalone::kXPU;
using torch::standalone::kMPS;
using torch::standalone::kMeta;
using torch::standalone::kHPU;
using torch::standalone::kVE;
using torch::standalone::kLazy;
using torch::standalone::kIPU;
using torch::standalone::kMTIA;
using torch::standalone::kPrivateUse1;
using torch::standalone::COMPILE_TIME_MAX_DEVICE_TYPES;
// clang-format on

C10_API std::string DeviceTypeName(DeviceType d, bool lower_case = false);

C10_API bool isValidDeviceType(DeviceType d);

C10_API void register_privateuse1_backend(const std::string& backend_name);
C10_API std::string get_privateuse1_backend(bool lower_case = true);
C10_API bool is_privateuse1_backend_registered();
} // namespace c10

namespace torch::standalone {
// Due to Argument-dependent lookup (ADL) rule, we have to put this in the
// torch::standalone namespace
C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);
} // namespace torch::standalone

namespace torch {
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DeviceType;
} // namespace torch
