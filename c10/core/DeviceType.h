#pragma once

#include <c10/macros/Export.h>

// If you modified DeviceType in caffe2/proto/caffe2.proto, please also sync
// your changes into torch/headeronly/core/DeviceType.h.
#include <torch/headeronly/core/DeviceType.h>

#include <optional>
#include <ostream>
#include <string>

namespace c10 {

C10_API std::string DeviceTypeName(DeviceType d, bool lower_case = false);

C10_API bool isValidDeviceType(DeviceType d);

C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);

C10_API void register_privateuse1_backend(const std::string& backend_name);
C10_API std::string get_privateuse1_backend(bool lower_case = true);

C10_API DeviceType register_privateuse_backend(const std::string& backend_name);
C10_API std::string get_privateuse_backend(
    DeviceType device_type,
    bool lower_case = true);
C10_API std::optional<DeviceType> get_privateuse_backend_device_type(
    const std::string& backend_name);
C10_API bool is_privateuse_backend(DeviceType device_type);
C10_API bool is_privateuse_backend_registered(DeviceType device_type);
C10_API bool is_privateuse1_backend_registered();

} // namespace c10

namespace torch {
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DeviceType;
} // namespace torch
