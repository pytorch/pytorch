#pragma once

#include <ATen/core/C++17.h>
#include <functional>
#include <iostream>

namespace c10 {

enum class DeviceTypeId : uint8_t {
    // Don't use the int values here in the enum (i.e. don't do static_cast to or from int).
    // Instead, if you want to serialize this, write a function with switch/case.
    CPU = 0,
    CUDA = 1,
    UNDEFINED
};

inline std::ostream& operator<<(std::ostream& stream, DeviceTypeId device_type_id) {
    switch(device_type_id) {
        case c10::DeviceTypeId::CPU: return stream << "DeviceTypeId(CPU)";
        case c10::DeviceTypeId::CUDA: return stream << "DeviceTypeId(CUDA)";
        case c10::DeviceTypeId::UNDEFINED: return stream << "DeviceTypeId(UNDEFINED)";
    }
    throw std::logic_error("Unknown DeviceTypeId: " + c10::guts::to_string(static_cast<int>(device_type_id)));
}

}

namespace std {

template <> struct hash<c10::DeviceTypeId> {
    size_t operator()(c10::DeviceTypeId v) const {
        return std::hash<uint8_t>()(static_cast<uint8_t>(v));
    }
};

}
