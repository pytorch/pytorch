#pragma once

#include <functional>
#include <iostream>

namespace c10 {

enum class DeviceId : uint8_t {
    UNDEFINED,
    CPU,
    CUDA
};

inline std::ostream& operator<<(std::ostream& stream, DeviceId device_id) {
    return stream << static_cast<uint8_t>(device_id);
}

}

namespace std {

template <> struct hash<c10::DeviceId> {
    size_t operator()(c10::DeviceId v) const {
        return std::hash<uint8_t>()(static_cast<uint8_t>(v));
    }
};

}
