#pragma once

namespace c10 { namespace hip {

inline DeviceIndex current_deviceMasqueradingAsCUDA() {
    return current_device();
}

inline void set_deviceMasqueradingAsCUDA(DeviceIndex device) {
    set_device(device);
}

}}  // namespace c10::hip
