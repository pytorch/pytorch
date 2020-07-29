#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <c10/core/Device.h>

namespace c10 {
namespace cuda {

DeviceIndex device_count() noexcept;

DeviceIndex current_device();

void set_device(DeviceIndex device);

// Returns are pair of an int containing the version and a string containing an error description,
// if the string is not empty then the function has failed and the integer value should be discarded
std::pair<int32_t, std::string> driver_version();

void device_synchronize();

}} // namespace c10::cuda
