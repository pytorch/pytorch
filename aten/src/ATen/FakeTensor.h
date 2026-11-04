#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

#include <memory>

namespace c10 {
struct FakeTensorMode;
struct TensorImpl;
} // namespace c10

namespace at {

TORCH_API void set_and_normalize_fake_device(
    c10::TensorImpl* impl,
    c10::Device device);

// Makes impl a fake of mode on device, as Python's FakeTensor.__new__ does for
// every fake it makes, views included. It also marks impl's storage so that
// reading its mutable data pointer warns, or throws if the mode disallows
// unsafe data pointer access; a later, stricter setting thus also makes a
// storage shared with earlier fakes throw once a new fake of it is made. Views
// made inside kernels, which Python never wraps, are not marked: they only get
// set_and_normalize_fake_device and set_fake_tensor_mode.
TORCH_API void init_cpp_fake(
    c10::TensorImpl* impl,
    c10::Device device,
    const std::shared_ptr<c10::FakeTensorMode>& mode);

} // namespace at
