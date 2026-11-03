#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

namespace c10 {
struct FakeTensorMode;
struct TensorImpl;
} // namespace c10

namespace at {

TORCH_API void set_and_normalize_fake_device(
    c10::TensorImpl* impl,
    c10::Device device);

// Marks the storage of a newly created C++ fake of mode, as Python's
// FakeTensor.__new__ does: a fake has no real data, so reading its mutable
// data pointer warns, or throws if mode disallows unsafe data pointer access.
// Views share their base's storage, so only call this where a fake is created.
TORCH_API void mark_fake_storage(
    c10::TensorImpl* impl,
    const c10::FakeTensorMode& mode);

} // namespace at
