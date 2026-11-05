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

// Makes impl a fake of mode on device and marks its storage so that reading its
// mutable data pointer warns, or throws if the mode disallows unsafe data
// pointer access, as Python's FakeTensor.__new__ does for every fake it makes,
// a view op's output included. Views made within a kernel's implementation
// (maybe_copy_fake_tensor_metadata), which Python never wraps, share their
// base's marked storage and do not mark it again.
//
// impl must read as meta: a meta tensor, or, inside a kernel, a fake. The
// latter is how the fake fallback (later in this stack) makes a new fake of a
// kernel's output that inherited a fake input's Fake key and ExtraMeta, such as
// a view op's output; its fake device and mode are replaced. A rejected impl is
// left unchanged, unless another thread sets allow_meta to False during the
// call.
TORCH_API void init_cpp_fake(
    c10::TensorImpl* impl,
    c10::Device device,
    const std::shared_ptr<c10::FakeTensorMode>& mode);

} // namespace at
