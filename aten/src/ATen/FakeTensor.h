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
// kernel output that inherited fake state from an input. A view whose kernel
// calls maybe_copy_fake_tensor_metadata (e.g. as_strided, alias, view,
// _reshape_alias) has the input's fake device and mode, while one whose kernel
// does not (view_as_real, view_as_complex, view.dtype) inherits only the Fake
// key. A shallow copy, such as detach's output, has a copy of the input's
// ExtraMeta (see ExtraMeta's copy constructor for what is kept), including the
// real tensor and recorded dispatch keys (see Note [C++ fake dispatch keys]).
// init_cpp_fake itself replaces only the fake device and mode and keeps the
// rest of the ExtraMeta; the fallback then sets the real tensor and dispatch
// keys as Python does for an op's output. A rejected impl is left unchanged,
// unless another thread sets allow_meta to False during the call.
TORCH_API void init_cpp_fake(
    c10::TensorImpl* impl,
    c10::Device device,
    const std::shared_ptr<c10::FakeTensorMode>& mode);

} // namespace at
