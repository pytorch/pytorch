#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

#include <memory>

namespace c10 {
struct FakeTensorMode;
struct TensorImpl;
struct VariableVersion;
} // namespace c10

namespace at {

class Tensor;

TORCH_API void set_and_normalize_fake_device(
    c10::TensorImpl* impl,
    c10::Device device);

// Mutates impl, and the storage it may share with other tensors: makes impl a
// fake of mode on device and marks its storage so that reading its
// mutable data pointer warns, or throws if the mode disallows unsafe data
// pointer access, as Python's FakeTensor.__new__ does for every fake it makes,
// a view op's output included. Temporary views inside a Meta kernel share
// their base's marked storage; only returned outputs are initialized as fakes.
//
// impl must read as meta. New views and shallow copies made inside a Meta
// kernel do not inherit fake state (see Note [C++ fake outputs inside Meta
// kernels]); the fallback initializes them when the kernel returns.
// init_cpp_fake itself replaces only the fake device and mode and keeps the
// rest of the ExtraMeta; the fallback then sets the real tensor and dispatch
// keys as Python does for an op's output. A rejected impl is left unchanged.
TORCH_API void init_cpp_fake(
    c10::TensorImpl* impl,
    c10::Device device,
    const std::shared_ptr<c10::FakeTensorMode>& mode);

namespace impl {

// FakeTensorMode.may_turn_const with CONSTANT_NUMEL_LIMIT == 1.
TORCH_API bool may_turn_const(const Tensor& tensor);

// Metadata-only fake kernels. The caller validates the inputs and disables
// dispatch, but must not enter InKernelInvocationGuard: the copies stay fake.
TORCH_API Tensor fake_detach(const Tensor& self);
TORCH_API Tensor& fake_shallow_copy_data_(Tensor& self, const Tensor& source);

// Unlike the kernels above, data access keeps Python's shallow-copy hook
// active.
TORCH_API Tensor fake_tensor_data(
    const Tensor& self,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change);

} // namespace impl

} // namespace at
