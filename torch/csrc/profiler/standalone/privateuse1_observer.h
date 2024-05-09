#pragma once
#include <torch/csrc/profiler/api.h>
#include <ATen/native/DispatchStub.h>

namespace torch {
namespace profiler {
namespace impl {

using namespace at::native;

using push_privateuse1_callbacks_fn = void (*) (const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);

DECLARE_DISPATCH(push_privateuse1_callbacks_fn, pushPRIVATEUSE1CallbacksStub)

} // namespace impl
} // namespace profiler
} // namespace torch