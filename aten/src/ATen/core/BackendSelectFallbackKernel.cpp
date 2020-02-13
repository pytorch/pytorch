#include <ATen/core/dispatch/Dispatcher.h>

using c10::DispatchKey;
using c10::Dispatcher;
using c10::KernelFunction;

namespace {

static auto registry = Dispatcher::singleton().registerBackendFallbackKernel(
    DispatchKey::BackendSelect,
    KernelFunction::makeFallthrough()
);

}
