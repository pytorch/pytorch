#include <torch/csrc/lazy/core/lazy_mode.h>

#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/library.h>

// TODO(whc) we can't actually depend on ts backend code from here, but we could refactor if reusing the callback
// turns out to be the best way to implement this
#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>

namespace torch {
namespace lazy {

// _nests, _inc, _dec are for handling nested lazy mode calls, and should only be called by LazyModeEnter,Exit
size_t& _lazy_mode_nests() {
    thread_local size_t _nest_counter{0};
    return _nest_counter;
}

size_t _lazy_mode_inc() {
    // postincrementing so first call returns 0
    return _lazy_mode_nests()++;
}

size_t _lazy_mode_dec() {
    // preincrementing so last call returns 0
    TORCH_CHECK(_lazy_mode_nests() > 0, "Attempting to exit from a lazy mode without entering");
    return --_lazy_mode_nests();
}

// in_lazy_mode is a real API used by other parts of lazy tensor code to adjust behavior for lazy mode
bool in_lazy_mode() {
    return _lazy_mode_nests() > 0;
}

void LazyModeEnter(c10::Device device) {
    // We ignore nested lazy modes mainly to enable lazy modes being applied to small regions of library code
    // and then again around larger regions.  Only the 'outer' mode scope should cause behavior to change.
    if (_lazy_mode_inc() == 0) {
        // It is straightforward why we want to set the lazy key on entering the mode:
        // we force operators (even on regular eager tensors) to route to lazy implementations
        c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::Lazy, true);
        return;
    }
}

void LazyModeExit(c10::Device device) {
    if (_lazy_mode_dec() == 0) {
        // Equally straightforward is that we no longer want the lazy key when we exit the mode:
        // this lets operations on eager tensors outside the mode go back to normal eager behavior
        c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::Lazy, false);

        // Less obvious is that we also set an 'unlazy' key on mode exit, which lets us specially handle
        // any 'lazy' tensors that are alive after the mode exit.  This could be avoided if we can find another way
        // to make lazy tensors interoperable with eager kernels.
        // TODO(whc) PrivateUse1 is just for prototyping
        c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::PrivateUse1, true);

        // At mode exit, we use the currently 'live' lazy tensors to define a graph to compile/execute
        auto backend_device = torch::lazy::atenDeviceToBackendDevice(device);
        auto backend_devices = {backend_device.toString()};
        // wait=true: means we definitely submit all gpu work before exiting,
        // does not sync on gpu completion
        torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(&backend_device, backend_devices, /*wait=*/true);

        // Live lazy tensors should now all have eager tensors replacing their 'ir_value' fields, which can be
        // accessed by eager kernels using the 'unlazy handler'
        return;
    }
}

void unlazy_handler(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
    LOG(ERROR) << "Unlazy Handler (isn't getting called!)";
    // This function makes lazy tensors left alive after a lazy mode exit compatible with eager operations.
    // it doesn't modify the lazy tensors, so the next time they are used they still have to be "unlazy'd" again.

    // What we need to have happen: 
    // 1) iterate over the arguments on the stack, and for each lazy tensor, dig out its boxed eager tensor
    //    preparing a new stack of all eager tensors
    // 2) redispatch the op to an eager kernel using the 'eager' stack

    // For now, just call the ts_eager_fallback code, since it does (1) and (2) for us already,
    // although it may introduce extra copies we want to avoid
    ts_eager_fallback(
      op, stack, torch::lazy::getBackend()->EagerFallbackDeviceType());
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&unlazy_handler>());
}


} // namespace lazy
} // namespace torch
