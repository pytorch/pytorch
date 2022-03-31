#include <torch/csrc/lazy/core/lazy_mode.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/library.h>

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

// utility used for lazy_mode and ts_eager_fallback
// TODO(whc) surely, there is such a utility in core already? I moved this from ts_backend code
c10::DispatchKey device_to_dispatch_key(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCPU: {
      return c10::DispatchKey::CPU;
    }
    case at::kCUDA: {
      return c10::DispatchKey::CUDA;
    }
    default: {
      AT_ERROR("Unsupported device type: ", device_type);
    }
  }
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
        // For now, it is set on all LTCTensorImpls by their ctor, and then behaves as a no-op if inside lazy mode

        // At mode exit, we use the currently 'live' lazy tensors to define a graph to compile/execute
        auto backend_device = atenDeviceToBackendDevice(c10::Device("lazy:0"));

        // TODO(whc) sync all devices since there is currently a bug where the device i passed here didn't match
        // the one that the tensors were created on
        std::vector<BackendDevice> backend_devices = {};
        std::vector<std::string> backend_device_strs = {};
        // wait=true: means we definitely submit all gpu work before exiting,
        // does not sync on gpu completion
        torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(/*backend_device=*/nullptr, backend_device_strs, /*wait=*/true);

        // Live lazy tensors should now all have eager tensors replacing their 'ir_value' fields, which can be
        // accessed by eager kernels using the 'unlazy handler'
        return;
    }
}

c10::DispatchKey GetUnlazyDispatchKey() {
    return c10::DispatchKey::TESTING_ONLY_GenericWrapper;
}

at::Tensor PrepareTensorForMetaKernel(at::Tensor tensor, BackendDevice lazy_device) {
    if (!in_lazy_mode()) {
        // This function is only useful for lazy mode, but its called all the time currently,
        // so at least make it a no-op with an assert for non-lazy-mode
        TORCH_INTERNAL_ASSERT(tensor.device().type() == c10::kLazy);
        return tensor;
    }
    // before calling meta kernels, we need to make sure all tensors are on the same device
    if(tensor.device().type() == c10::kLazy) {
        LOG(INFO) << "PrepareTensorForMetaKernel skip move for already-lazy tensor on " << tensor.device();
        TORCH_INTERNAL_ASSERT(!tensor.device().has_index());
        return tensor;
    } else {
        TORCH_INTERNAL_ASSERT(tensor.device().type() != c10::kLazy);
        LOG(INFO) << "PrepareTensorForMetaKernel move tensor from " << tensor.device() << " to " << lazy_device;
        // TODO: cache these so we only have to do each wrapping once

        // This was going down a confusing path of redispatching. Why bother, when I can just do this:
        // return tensor.to(lazy_device);
        return CreateAtenFromLtcTensor(GetOrCreateLtcTensor(tensor, lazy_device));
    }
}

at::Tensor unwrap_materialized_lazy_tensor(at::Tensor tensor) {
    auto lazy_tensor = GetLtcTensor(tensor);
    // This is a specialized helper for lazy mode, because it makes more assumptions (and checks them)
    // than the usual 'ToTensor' method which would happily compile/materialize an IR-backed tensor
    TORCH_CHECK(lazy_tensor->CurrentDataHandle(), "Tensor should have been materialized by mode exit");
    TORCH_CHECK(!lazy_tensor->CurrentIrValue(), "Materialized tensor should not have an IR value");

    // For the TS backend, 'MakeTensorFromComputationData' amounts to a cast,
    // but for other backends this API could perform a copy or a transfer as needed
    auto unwrapped_tensor = getBackend()->MakeTensorFromComputationData(
        lazy_tensor->CurrentDataHandle(),
        /*logical_scalar_type=*/c10::nullopt);
    
    // We want to make sure that all the unwrapped tensors are on the same device,
    // And we want it to be the efficient device for that backend, which at least for TS backend
    // is the device that EagerFallbackDeviceType reports
    TORCH_CHECK(unwrapped_tensor.device().type() == getBackend()->EagerFallbackDeviceType());
    return unwrapped_tensor;
}

void unlazy_handler(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
    if (in_lazy_mode()) {
        LOG(INFO) << "unlazy_handler is a no-op inside lazy mode";
        op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Lazy), stack);
        return;
    }

    LOG(INFO) << "unlazy_handler is kicking in outside lazy mode";
    // This function makes lazy tensors left alive after a lazy mode exit compatible with eager operations.
    // it doesn't modify the lazy tensors, so the next time they are used they still have to be "unlazy'd" again.

    // Avoid death by infinite recursion
    c10::impl::ExcludeDispatchKeyGuard no_recursive_unlazy(GetUnlazyDispatchKey());

    // 1) iterate over the arguments on the stack, and for each lazy tensor, dig out its boxed eager tensor
    //    preparing a new stack of all eager tensors
    auto& schema_args = op.schema().arguments();
    const auto num_arguments = schema_args.size();
    auto arguments = torch::jit::last(stack, num_arguments);
    const auto arguments_begin = stack->size() - num_arguments;
    for (int64_t idx = 0; idx < arguments.size(); ++idx) {
        const auto& ivalue = arguments[idx];
        if (ivalue.isTensor()) {
            auto lazy_tensor_arg = ivalue.toTensor();
            if (lazy_tensor_arg.device().type() == at::kLazy) {
                (*stack)[arguments_begin + idx] = unwrap_materialized_lazy_tensor(lazy_tensor_arg);
            }
        } else if (ivalue.isTensorList()) {
            // TODO(whc) handle tensorlist, something like this, calling
            // unwrap_materialized_lazy_tensor on each lazy tensor in the list:
            // (*stack)[arguments_begin + idx] = c10::IValue(c10::List<at::Tensor>(...
        }
    }

    // 2) redispatch the op to an eager kernel using the 'eager' stack
    op.redispatchBoxed(c10::DispatchKeySet(device_to_dispatch_key(getBackend()->EagerFallbackDeviceType())), stack);
}

TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&unlazy_handler>());
}


} // namespace lazy
} // namespace torch
