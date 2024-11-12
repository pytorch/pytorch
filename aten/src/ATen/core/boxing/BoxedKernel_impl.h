#pragma once

namespace c10 {

inline BoxedKernel::BoxedKernel()
    : functor_()
, boxed_kernel_func_(nullptr)
{}

inline BoxedKernel::BoxedKernel(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func)
: functor_(std::move(functor))
, boxed_kernel_func_(boxed_kernel_func)
{}

template<BoxedKernel::BoxedKernelFunction* func>
inline void BoxedKernel::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet, Stack* stack) {
    // Note that we're dropping the DispatchKeySet argument.
    // See Note [Plumbing Keys Through The Dispatcher 2] for details.
    func(opHandle, stack);
}

template<BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline void BoxedKernel::make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, DispatchKeySet ks, Stack* stack) {
    // See Note [Plumbing Keys Through The Dispatcher 2] for details.
    func(opHandle, ks, stack);
}

inline bool BoxedKernel::isValid() const {
    return boxed_kernel_func_ != nullptr;
}

inline bool BoxedKernel::isFallthrough() const {
    return boxed_kernel_func_ == &fallthrough_kernel;
}

inline void BoxedKernel::callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        boxed_kernel_func_ != nullptr,
        "Tried to call BoxedKernel::callBoxed() on an uninitialized BoxedKernel."
    );
    (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
}

template<BoxedKernel::BoxedKernelFunction* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
    return BoxedKernel(
        nullptr,  // no functor_ object
        &make_boxed_function<func>
    );
}

template<BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
    return BoxedKernel(
        nullptr,  // no functor_ object
        &make_boxed_function<func>
    );
}

inline BoxedKernel BoxedKernel::makeFallthrough() {
    return BoxedKernel(
        nullptr,  // no functor_ object
        &fallthrough_kernel
    );
}

inline BoxedKernel BoxedKernel::makeAmbiguousAutogradOther() {
    return BoxedKernel(
        nullptr,  // no functor_ object
        &ambiguous_autogradother_kernel
    );
}

inline BoxedKernel BoxedKernel::makeNamedNotSupported() {
    return BoxedKernel(
        nullptr,  // no functor_ object
        &named_not_supported_kernel
    );
}

template<class KernelFunctor>
inline BoxedKernel BoxedKernel::makeFromFunctor(std::unique_ptr<KernelFunctor> kernelFunctor) {
    static_assert(std::is_base_of_v<OperatorKernel, KernelFunctor>, "Tried to call BoxedKernel::makeFromFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
    return BoxedKernel(
        std::move(kernelFunctor),
        [](OperatorKernel* kernel, const OperatorHandle& op, DispatchKeySet ks, Stack* stack) {
          (*static_cast<KernelFunctor*>(kernel))(op, ks, stack);
        }
    );
}

inline OperatorKernel* BoxedKernel::getFunctor() const {
  return functor_.get();
}
inline BoxedKernel::InternalBoxedKernelFunction* BoxedKernel::getFnPtr() const {
  return boxed_kernel_func_;
}

}  // namespace c10
