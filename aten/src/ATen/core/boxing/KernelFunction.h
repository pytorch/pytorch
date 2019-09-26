#pragma once

#include <ATen/core/stack.h>
#include <c10/util/TypeList.h>
#include <ATen/core/boxing/kernel_functor.h>
#include <ATen/core/boxing/kernel_function.h>
#include <ATen/core/boxing/kernel_lambda.h>

namespace c10 {

namespace detail {
template<class Return, class... Args> struct boxAndCallBoxedFunc;
}

/**
 * KernelFunction is similar to std::function but stores a kernel function.
 * You can create a KernelFunction from a boxed or unboxed function/functor/lambda
 * and call it in a boxed or unboxed way. If the way it was created doesn't
 * match the way it was called, it will do boxing or unboxing as necessary.
 */
class CAFFE2_API KernelFunction final {
public:
  using BoxedKernelFunction = void(OperatorKernel*, Stack*);

  KernelFunction()
  : functorFactory_()
  , functor_(nullptr)
  , boxed_kernel_func_(nullptr)
  , unboxed_kernel_func_(nullptr)
  {}

  bool isValid() const {
    // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this should only check boxed_kernel_func_.
    return boxed_kernel_func_ != nullptr || unboxed_kernel_func_ != nullptr;
  }

  /**
   * Call the function in a boxed way.
   * If the kernel function was created with an unboxed function,
   * this will call an unboxing wrapper which then calls into that
   * unboxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.callBoxed(stack);
   *
   * Or, with an unboxed implementation:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.callBoxed(stack);
   */
  void callBoxed(Stack* stack) const {
    if (C10_UNLIKELY(boxed_kernel_func_ == nullptr)) {
      if (unboxed_kernel_func_ == nullptr) {
        TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxed() on an uninitialized KernelFunction.");
      } else {
        // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this case should be impossible.
        TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed().");
      }
    }

    (*boxed_kernel_func_)(getFunctor_(), stack);
  }

  /**
   * Call the function in an unboxed way.
   * As the "Only" in the name suggests, this only works for KernelFunctions
   * that are backed by an unboxed kernel. If the KernelFunction was created
   * in a boxed way, this will fail (also see KernelFunction::callUnboxed()).
   *
   * KernelFunction::callUnboxed() is generally better, since it will allow
   * calling KernelFunctions that are backed by either boxed or unboxed
   * kernels, but that one will not work for all types.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.callUnboxedOnly<Tensor, Tensor, bool>(tensor1, true);
   */
  template<class Return, class... Args>
  Return callUnboxedOnly(Args... args) const {
    // note: Args above is intentionally not Args&&. We don't want perfect
    // forwarding, which would require Args to be deduced, but instead we
    // want callers to explicitly specify the Args.

    // TODO Remove this function once all kernels support a boxed variant

    if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
      using ActualSignature = Return (OperatorKernel*, Args...);
      ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func_);
      return (*func)(getFunctor_(), std::forward<Args>(args)...);
    }

    TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callUnboxedOnly() for a kernel that doesn't have an unboxed version.");
  }

  /**
   * Call the function in an unboxed way.
   * If the kernel function was created with a boxed function,
   * this will box all inputs and then call into that boxed function.
   *
   * Note that this doesn't work for all types yet.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.callUnboxed<Tensor, Tensor, bool>(tensor1, true);
   *
   * Or, with a boxed implementation:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.callUnboxed<Tensor, Tensor, bool>(tensor1, true);
   */
  template<class Return, class... Args>
  Return callUnboxed(Args... args) const {
    // note: Args above is intentionally not Args&&. We don't want perfect
    // forwarding, which would require Args to be deduced, but instead we
    // want callers to explicitly specify the Args.

    if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
      using ActualSignature = Return (OperatorKernel*, Args...);
      ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func_);
      return (*func)(getFunctor_(), std::forward<Args>(args)...);
    }

    TORCH_INTERNAL_ASSERT(boxed_kernel_func_ != nullptr, "Tried to call KernelFunction::callUnboxed() on an uninitialized KernelFunction.");
    return detail::boxAndCallBoxedFunc<Return, Args...>::call(boxed_kernel_func_, getFunctor_(), std::forward<Args>(args)...);
  }

  /**
   * Create a KernelFunction from a boxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   */
  static KernelFunction makeFromBoxedFunction(BoxedKernelFunction* func) {
    return KernelFunction(
      nullptr,  // no functorFactory_, this can only be called in a boxed way.
      nullptr,  // no functor_ object either
      func,
      nullptr  // no unboxed function pointer
    );
  }

  /**
   * Create a KernelFunction from an unboxed functor.
   *
   * Example:
   *
   * > class MyFunctor final {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunctor(std::make_shared<MyFunctor>());
   */
  template<bool AllowLegacyTypes = false, class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(std::shared_ptr<KernelFunctor> kernelFunctor) {
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
      nullptr, // no functorFactory_ because we already have the functor_
      std::move(kernelFunctor),
      &detail::wrap_kernel_functor_boxed<KernelFunctor, AllowLegacyTypes>::call,
      reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
  }

  /**
   * Create a KernelFunction from an unboxed functor and delay functor creation
   * until the first call to the KernelFunction. This is useful for functors
   * that are registered at static initialization time but can't be created
   * there yet. For example, we want to allow functors to store Tensor members
   * (we can't create Tensor objects at static initialization time because of SIOF)
   * but these functors are registered as kernels at static initialization time.
   * Using this method, we can delay functor instantiation until the operator
   * is called for the first time.
   *
   * Example:
   *
   * > class MyFunctor final {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunctor([] {
   * >   return std::make_shared<MyFunctor>();
   * > });
   */
  template<class KernelFunctor, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedFunctorFactory(std::function<std::shared_ptr<KernelFunctor>()> kernelFunctorFactory) {
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
      std::move(kernelFunctorFactory),
      nullptr, // delay creation of functor_ (it will be created by calling functorFactory_ later)
      &detail::wrap_kernel_functor_boxed<KernelFunctor, AllowLegacyTypes>::call,
      reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
  }

  /**
   * Create a KernelFunction from an unboxed functor and prevent creation of an
   * unboxing-wrapper. This means that you can only call this KernelFunction
   * using KernelFunction::callUnboxedOnly(), not using KernelFunction::callBoxed()
   * or KernelFunction::callUnboxed().
   *
   * This is necessary because our unboxing wrappers don't work for all types
   * yet, so if you want to use one of these types as function arguments,
   * you need to use makeFromUnboxedOnlyFunctor.
   *
   * Example:
   *
   * > class MyFunctor final {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor(std::make_shared<MyFunctor>());
   */
  template<class KernelFunctor>
  static KernelFunction makeFromUnboxedOnlyFunctor(std::shared_ptr<KernelFunctor> kernelFunctor) {
    // TODO We want to get rid of kernels that have only an unboxed function pointer.
    //      All kernels should have a boxed pointer.

    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
      nullptr, // no functorFactory_ because we already have the functor_
      std::move(kernelFunctor),
      nullptr, // Don't create a boxed kernel for this
      reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call)
    );
  }

  /**
   * Create a KernelFunction from an unboxed function.
   * This is usually better than KernelFunction::makeFromUnboxedRuntimeFunction
   * because knowing the function pointer as a template argument (i.e. at
   * compile time) allows the compiler to inline the function into its
   * unboxing wrapper and yields better performance when calling the function.
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(unboxed_func), &unboxed_func>();
   */
  template<class FuncType, FuncType* func, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedFunction() {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with invalid template parameters. They must be <FuncType, *func_ptr>.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    static_assert(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes>(
      std::make_shared<typename detail::WrapKernelFunction<FuncType, func>::type>()
    );
  }

  /**
   * Create a KernelFunction from an unboxed function and prevent creation of an
   * unboxing-wrapper. This means that you can only call this KernelFunction
   * using KernelFunction::callUnboxedOnly(), not using KernelFunction::callBoxed()
   * or KernelFunction::callUnboxed().
   *
   * This is necessary because our unboxing wrappers don't work for all types
   * yet, so if you want to use one of these types as function arguments,
   * you need to use makeFromUnboxedOnlyFunctor.
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction<decltype(unboxed_func), &unboxed_func>();
   */
  template<class FuncType, FuncType* func>
  static KernelFunction makeFromUnboxedOnlyFunction() {
    // TODO We want to get rid of kernels that have only an unboxed function pointer.
    //      All kernels should have a boxed pointer.

    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedOnlyFunction with invalid template parameters. They must be <FuncType, *func_ptr>.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedOnlyFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    static_assert(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedOnlyFunctor(
      std::make_shared<typename detail::WrapKernelFunction<FuncType, func>::type>()
    );
  }

  /**
   * Create a KernelFunction from an unboxed function.
   * KernelFunction::makeFromUnboxedFunction is usually a better choice than
   * this if you know the function pointer at compile time, see doc comment
   * there for an explanation.
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&unboxed_func);
   */
  template<bool AllowLegacyTypes = false, class FuncType>
  static KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes>(
      std::make_shared<detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(func)
    );
  }

  template<class FuncType>
  static KernelFunction makeFromUnboxedOnlyRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a non-function type.");
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, "Tried to call KernelFunction::makeFromUnboxedRuntimeFunction with a boxed function pointer. Please use KernelFunction::makeFromBoxedFunction instead.");
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedOnlyFunctor(
      std::make_shared<detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(func)
    );
  }

  /**
   * Create a KernelFunction from an unboxed lambda.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   */
  template<bool AllowLegacyTypes = false, class Lambda>
  static KernelFunction makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<guts::decay_t<Lambda>>::value, "Tried to call KernelFunction::makeFromUnboxedLambda with a non-lambda type.");

    return makeFromUnboxedFunctor<AllowLegacyTypes>(
      std::make_shared<detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
  }

private:

  explicit KernelFunction(std::function<std::shared_ptr<OperatorKernel>()> functorFactory, std::shared_ptr<OperatorKernel> functor, BoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func)
  : functorFactory_(std::move(functorFactory))
  , functor_(std::move(functor))
  , boxed_kernel_func_(boxed_kernel_func)
  , unboxed_kernel_func_(unboxed_kernel_func)
  {}

  OperatorKernel* getFunctor_() const {
    if (functor_.get() == nullptr) {
      if (!functorFactory_) {
        return nullptr;
      }
      functor_ = functorFactory_();
    }
    return functor_.get();
  }

  // If the operator has an unboxed_kernel_func, then either
  // functorFactory_ or functor_ must be set, possibly both.
  // If functor_ is not set but functorFactory_ is, we will create
  // functor_ by calling functorFactory_ the first time it is needed.
  // We use this indirection because many KernelFunctions are created
  // at static initialization time but are created with functors that
  // store Tensor and we can't call the Tensor() constructor at static
  // initialization time yet (SIOF). So these register with a
  // functorFactory_ instead of a functor_ and will be initialized
  // on the first call to the KernelFunction.
  std::function<std::shared_ptr<OperatorKernel>()> functorFactory_;
  mutable std::shared_ptr<OperatorKernel> functor_;

  BoxedKernelFunction* boxed_kernel_func_;
  void* unboxed_kernel_func_;
};

namespace detail {
template<class Return, class... Args>
struct boxAndCallBoxedFunc final {
  static Return call(KernelFunction::BoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, Args... args) {
    // TODO Reuse stack vector instead of allocating?
    std::vector<IValue> stack {std::forward<Args>(args)...};

    (*boxed_kernel_func)(functor, &stack);

    TORCH_INTERNAL_ASSERT(stack.size() == 1, "A boxed kernel should only push one return to the stack");
    return std::move(stack[0]).to<Return>();
  }
};
template<class... Args>
struct boxAndCallBoxedFunc<void, Args...> final {
  static void call(KernelFunction::BoxedKernelFunction* boxed_kernel_func, OperatorKernel* functor, Args... args) {
    // TODO Reuse stack vector instead of allocating?
    std::vector<IValue> stack {std::forward<Args>(args)...};

    (*boxed_kernel_func)(functor, &stack);

    TORCH_INTERNAL_ASSERT(stack.size() == 0, "A boxed kernel returned a value but when we called it with KernelFunction::callUnboxed, we expected it to return void.");
  }
};
}

}
