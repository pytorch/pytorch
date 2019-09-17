#pragma once

#include <ATen/core/dispatch/KernelCache.h>
#include <ATen/core/stack.h>
#include <c10/util/TypeList.h>
#include <ATen/core/op_registration/kernel_functor.h>
#include <ATen/core/op_registration/kernel_function.h>
#include <ATen/core/op_registration/kernel_lambda.h>

namespace c10 {

namespace detail {
template<class Type>
constexpr uint64_t hashType() {
  // TODO Also consider modifiers like const/volatile/reference/rvalue_ref/...
  return typeid(Type).hash_code();
}
template<class TypeList> struct hashTypeList_ final {};
template<class Head, class... Tail>
struct hashTypeList_<guts::typelist::typelist<Head, Tail...>> final {
  static constexpr uint64_t call(uint64_t index) {
    return index * hashType<Head>() + hashTypeList_<guts::typelist::typelist<Tail...>>::call(index + 1);
  }
};
template<>
struct hashTypeList_<guts::typelist::typelist<>> final {
  static constexpr uint64_t call(uint64_t index) {
    return 0;
  }
};

template<class TypeList>
constexpr uint64_t hashTypeList() {
  return hashTypeList_<TypeList>::call(1);
}

template<class FuncSignature>
constexpr uint64_t hashFunctionSignature() {
  using func_traits = guts::infer_function_traits_t<FuncSignature>;
  return hashTypeList<
    guts::typelist::concat_t<
      guts::typelist::typelist<typename func_traits::return_type>,
      typename func_traits::parameter_types
    >>();
}
}

class CAFFE2_API KernelFunction final {
public:
  using BoxedKernelFunction = void(Stack*, KernelCache*); // TODO Switch argument order, KernelCache first, and use OperatorKernel instead.

  KernelFunction()
  : functor_()
  , boxed_kernel_func_(nullptr)
  , unboxed_kernel_func_(nullptr)
  , signature_hash_(c10::nullopt)
  {}

  bool isValid() const {
    // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this should only check boxed_kernel_func_.
    return boxed_kernel_func_ != nullptr || unboxed_kernel_func_ != nullptr;
  }

  void callBoxed(Stack* stack) const {
    if (C10_UNLIKELY(boxed_kernel_func_ == nullptr)) {
      if (unboxed_kernel_func_ == nullptr) {
        TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxed() on an uninitizliaed KernelFunction.");
      } else {
        // TODO We want to introduce the invariant that all kernels must be callable in a boxed way, then this case should be impossible.
        TORCH_INTERNAL_ASSERT(false, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed().");
      }
    }

    (*boxed_kernel_func_)(stack, functor_.get());
  }

  template<class Return, class... Args>
  Return callUnboxed(Args... args) const {
    TORCH_INTERNAL_ASSERT(!signature_hash_.has_value() || (detail::hashFunctionSignature<Return (Args...)>() == *signature_hash_),
      "Called KernelFunction::callUnboxed with wrong argument types");

    if (unboxed_kernel_func_ != nullptr) {
      using ActualSignature = Return (c10::KernelCache*, Args...);
      ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func_);
      return (*func)(functor_.get(), std::forward<Args>(args)...);
    }

    TORCH_INTERNAL_ASSERT(false, "calling boxed kernels with callUnboxed() not implemented yet");
    // TODO return boxAndCallBoxedFunc_<Return, Args...>(std::forward<Args>(args)...);
  }

  static KernelFunction makeFromBoxedFunction(BoxedKernelFunction* func) {
    return KernelFunction(
      nullptr,  // no functor object
      func,
      nullptr,  // no unboxed function pointer
      c10::nullopt  // signature is not known, we can't error check unboxed calls.
    );
  }

  template<bool AllowLegacyTypes = false, class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(std::shared_ptr<KernelFunctor> kernelFunctor) {
    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
      std::move(kernelFunctor),
      &detail::wrap_kernel_functor_boxed<KernelFunctor, AllowLegacyTypes>::call,
      reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call),
      detail::hashFunctionSignature<KernelFunctor>()
    );
  }

  template<class KernelFunctor>
  static KernelFunction makeFromUnboxedOnlyFunctor(std::shared_ptr<KernelFunctor> kernelFunctor) {
    // TODO We want to get rid of kernels that have only an unboxed function pointer.
    //      All kernels should have a boxed pointer.

    static_assert(guts::is_functor<KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor> but the argument is not a functor.");
    static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to call KernelFunction::makeFromUnboxedFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");

    return KernelFunction(
      std::move(kernelFunctor),
      nullptr, // Don't create a boxed kernel for this
      reinterpret_cast<void*>(&detail::wrap_kernel_functor_unboxed<KernelFunctor>::call),
      detail::hashFunctionSignature<KernelFunctor>()
    );
  }

  template<class FuncType, FuncType* func, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedFunction() {
    static_assert(guts::is_function_type<FuncType>::value, ""); // TODO
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, ""); // TODO
    static_assert(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes>(
      std::make_shared<typename detail::WrapKernelFunction<FuncType, func>::type>()
    );
  }

  template<class FuncType, FuncType* func>
  static KernelFunction makeFromUnboxedOnlyFunction() {
    // TODO We want to get rid of kernels that have only an unboxed function pointer.
    //      All kernels should have a boxed pointer.

    static_assert(guts::is_function_type<FuncType>::value, ""); // TODO
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, ""); // TODO
    static_assert(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedOnlyFunctor(
      std::make_shared<typename detail::WrapKernelFunction<FuncType, func>::type>()
    );
  }

  template<class FuncType, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func) {
    static_assert(guts::is_function_type<FuncType>::value, ""); // TODO
    static_assert(!std::is_same<FuncType, BoxedKernelFunction>::value, ""); // TODO
    TORCH_INTERNAL_ASSERT(func != nullptr, "Kernel function cannot be nullptr");

    return makeFromUnboxedFunctor<AllowLegacyTypes>(
      std::make_shared<detail::WrapRuntimeKernelFunctor<guts::decay_t<FuncType>>>(func)
    );
  }

  template<class Lambda, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedLambda(Lambda&& lambda) {
    static_assert(guts::is_functor<Lambda>::value, ""); // TODO
    static_assert(guts::is_stateless_lambda<guts::decay_t<Lambda>>::value, ""); // TODO

    return makeFromUnboxedFunctor<AllowLegacyTypes>(
      std::make_shared<detail::WrapRuntimeKernelFunctor<guts::decay_t<Lambda>>>(std::forward<Lambda>(lambda))
    );
  }

private:

  template<class Return, class... Args>
  Return boxAndCallBoxedFunc_(Args... args) const {
    TORCH_INTERNAL_ASSERT(boxed_kernel_func_ != nullptr, "Tried to call KernelFunction::callUnboxed() on an uninitizliaed KernelFunction.");

    // TODO Reuse stack vector instead of allocating?
    std::vector<IValue> stack {std::forward<Args>(args)...};

    (*boxed_kernel_func_)(&stack, functor_.get());

    TORCH_INTERNAL_ASSERT(stack.size() == 1, "A boxed kernel should only push one return to the stack");
    return std::move(stack[0]).to<Return>();
  }

  explicit KernelFunction(std::shared_ptr<OperatorKernel> functor, BoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func, c10::optional<uint64_t> signature_hash)
  : functor_(std::move(functor))
  , boxed_kernel_func_(boxed_kernel_func)
  , unboxed_kernel_func_(unboxed_kernel_func)
  , signature_hash_(signature_hash)
  {}

  std::shared_ptr<OperatorKernel> functor_;
  BoxedKernelFunction* boxed_kernel_func_;
  void* unboxed_kernel_func_;

  // signature_hash_ is set to the hash of the function signature if the
  // KernelFunction was created in a way that allowed us to know the function
  // signature. If this is set, it will be used in unboxed function calls
  // to verify their arguments against the known function signature.
  c10::optional<uint64_t> signature_hash_;
};

}
