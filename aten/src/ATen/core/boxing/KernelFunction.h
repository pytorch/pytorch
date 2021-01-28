#pragma once

#include <ATen/core/stack.h>
#include <c10/util/TypeList.h>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.

class OperatorHandle;
struct OperatorKernel;

// This kernel implements the behavior of falling through to the next available
// registered dispatch key.  The implementation of this function is FAST; it is
// no overhead to fallthrough to the next key.  See cpp file for some more
// implementation notes; notably, this does NOT actually go through the
// boxing/unboxing codepath.
TORCH_API void fallthrough_kernel(OperatorKernel*, const OperatorHandle&, Stack*);

// Note [Ambiguity in AutogradOther kernel]
// This kernel implements reporting an error message when there're kernels registered
// to both Math and a backend of AutogradOther, we don't know which kernel to pick:
// - if we pick Math kernel for AutogradOther, the kernel registered to backend will be
//   silently ignored and never called.
// - if we skip using Math kernel for AutogradOther (it might pick Autograd kernel if available),
//   it'll break all backends mapped to AutogradOther without a direct registration to backend.
//   See c10/core/DispatchKeySet.cpp for a list of backends mapped to AutogradOther.
// Thus if backend extender indeed want to override Math kernel behavior, they should request
// a dedicated Autograd key for their backend to resolve the ambiguity.
TORCH_API void ambiguous_autogradother_kernel(OperatorKernel*, const OperatorHandle&, Stack*);

// Note [named_not_supported_kernel]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This kernel implements reporting an error message saying that named tensor is
// not supported.  This kernel doesn't rely on the Stack, and so it is special
// cased in the dispatcher to be triggered before we attempt boxing (so we can
// give a good error message in cases when boxing is not supported).  When
// boxing is universally supported this can be removed.
[[noreturn]] TORCH_API void named_not_supported_kernel(OperatorKernel*, const OperatorHandle&, Stack*);

/**
 * KernelFunction is similar to std::function but stores a kernel function.
 * You can create a KernelFunction from a boxed or unboxed function/functor/lambda
 * and call it in a boxed or unboxed way. If the way it was created doesn't
 * match the way it was called, it will do boxing or unboxing as necessary.
 */
class TORCH_API KernelFunction final {
public:
  // This is how boxed kernels are actually stored
  using InternalBoxedKernelFunction = void(OperatorKernel*, const OperatorHandle&, Stack*);
  // This is the public API for how boxed kernels are defined
  using BoxedKernelFunction = void(const OperatorHandle&, Stack*);

  KernelFunction();

  bool isValid() const;
  bool isFallthrough() const;

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
  void callBoxed(const OperatorHandle& opHandle, Stack* stack) const;

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
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   *
   * Or, with a boxed implementation:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   */
  template<class Return, class... Args>
  Return call(const OperatorHandle& opHandle, Args... args) const;

  /**
   * Create a KernelFunction from a boxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction<&boxed_func>();
   */
  template<BoxedKernelFunction* func>
  static KernelFunction makeFromBoxedFunction();

  /**
   * Create a KernelFunction from an unboxed functor.
   *
   * Example:
   *
   * > class MyFunctor final {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunctor(std::make_unique<MyFunctor>());
   */
  template<bool AllowLegacyTypes = false, class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor);

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
  template<class FuncPtr, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedFunction(FuncPtr);

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
  static KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func);

  static KernelFunction makeFallthrough();
  static KernelFunction makeAmbiguousAutogradOther();
  static KernelFunction makeNamedNotSupported();

  /**
   * Create a KernelFunction from an unboxed lambda.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   */
  template<bool AllowLegacyTypes = false, class Lambda>
  static std::enable_if_t<guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda);
  template<bool AllowLegacyTypes = false, class Lambda>
  static std::enable_if_t<!guts::is_stateless_lambda<std::decay_t<Lambda>>::value, KernelFunction> makeFromUnboxedLambda(Lambda&& lambda);

  std::string dumpState() const;
  // For testing internal invariants only
  bool _equalsBoxedAndUnboxed(const KernelFunction&) const;

private:

  explicit KernelFunction(std::unique_ptr<OperatorKernel> functor, InternalBoxedKernelFunction* boxed_kernel_func, void* unboxed_kernel_func);

  template<BoxedKernelFunction* func>
  static void make_boxed_function(OperatorKernel*, const OperatorHandle& opHandle, Stack* stack);

  OperatorKernel* getFunctor_() const;

  std::shared_ptr<OperatorKernel> functor_;

  InternalBoxedKernelFunction* boxed_kernel_func_;
  void* unboxed_kernel_func_;
};

}

#include <ATen/core/boxing/KernelFunction_impl.h>
