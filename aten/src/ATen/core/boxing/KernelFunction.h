#pragma once

#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/TypeList.h>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack to the c10 namespace.

class OperatorHandle;
struct OperatorKernel;
class KernelFunction;

/**
 * KernelFunction is similar to std::function but stores a kernel function.
 * You can create a KernelFunction from a boxed or unboxed function/functor/lambda
 * and call it in a boxed or unboxed way. If the way it was created doesn't
 * match the way it was called, it will do boxing or unboxing as necessary.
 */
class TORCH_API KernelFunction final {
public:

  using InternalBoxedKernelFunction = BoxedKernel::InternalBoxedKernelFunction;
  using BoxedKernelFunction = BoxedKernel::BoxedKernelFunction;
  using BoxedKernelFunction_withDispatchKeys = BoxedKernel::BoxedKernelFunction_withDispatchKeys;

  KernelFunction();

  // Fast path for dispatch to allow not touching the boxed kernel in
  // the common case where unboxed is available.
  bool isValidUnboxed() const;
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
  void callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const;

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
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const;

  /**
   * Create a KernelFunction from a BoxedKernel.
   */
  static KernelFunction makeFromBoxedKernel(BoxedKernel boxed_fn);

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
   * TODO: This will only be useful if we write a backend fallback that plumbs dispatch keys (currently there are none)
   * See Note [Plumbing Keys Through The Dispatcher] for details.
   */
  template<BoxedKernelFunction_withDispatchKeys* func>
  static KernelFunction makeFromBoxedFunction();

  /**
   * Create a KernelFunction from an unboxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromUnboxedFunctor<MyFunctor>(std::make_unique<MyFunctor>());
   */
  template<bool AllowLegacyTypes = false, class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor);

  /**
   * Create a KernelFunction from a boxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     void operator()(const OperatorHandle&, DispatchKeySet, Stack*) {...}
   * > };
   * > KernelFunction func = KernelFunction::makeFromBoxedFunctor(std::make_unique<MyFunctor>());
   */
  template<class KernelFunctor>
  static KernelFunction makeFromBoxedFunctor(std::unique_ptr<KernelFunctor> kernelFunctor);

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

  explicit KernelFunction(
      std::unique_ptr<OperatorKernel> functor,
      InternalBoxedKernelFunction* boxed_kernel_func,
      void* unboxed_kernel_func);
  explicit KernelFunction(
      BoxedKernel boxed_fn,
      void* unboxed_kernel_func);

  BoxedKernel boxed_kernel_func_;
  void* unboxed_kernel_func_;
};

}

#include <ATen/core/boxing/KernelFunction_impl.h>
