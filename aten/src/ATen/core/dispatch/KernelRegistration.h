#pragma once

#include <c10/util/Optional.h>
#include <ATen/core/dispatch/Dispatcher.h>

/**
 * To register your own kernel for an operator, do in one (!) cpp file:
 *   C10_REGISTER_KERNEL(OperatorHandle)
 *      .kernel<decltype(&kernel_func), &kernel_func>()
 *      .dispatchKey(dispatch_key);
 *
 * Example:
 *
 *  Tensor my_kernel_cpu(Tensor in) {...}
 *
 *  C10_REGISTER_KERNEL(MyOpSchema)
 *      .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>()
 *      .dispatchKey(CPUTensorId());
 */

namespace c10 {

// TODO Test different order for builder
// TODO Test no dispatch key defined

/**
 * Class which, on construction, registers an operator in the dispatch table. The intent is that
 * this class is constructed at static initialization time so that operators automatically get
 * registered when a dlopen() occurs.
 *
 * You shouldn't call this directly; instead, use the C10_REGISTER_KERNEL macros.
 */
class KernelRegistrar final {
public:
  using OpHandleGetter = const OperatorHandle& ();

  /**
   * @param op The operator to register the kernel for
   * @param dispatch_key  The dispatch key to register the function to
   * @param kernel The concrete function implementation to register
   * @param cache_creator A function initializing the cache for the kernel
   */
  explicit KernelRegistrar(OpHandleGetter *op, TensorTypeId dispatch_key, KernelFunction* kernel, KernelCacheCreatorFunction* cache_creator)
  : op_(std::move(op)), dispatch_key_(std::move(dispatch_key)), owns_registration_(true) {
    Dispatcher::singleton().registerKernel(op_(), dispatch_key_, kernel, cache_creator);
  }

  KernelRegistrar(KernelRegistrar&& rhs)
  : op_(std::move(rhs.op_)), dispatch_key_(std::move(rhs.dispatch_key_)), owns_registration_(true) {
    rhs.owns_registration_ = false;
  }

  // not needed for now
  KernelRegistrar& operator=(KernelRegistrar&& rhs) = delete;

  ~KernelRegistrar() {
    if (owns_registration_) {
      Dispatcher::singleton().deregisterKernel(op_(), dispatch_key_);
    }
  }

private:
  OpHandleGetter *op_;
  const TensorTypeId dispatch_key_;
  bool owns_registration_;

  C10_DISABLE_COPY_AND_ASSIGN(KernelRegistrar);
};

namespace detail {
// ivalue_to_arg_type<T>: Take an IValue that is an argument to a kernel and
// cast it to the type that should be passed to the kernel function.
// Examples: If the IValue contains a plain type like an int, return that.
//           If the IValue contains an IntList, return it as ArrayRef<int>.
template<class T>
struct ivalue_to_arg_type {
  static T call(const IValue& v) {
    return std::move(v).to<T>();
  }
};
template<class T>
struct ivalue_to_arg_type<ArrayRef<T>> {
  static ArrayRef<T> call(const IValue& v) {
    return v.to<intrusive_ptr<ivalue::List<T>>>()->elements();
  }
};

// call_with_ivalue_args: Take a function pointer and an ArrayRef<IValue>
// containing the arguments to call the function pointer with, and call it.
// The extra_args are appended as additional arguments at the end of the function call.
// Example:
// int myfunc(int a, ArrayRef<int> b, string c);
// int main() {
//   std::vector<IValue> ivalue_args = {IValue(2), IntList::create(3, 4)};
//   call_with_ivalue_args<decltype(myfunc), &myfunc>(ivalue_args, "extra_arg");
// }
template<class FuncType, FuncType* func, class... ExtraArgs, size_t... ivalue_arg_indices>
typename guts::function_traits<FuncType>::return_type call_with_ivalue_args_(ArrayRef<IValue> ivalue_args, guts::index_sequence<ivalue_arg_indices...>, ExtraArgs&&... extra_args) {
  using IValueArgTypes = typename guts::function_traits<FuncType>::parameter_types;
  return (*func)(ivalue_to_arg_type<guts::remove_cv_t<guts::remove_reference_t<guts::typelist::element_t<ivalue_arg_indices, IValueArgTypes>>>>::call(ivalue_args[ivalue_arg_indices])..., std::forward<ExtraArgs>(extra_args)...);
}

template<class FuncType, FuncType* func, class... ExtraArgs>
typename guts::function_traits<FuncType>::return_type call_with_ivalue_args(ArrayRef<IValue> ivalue_args, ExtraArgs&&... extra_args) {
  constexpr size_t num_ivalue_args = guts::function_traits<FuncType>::number_of_parameters - sizeof...(ExtraArgs);
  AT_ASSERTM(num_ivalue_args == ivalue_args.size(), "Wrong number of ivalue arguments");
  return call_with_ivalue_args_<FuncType, func>(ivalue_args, guts::make_index_sequence<num_ivalue_args>(), std::forward<ExtraArgs>(extra_args)...);
}

template<class OutputType>
struct push_outputs final {
  static void call(OutputType&& output, Stack* stack) {
    push_outputs<std::tuple<OutputType>>(std::tuple<OutputType>(std::move(output)), stack);
  }
};
template<class... OutputTypes>
struct push_outputs<std::tuple<OutputTypes...>> final {
  static void call(std::tuple<OutputTypes...>&& output, Stack* stack) {
    for (size_t i = 0; i < sizeof...(OutputTypes); ++i) {
      torch::jit::push(return_type_to_ivalue(std::move(output)));
    }
  }
};

// SFINAE over (1) does the operator kernel have a cache and (2) does it return a value or void
template<class CacheTypeOrVoid, class FuncType, FuncType* kernel, class Enable = void> struct wrap_kernel {};
// SFINAE version for kernels with output and with cache
template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<!std::is_same<void, CacheTypeOrVoid>::value && !std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(Stack* stack, KernelCache* cache) {
    constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters - 1; // -1 because it takes the kernel cache as last argument
    auto output = call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs), static_cast<CacheTypeOrVoid*>(cache));
    push_outputs<typename guts::function_traits<FuncType>::return_type>(std::move(output), stack);
  }
};
// SFINAE version for kernels with output and without a cache
template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<std::is_same<void, CacheTypeOrVoid>::value && !std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(Stack* stack, c10::KernelCache* /*cache*/) {
    constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters;
    auto output = call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs));
    push_outputs<typename guts::function_traits<FuncType>::return_type>(std::move(output), stack);
  }
};
// SFINAE version for kernels without output and with a cache
template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<!std::is_same<void, CacheTypeOrVoid>::value && std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(Stack* stack, c10::KernelCache* cache) {
    constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters - 1; // -1 because it takes the kernel cache as last argument
    call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs), static_cast<CacheTypeOrVoid*>(cache));
  }
};
// SFINAE version for kernels without output and without a cache
template<class CacheTypeOrVoid, class FuncType, FuncType* kernel>
struct wrap_kernel<CacheTypeOrVoid, FuncType, kernel, guts::enable_if_t<std::is_same<void, CacheTypeOrVoid>::value && std::is_same<void, typename guts::function_traits<FuncType>::return_type>::value>> final {
  static typename guts::function_traits<FuncType>::return_type call(Stack* stack, c10::KernelCache* /*cache*/) {
    constexpr size_t num_inputs = guts::function_traits<FuncType>::number_of_parameters;
    call_with_ivalue_args<FuncType, kernel>(torch::jit::last(*stack, num_inputs));
  }
};
}

/**
 * Helper class for building a KernelRegistrar.  This permits "keyword-argument" like syntax
 * when performing operator registration, e.g., as in:
 *
 * C10_REGISTER_KERNEL(::ops::add_notensor)
 *      .kernel(&add_notensor_op)
 *      .dispatchKey("bla");
 *
 * Expanded, this macro invocation looks like:
 *
 * static KernelRegistrar<::ops::add_notensor> _anon0 =
 *    KernelRegistrationBuilder<::ops::add_notensor, false, false>()
 *      .kernel(&add_notensor_op)
 *      .dispatchKey("bla");
 *
 * The resulting full expression is implicitly convertible to a KernelRegistrar.
 */
template<class CacheTypeOrVoid, uint64_t FieldsPresentFlags>
class KernelRegistrationBuilder final {
private:
  static constexpr uint64_t DISPATCH_KEY_PRESENT = 0x01 << 0;
  static constexpr uint64_t KERNEL_PRESENT = 0x01 << 1;
  static constexpr uint64_t CACHE_PRESENT = 0x01 << 2;

  using OpHandleGetter = KernelRegistrar::OpHandleGetter;

  static std::unique_ptr<c10::KernelCache> defaultCacheCreator() {
    return nullptr;
  }

  template<class Cache>
  static std::unique_ptr<c10::KernelCache> cacheCreator() {
    static_assert(std::is_default_constructible<Cache>::value, "Cache class must be default constructible");
    return guts::make_unique<Cache>();
  }

  OpHandleGetter *op_;
  c10::optional<TensorTypeId> dispatch_key_;
  KernelFunction* kernel_;
  KernelCacheCreatorFunction* cache_creator_;

 public:
  constexpr explicit KernelRegistrationBuilder(OpHandleGetter *op)
      : KernelRegistrationBuilder(std::move(op), c10::nullopt, nullptr, &defaultCacheCreator) {}

  constexpr explicit KernelRegistrationBuilder(
      OpHandleGetter *op,
      c10::optional<TensorTypeId> dispatch_key,
      KernelFunction* kernel,
      KernelCacheCreatorFunction* cache_creator)
      : op_(std::move(op)), dispatch_key_(std::move(dispatch_key)), kernel_(kernel), cache_creator_(cache_creator)  {}

  /**
   * Implicit coercion to KernelRegistrar that finalizes the builder and
   * creates the object.
   * @return Produced KernelRegistrar
   */
  operator KernelRegistrar() && {
    static_assert(FieldsPresentFlags & KERNEL_PRESENT, "Forgot to call .kernel() in kernel registration");
    static_assert(FieldsPresentFlags & DISPATCH_KEY_PRESENT, "Forgot to call .dispatchKey() in kernel registration");
    return KernelRegistrar(op_, std::move(*dispatch_key_), kernel_, cache_creator_);
  }

  /**
   * Specify the dispatch key for this dispatch registration
   * @param dispatch_key dispatch key to register the function to
   * @return "this" for method chaining
   */
  AT_CPP14_CONSTEXPR KernelRegistrationBuilder<CacheTypeOrVoid, FieldsPresentFlags | DISPATCH_KEY_PRESENT> dispatchKey(TensorTypeId dispatch_key) && {
    static_assert(!(FieldsPresentFlags & DISPATCH_KEY_PRESENT), "Tried to define kernel twice in same op registration");
    return KernelRegistrationBuilder<CacheTypeOrVoid, FieldsPresentFlags | DISPATCH_KEY_PRESENT>(std::move(op_), std::move(dispatch_key), kernel_, cache_creator_);
  }

  /**
   * Specify the concrete function implementation for this dispatch registration
   * @param kernel concrete function implementation to be registered
   * @return "this" for method chaining
   */
  template<KernelFunction* kernel_func>
  AT_CPP14_CONSTEXPR KernelRegistrationBuilder<CacheTypeOrVoid, FieldsPresentFlags | KERNEL_PRESENT> kernel() && {
    static_assert(!(FieldsPresentFlags & KERNEL_PRESENT), "Tried to define kernel twice in same op registration");
    // TODO Better error message when kernel function mismatches, one common mismatch is missing cache parameter or cache parameter present while not expected.
    return KernelRegistrationBuilder<CacheTypeOrVoid, FieldsPresentFlags | KERNEL_PRESENT>(std::move(op_), std::move(dispatch_key_), kernel_func, cache_creator_);
  }

  /**
   * Specify the concrete function implementation for this dispatch registration
   * @param kernel concrete function implementation to be registered
   * @return "this" for method chaining
   */
  template<class FuncType, FuncType* kernel_func>
  AT_CPP14_CONSTEXPR KernelRegistrationBuilder<CacheTypeOrVoid, FieldsPresentFlags | KERNEL_PRESENT> kernel() && {
    // TODO Better error message if FuncType is not a func type
    return std::move(*this).template kernel<&detail::wrap_kernel<CacheTypeOrVoid, FuncType, kernel_func>::call>();
  }

  /**
   * Specify the dispatch key for this dispatch registration
   * @param dispatch_key dispatch key to register the function to
   * @return "this" for method chaining
   */
  template<class Cache>
  AT_CPP14_CONSTEXPR KernelRegistrationBuilder<Cache, FieldsPresentFlags | CACHE_PRESENT> withCache() && {
    static_assert(!(FieldsPresentFlags & CACHE_PRESENT), "Tried to define cache twice in same op registration");
    static_assert(std::is_base_of<c10::KernelCache, Cache>::value, "Cache must inherit from c10::KernelCache");

    static_assert(!(FieldsPresentFlags & KERNEL_PRESENT), "Cannot set the cache after the kernel function is already set. Please call .withCache() first and .kernel() later in the chain.");

    return KernelRegistrationBuilder<Cache, FieldsPresentFlags | CACHE_PRESENT>(std::move(op_), std::move(dispatch_key_), kernel_, &cacheCreator<Cache>);
  }
};

} // namespace c10

// NB: Semicolon after applying this macro is MANDATORY
#define C10_REGISTER_KERNEL(OperatorHandle)                                                           \
  static KernelRegistrar MACRO_CONCAT(__kernelRegistrationBuilder_, __COUNTER__) = KernelRegistrationBuilder<void, 0>(OperatorHandle)
