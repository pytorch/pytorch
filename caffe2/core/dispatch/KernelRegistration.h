#pragma once

#include "c10/util/Optional.h"
#include "caffe2/core/dispatch/Dispatcher.h"
#include "caffe2/core/dispatch/OpSchema.h"

/**
 * To register your own kernel for an operator, do in one (!) cpp file:
 *   C10_REGISTER_KERNEL(OpSchemaDef)
 *      .kernel(&kernel_func)
 *      .dispatchKey(dispatch_key);
 */

namespace c10 {

// TODO Test different order for builder
// TODO Test no dispatch key defined

/**
 * Class which, on construction, registers an operator in the dispatch table.  The intent is that
 * this class is constructed at static initialization time so that operators automatically get
 * registered when a dlopen() occurs.
 *
 * You shouldn't call this directly; instead, use the KernelRegistrationBuilder
 *
 * @tparam OpSchemaDef
 */
template<class OpSchemaDef>
class KernelRegistrar final {
private:
    using Schema = OpSchema<OpSchemaDef>;
public:
  /**
   * @param kernel The concrete function implementation to register
   * @param dispatch_key  The dispatch key to register the function to
   */
  KernelRegistrar(typename Schema::signature::func_type* kernel, typename Schema::dispatch::dispatch_key_type dispatch_key)
  : dispatch_key_(std::move(dispatch_key)), owns_registration_(true) {
    Dispatcher<OpSchemaDef>::registerKernel(kernel, dispatch_key_);
  }

  KernelRegistrar(KernelRegistrar&& rhs)
  : dispatch_key_(std::move(rhs.dispatch_key_)), owns_registration_(true) {
    rhs.owns_registration_ = false;
  }

  // not needed for now
  KernelRegistrar& operator=(KernelRegistrar&& rhs) = delete;

  ~KernelRegistrar() {
    if (owns_registration_) {
      Dispatcher<OpSchemaDef>::deregisterKernel(dispatch_key_);
    }
  }

private:
  const typename Schema::dispatch::dispatch_key_type dispatch_key_;
  bool owns_registration_;

  C10_DISABLE_COPY_AND_ASSIGN(KernelRegistrar);
};

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
 *
 * @tparam OpSchemaDef The operator schema this is building a KernelRegistration for
 * @tparam hasKernel Boolean for compile-time checking that a kernel is specified before finalizing the builder
 * @tparam hasDispatchKey Boolean for compile-time checking thhat a dispatch key is specified before finalizing the builder
 */
template<class OpSchemaDef, uint64_t FieldsPresentFlags>
class KernelRegistrationBuilder final {
private:
  using Schema = OpSchema<OpSchemaDef>;

  static constexpr uint64_t KERNEL_PRESENT = 0x01 << 0;
  static constexpr uint64_t DISPATCH_KEY_PRESENT = 0x01 << 1;

  c10::optional<typename Schema::signature::func_type*> kernel_;
  c10::optional<typename Schema::dispatch::dispatch_key_type> dispatch_key_;

 public:
  constexpr KernelRegistrationBuilder()
      : KernelRegistrationBuilder(c10::nullopt, c10::nullopt) {}

  constexpr KernelRegistrationBuilder(
      c10::optional<typename Schema::signature::func_type*> kernel,
      c10::optional<typename Schema::dispatch::dispatch_key_type> dispatch_key)
      : kernel_(std::move(kernel)), dispatch_key_(std::move(dispatch_key)) {}

  /**
   * Implicit coercion to KernelRegistrar<OpSchemaDef> that finalizes the builder and
   * creates the object.
   * @return Produced KernelRegistrar
   */
  constexpr operator KernelRegistrar<OpSchemaDef>() && {
    static_assert(FieldsPresentFlags & KERNEL_PRESENT, "Forgot to call .kernel() in kernel registration");
    static_assert(FieldsPresentFlags & DISPATCH_KEY_PRESENT, "Forgot to call .dispatchKey() in kernel registration");
    return KernelRegistrar<OpSchemaDef>(std::move(*kernel_), std::move(*dispatch_key_));
  }

  /**
   * Specify the concrete function implementation for this dispatch registration
   * @param kernel concrete function implementation to be registered
   * @return "this" for method chaining
   */
  constexpr KernelRegistrationBuilder<OpSchemaDef, FieldsPresentFlags | KERNEL_PRESENT> kernel(typename Schema::signature::func_type* kernel_func) && {
    static_assert(!(FieldsPresentFlags & KERNEL_PRESENT), "Tried to define kernel twice in same op registration");
    return KernelRegistrationBuilder<OpSchemaDef, FieldsPresentFlags | KERNEL_PRESENT>(*kernel_func, std::move(dispatch_key_));
  }

  /**
   * Specify the dispatch key for this dispatch registration
   * @param dispatch_key dispatch key to register the function to
   * @return "this" for method chaining
   */
  constexpr KernelRegistrationBuilder<OpSchemaDef, FieldsPresentFlags | DISPATCH_KEY_PRESENT> dispatchKey(typename Schema::dispatch::dispatch_key_type dispatch_key) && {
    static_assert(!(FieldsPresentFlags & DISPATCH_KEY_PRESENT), "Tried to define kernel twice in same op registration");
    return KernelRegistrationBuilder<OpSchemaDef, FieldsPresentFlags | DISPATCH_KEY_PRESENT>(std::move(kernel_), std::move(dispatch_key));
  }
};

} // namespace c10

// TODO Can the builder logic be moved to compile time?
// NB: Semicolon after applying this macro is MANDATORY
#define C10_REGISTER_KERNEL(OpSchemaDef)                                                           \
  static KernelRegistrar<OpSchemaDef> MACRO_CONCAT(__kernelRegistrationBuilder_, __COUNTER__) = KernelRegistrationBuilder<OpSchemaDef, 0>()
