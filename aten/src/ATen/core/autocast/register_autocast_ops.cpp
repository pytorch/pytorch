#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
// TODO:  Remove this dependency once callBoxed is possible for all ops
#include <torch/csrc/jit/operator.h>

#include <iostream>
#include <exception>

namespace at {
namespace autocast {

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache FP16 casts of FP32 model weights (see cached_cast below).
//
// After discussion with @ezyang, the cache uses the following structure:
// The key is the source tensor's TensorImpl*, a proxy for a Tensor uuid that's unchanged across shallow copies.
// The value is a tuple of two Tensors:  the source tensor and the casted tensor.  Both are stored as full
// Tensors to ensure their TensorImpls stay alive if other references to them die.
//
// We must keep the source tensor alive because if the source tensor were deallocated, another random Tensor
// could be allocated whose TensorImpl* happened to have the same value.  This TensorImpl* would then mistakenly
// hit in cache, which would be a nasty bug (rare, intermittent, unpredictable).
//
// Since our heuristic caches casts for model weights only, the source Tensors should always stay alive anyway,
// because the model stores them explicitly.  So in the common case, storing a live reference to the
// source tensor in cached_casts is unnecessary but also does no harm (does not increase memory use).
//
// When the autocast context manager exits, which should occur at the end of each forward pass, it calls
// clear_cache to ensure cached Tensors don't leak outside the autocasting region.
thread_local std::unordered_map<TensorImpl*, std::tuple<Tensor, Tensor>> cached_casts;
}

void clear_cache() {
  cached_casts.clear();
}

enum class CastPolicy : uint8_t {
  fp16 = 0,
  fp32,
  promote, // Run in the widest dtype among several args.
  firstarg, // Run in the dtype of the first argument.
  passthrough // Run without casting any args (workaround for some ops the boxed fallback can't handle)
};

enum class Behavior : uint8_t {
  wellbehaved = 0,
  inplace,
  user_supplied_out
};

/*****************************************************************
Logic to pick out Tensor arguments and determine the promote type.
*****************************************************************/

// Overload to catch Tensor args.
// If nextArg is floating-point, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
at::ScalarType prioritize(at::ScalarType current, const Tensor & nextArg) {
  if (!nextArg.is_floating_point()) {
    return current;
  } else {
    auto next = nextArg.scalar_type();
    // For promotion purposes, prioritize double, then float, then half.
    if (current == at::kDouble || next == at::kDouble) {
      return at::kDouble;
    } else if (current == at::kFloat || next == at::kFloat) {
      return at::kFloat;
    } else if (current == at::kHalf && next == at::kHalf) {
      return at::kHalf;
    } else {
      AT_ERROR("Unexpected floating ScalarType in autograd::prioritize");
      return current;
    }
  }
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template<typename T>
at::ScalarType prioritize(at::ScalarType current, T nextArg) {
  return current;
}

// Overload for the tail case.
at::ScalarType promote_type(at::ScalarType current) {
  return current;
}

// Unpack args and pick the widest floating-point dtype among Tensors to use for promotion.
// Non-Tensor arguments are ignored.
template<typename Arg0, typename... Args>
at::ScalarType promote_type(at::ScalarType current, Arg0 arg0, Args... args) {
  auto new_current = prioritize(current, arg0);
  return promote_type(new_current, args...);
}

/***************************************************
Logic to apply a cached cast to any Tensor argument.
***************************************************/

// Overload to catch Tensor args, and cast if necessary.
Tensor cached_cast(at::ScalarType to_type, const Tensor & arg) {
  if (arg.is_floating_point() && arg.scalar_type() != to_type) {
    // Heuristic:  Do what Apex does, and cache FP16 casts of FP32 model weights (leaves).
    // See cached_casts declaration above for detailed strategy.
    bool can_try_cache = (to_type == at::kHalf && arg.scalar_type() == at::kFloat && arg.requires_grad() && arg.is_leaf());
    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      } else {
        auto casted_arg = arg.to(to_type);
        cached_casts.emplace(arg.unsafeGetTensorImpl(), std::tuple<Tensor, Tensor>{arg, casted_arg});
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

// Template to catch non-Tensor args.
// Should we bother with perfect forwarding here?  I lean towards no, because types are deduced from the
// patched function's signature/schema, not from the args.
// Besides, we know exactly what argument types we're dealing with.  Function signatures are given
// explicitly in registration below.  If something looks like it might benefit from perfect forwarding,
// we can deal with that when the time comes.
template<typename T>
T cached_cast(at::ScalarType to_type, T arg) {
  return arg;
}

/*************************************************************************************************************************
Templates to provide wrapper functions

I'm copying the pattern used in core/boxing/kernel_function.h to extract args and return type.
(see also https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)
This strategy uses an exterior "WrapFunction" that extracts arguments on behalf of (in my case several specializations of)
an interior "WrapFunction_".

Interior WrapFunction_ specializations are defined for each function CastPolicy and Behavior.
*************************************************************************************************************************/

// Base template for the interior "WrapFunction_"
template<CastPolicy policy, Behavior behavior, class FuncType, FuncType* F, class Ret, class ArgList> struct WrapFunction_ {};
// The exterior "WrapFunction" is defined later, below all the WrapFunction_ specializations.

/******************************************************************
Well-behaved ops

Well-behaved means
- The op has an at:: exposure, AND
- if the op requires casting or promotion, all Tensor arguments are
  received as "const Tensor &", which means the op does not modify
  any Tensor arguments in-place.

CastPolicy::passthrough ops (which don't cast or promote) may
receive non-const Tensor & arguments and remain well-behaved as
long as they have an at:: exposure.
******************************************************************/

// Separate struct specializations for the four CastPolicies so the wrapper instantiation for each op
// only compiles with the type selection logic it actually needs (ie, promote_type is only used by
// CastPolicy::promote ops).

// CastPolicy::fp16
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto to_type = at::kHalf;
    return (*F)(cached_cast(to_type, args)...);
  }
};

// CastPolicy::fp32
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto to_type = at::kFloat;
    return (*F)(cached_cast(to_type, args)...);
  }
};

// CastPolicy::promote
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::promote, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto to_type = promote_type(at::kHalf, args...);
    return (*F)(cached_cast(to_type, args)...);
  }
};

// The passthrough specialization supplies explicitly-registerable unboxed kernels
// to serve ops that meet all of the following:
// - don't require casting, so in principle they should use the boxed fallback
// - don't play well with the boxed fallback, unfortunately
// - are well-behaved
// Example:  at::detach_
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::passthrough, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return (*F)(args...);
  }
};

/**********************************************************************************
Different treatment for ops that aren't well-behaved

Generally, these ops fall into two categories, with two possible subcategories:
1. Ops that modify Tensor arguments in-place
  a. Ops that do have an at:: exposure
  b. Ops that don't have an at:: exposure and can only be called as a Tensor method
2. Ops that write to a user-supplied `out=...` buffer
  a. Ops that do have an at:: exposure
  b. Ops that don't have an at:: exposure and can only be called as a Tensor method

There seems to be a correlation (perhaps coincidental) between
not having an at::* exposure and being in-place.

The tools we have to reduce manual special casing are templates,
macros, and codegen.  Based on the observed commonalities among
in-place ops and the observed commonalities among ops with
user-supplied out, I'll try templates for now.

For in-place ops and ops with a user-supplied out, functions with multiple arguments
that need type standardization are called "firstarg" instead of "promote."
They cast other arguments to the type of the in-place or out buffer before running,
because they must eventually write to that buffer anyway.  This mimics Apex's
current strategy.
**********************************************************************************/

// In-place ops. Basically, I try to imitate what VariableType*.cpp is doing:
// forward calls to an at:: function if available and a Tensor method, ie self.method, otherwise.

// 1.a. Most of the time VariableType forwards to an at:: call.
// For these, I can call into the "F" template parameter directly.
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Behavior::inplace, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs> static
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto run_as_type = at::kHalf;
    if (self.scalar_type() == run_as_type) {
      (*F)(self, cached_cast(run_as_type, args)...);
    } else {
      auto fp16_self_lvalue = cached_cast(run_as_type, self);
      (*F)(fp16_self_lvalue, cached_cast(run_as_type, args)...);
      self.copy_(fp16_self_lvalue);
    }
    return self;
  }
  static Tensor & call(Args... args) {
    return peel_first_arg_and_run(args...);
  }
};

template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, Behavior::inplace, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs> static
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto run_as_type = at::kFloat;
    // TODO:  how do we want to handle double-precision inputs?
    if (self.scalar_type() == run_as_type) {
      (*F)(self, cached_cast(run_as_type, args)...);
    } else {
      AT_ASSERT(false, "In-place operators that autocast to torch.float32 require that the argument modified in-place is already torch.float32.  (Other arguments may be any type.)");
    }
    return self;
  }
  static Tensor & call(Args... args) {
    return peel_first_arg_and_run(args...);
  }
};

// 1.b. For ops that don't have an at:: exposure, VariableType's functions forward to a Tensor method,
// eg self_.addmm_.  For these, I need to create a macro that defines a specialization, so I can request self.FUNC.
// Pure macros or codegen are looking better and better...
// fp16
#define SPECIALIZE_FP16_INPLACE_NO_AT_EXPOSURE(METHOD, OVERLOAD) \
template<class Ret, class... Args> \
struct WrapFunction_<CastPolicy::fp16, Behavior::inplace, decltype(OVERLOAD), OVERLOAD, Ret, guts::typelist::typelist<Args...>> { \
  template<typename... RemainingArgs> static \
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) { \
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId); \
    auto run_as_type = at::kHalf; \
    if (self.scalar_type() == run_as_type) { \
      self.METHOD(cached_cast(run_as_type, args)...); \
    } else { \
      auto fp16_self_lvalue = cached_cast(run_as_type, self); \
      fp16_self_lvalue.METHOD(cached_cast(run_as_type, args)...); \
      self.copy_(fp16_self_lvalue); \
    } \
    return self; \
  } \
  static Tensor & call(Args... args) { \
    return peel_first_arg_and_run(args...); \
  } \
};

// fp32
#define SPECIALIZE_FP32_INPLACE_NO_AT_EXPOSURE(METHOD, OVERLOAD) \
template<class Ret, class... Args> \
struct WrapFunction_<CastPolicy::fp32, Behavior::inplace, decltype(OVERLOAD), OVERLOAD, Ret, guts::typelist::typelist<Args...>> { \
  template<typename... RemainingArgs> static \
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) { \
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId); \
    auto run_as_type = at::kFloat; \
    if (self.scalar_type() == run_as_type) { \
      self.METHOD(cached_cast(run_as_type, args)...); \
    } else { \
      AT_ASSERT(false, "In-place operators that autocast to torch.float32 require that the argument modified in-place is already torch.float32.  (Other arguments may be any type.)"); \
    } \
    return self; \
  } \
  static Tensor & call(Args... args) { \
    return peel_first_arg_and_run(args...); \
  } \
};

// firstarg
#define SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(METHOD, OVERLOAD) \
template<class Ret, class... Args> \
struct WrapFunction_<CastPolicy::firstarg, Behavior::inplace, decltype(OVERLOAD), OVERLOAD, Ret, guts::typelist::typelist<Args...>> { \
  template<typename... RemainingArgs> static \
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) { \
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId); \
    return self.METHOD(cached_cast(self.scalar_type(), args)...); \
  } \
  static Tensor & call(Args... args) { \
    return peel_first_arg_and_run(args...); \
  } \
};

// Define the specializations that will serve each 1.b. METHOD.
// For each specialization, I must also supply a dummy function with the right signature
// (aka the signature copied from VariableType) to provide the type information for instantiation.
// Most of the time there is only one overload so signature-based disambiguation is not required.
// pow_ and eq_ are exceptions.
// fp16 ops
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) { return self; }
SPECIALIZE_FP16_INPLACE_NO_AT_EXPOSURE(addmm_, addmm_)
Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) { return self; }
SPECIALIZE_FP16_INPLACE_NO_AT_EXPOSURE(addr_, addr_)
// fp32 ops
Tensor & erfinv_(Tensor & self) { return self; }
SPECIALIZE_FP32_INPLACE_NO_AT_EXPOSURE(erfinv_, erfinv_)
Tensor & pow_scalar_(Tensor & self, Scalar exponent) { return self; }
SPECIALIZE_FP32_INPLACE_NO_AT_EXPOSURE(pow_, pow_scalar_)
Tensor & pow_tensor_(Tensor & self, const Tensor & exponent) { return self; }
SPECIALIZE_FP32_INPLACE_NO_AT_EXPOSURE(pow_, pow_tensor_)
// firstarg ops
Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(addcdiv_, addcdiv_)
Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(addcmul_, addcmul_)
Tensor & atan2_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(atan2_, atan2_)
Tensor & eq_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(eq_, eq_scalar_)
Tensor & eq_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(eq_, eq_tensor_)
Tensor & ge_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(ge_, ge_scalar_)
Tensor & ge_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(ge_, ge_tensor_)
Tensor & gt_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(gt_, gt_scalar_)
Tensor & gt_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(gt_, gt_tensor_)
Tensor & le_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(le_, le_scalar_)
Tensor & le_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(le_, le_tensor_)
Tensor & lt_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(lt_, lt_scalar_)
Tensor & lt_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(lt_, lt_tensor_)
Tensor & ne_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(ne_, ne_scalar_)
Tensor & ne_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(ne_, ne_tensor_)
Tensor & add_tensor_(Tensor & self, const Tensor & other, Scalar alpha) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(add_, add_tensor_)
Tensor & add_scalar_(Tensor & self, Scalar other, Scalar alpha) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(add_, add_scalar_)
Tensor & div_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(div_, div_tensor_)
Tensor & div_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(div_, div_scalar_)
Tensor & mul_tensor_(Tensor & self, const Tensor & other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(mul_, mul_tensor_)
Tensor & mul_scalar_(Tensor & self, Scalar other) { return self; }
SPECIALIZE_FIRSTARG_INPLACE_NO_AT_EXPOSURE(mul_, mul_scalar_)

// 2.a. Specializations for functions with out=... arguments
// According to VariableType, these don't support automatic differentiation.
// fp16
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Behavior::user_supplied_out, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs>
  static Tensor & peel_first_arg_and_run(Tensor & out, RemainingArgs... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto run_as_type = at::kHalf;
    if (out.scalar_type() == run_as_type) {
      (*F)(out, cached_cast(run_as_type, args)...);
    } else {
      auto fp16_out_lvalue = at::empty_like(out, out.options().dtype(run_as_type));
      (*F)(fp16_out_lvalue, cached_cast(run_as_type, args)...);
      out.copy_(fp16_out_lvalue);
    }
    return out;
  }
  static Tensor & call(Args... args) {
    return peel_first_arg_and_run(args...);
  }
};

// fp32
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, Behavior::user_supplied_out, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs>
  static Tensor & peel_first_arg_and_run(Tensor & out, RemainingArgs... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto run_as_type = at::kFloat;
    if (out.scalar_type() == run_as_type) {
      (*F)(out, cached_cast(run_as_type, args)...);
    } else {
      AT_ASSERT(false, "If you supply an 'out=my_output' argument to an op that autocasts to torch.float32, my_output must be torch.float32.  (Other arguments may be any type.)");
    }
    return out;
  }
  static Tensor & call(Args... args) {
    return peel_first_arg_and_run(args...);
  }
};

// firstarg
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::firstarg, Behavior::user_supplied_out, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs>
  static Tensor & peel_first_arg_and_run(Tensor & out, RemainingArgs... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return (*F)(out, cached_cast(out.scalar_type(), args)...);
  }
  static Tensor & call(Args... args) {
    return peel_first_arg_and_run(args...);
  }
};

// 2.b. Haven't encountered a 2b yet, luckily.


/***************************************************************
Wrapper to extract args as used by core/boxing/kernel_function.h
***************************************************************/
template<CastPolicy policy, Behavior behavior, class FuncType, FuncType* F>
struct WrapFunction final {
  using type = WrapFunction_<
      policy,
      behavior,
      FuncType,
      F,
      typename guts::function_traits<FuncType>::return_type,
      typename guts::function_traits<FuncType>::parameter_types
  >;
};

/*******************************
Boxed fallback for all other ops
*******************************/

// Temporary workaround used by autocast_fallback below.
void callBoxedWorkaround(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto s = Symbol::fromQualString(op.schema().name());
  auto operators = torch::jit::getAllOperatorsFor(s);
  // Find the exact match
  std::shared_ptr<torch::jit::Operator> jit_op;
  for (const auto& candidate_op : operators) {
    auto candidate_schema = candidate_op->schema();
    // NB: this is a VERY slow equality test
    if (candidate_schema == op.schema()) {
      jit_op = candidate_op;
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(jit_op);

  auto offset = jit_op->getOperation()(*stack);
  TORCH_INTERNAL_ASSERT(offset == 0);
}

// void autocast_fallback(const c10::OperatorHandle& op, c10::Stack* stack) {
void autocast_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "autocast_fallback" << std::endl;

  c10::impl::ExcludeTensorTypeIdGuard no_autocast(TensorTypeId::AutocastTensorId);
  // Temporary workaround.
  // TODO:  Replace callBoxedWorkaround with op.callBoxed(stack) once callBoxed is possible for all ops.
  callBoxedWorkaround(op, stack);
  // Alternative:  Have the fallback go straight for the forward-compatible call, and if anything breaks,
  // manually register a Patch<CastPolicy::passthrough, etc>::call for it.
  // op.callBoxed(stack);
}
// TODO:  Once fallthrough is implemented (https://github.com/pytorch/pytorch/issues/29548),
// autocast_fallback can be deleted entirely.

#ifndef USE_STATIC_DISPATCH
namespace {
/*****************************************************************************************************************
This section performs load-time registration for all autocast wrappers.

It's debatable at which level operations should be patched.

Option 1:  Patch only at the level of explicit calls into cudnn/cublas (cudnn_convolution, etc),
because those are the code paths that are guaranteed to use Tensor Cores, therefore they're the ones that
will benefit most from FP16.  One difficulty with this approach is that convolutions (and other ops) are wrapped
in several layers of at::* calls.  Several non-explicitly-registered layers (e.g. at::convolution) could be called
before we reach cudnn_convolution.  Because they are not explicitly registered, these layers would invoke the
boxed fallback, which would exclude AutocastTensorId, so by the time at::cudnn_convolution is actually
called, it wouldn't route back through the autocasting logic at all.

Option 2:  Patch at the Python-exposed level, to make 100% sure autograd history
recording can't sneak in ahead of autocast.  This mirrors Apex most closely.

I think Option 2 is the right answer for all ops, not just convolutions.  Option 1 is what I implement here.
*****************************************************************************************************************/

/**************************
Boxed fallback registration
**************************/
auto register_fallback = c10::Dispatcher::singleton()
  .registerBackendFallbackKernel(TensorTypeId::AutocastTensorId,
                                 KernelFunction::makeFromBoxedFunction<&autocast_fallback>());

/********************
Explicit registration
********************/
// TODO:  Codegen the stuff below?  Ed said
// > you are going to have to write the function definition at some point, I wouldn't try to get clever about it
// Therefore, for the moment, this is all copy pasted in from VariableTypeEverything.cpp with appropriate substitutions.

// Macros to reduce boilerplate somewhat
#define PATCH(FUNC, POLICY, BEHAVIOR) &WrapFunction<CastPolicy::POLICY, Behavior::BEHAVIOR, decltype( FUNC ), FUNC>::type::call

#define KERNEL(FUNC, SCHEMA, SIGNATURE, POLICY, BEHAVIOR) \
  .op(torch::RegisterOperators::options() \
    .schema(SCHEMA) \
    .kernel<SIGNATURE>(TensorTypeId::AutocastTensorId, \
    &WrapFunction<CastPolicy::POLICY, Behavior::BEHAVIOR, SIGNATURE, FUNC>::type::call) \
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))

#define KERNEL_UNBOXED_ONLY(FUNC, SCHEMA, SIGNATURE, POLICY, BEHAVIOR) \
  .op(torch::RegisterOperators::options() \
    .schema(SCHEMA) \
    .impl_unboxedOnlyKernel<SIGNATURE, \
    &WrapFunction<CastPolicy::POLICY, Behavior::BEHAVIOR, SIGNATURE, FUNC>::type::call \
    >(TensorTypeId::AutocastTensorId) \
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))

/*****************************************
Explicit registration for well-behaved ops
*****************************************/
auto register_well_behaved = torch::RegisterOperators()
  // fp16
  // TODO (to consider): should the slow_conv*d from python_nn_functions_dispatch.h go here?
  // They use unfolding + admm (which may use cublas) so it's possible they benefit from Tensor Cores,
  // although if that code path is being used, it's probably unsalvageably slow regardless.
  KERNEL_UNBOXED_ONLY(at::_convolution, "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::_convolution_nogroup, "aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::conv1d, "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::conv2d, "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::conv3d, "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::conv_tbc, "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::conv_transpose1d, "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::conv_transpose2d, "aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(conv_transpose3d, "aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::convolution, "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::cudnn_convolution, "aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::cudnn_convolution_transpose, "aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16, wellbehaved)
  KERNEL(at::prelu, "aten::prelu(Tensor self, Tensor weight) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL(at::addmm, "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, wellbehaved)
  KERNEL(at::addmv, "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, wellbehaved)
  KERNEL(at::addr, "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, wellbehaved)
  KERNEL(at::matmul, "aten::matmul(Tensor self, Tensor other) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL(at::mm, "aten::mm(Tensor self, Tensor mat2) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL(at::mv, "aten::mv(Tensor self, Tensor vec) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  // fp32
  KERNEL(at::acos, "aten::acos(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::asin, "aten::asin(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::cosh, "aten::cosh(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::erfinv, "aten::erfinv(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::exp, "aten::exp(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::expm1, "aten::expm1(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::log, "aten::log(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::log10, "aten::log10(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::log2, "aten::log2(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::log1p, "aten::log1p(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::reciprocal, "aten::reciprocal(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::rsqrt, "aten::rsqrt(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::sinh, "aten::sinh(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::tan, "aten::tan(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL(at::pow, "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor", Tensor (const Tensor &, Scalar), fp32, wellbehaved)
  KERNEL(at::pow, "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor", Tensor (const Tensor &, const Tensor &), fp32, wellbehaved)
  KERNEL(at::pow, "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor", Tensor (Scalar, const Tensor &), fp32, wellbehaved)
  // promote
  KERNEL(at::addcdiv, "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), promote, wellbehaved)
  KERNEL(at::addcmul, "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), promote, wellbehaved)
  KERNEL(at::atan2, "aten::atan2(Tensor self, Tensor other) -> Tensor", Tensor (const Tensor &, const Tensor &), promote, wellbehaved)
  KERNEL(at::cross, "aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor", Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>), promote, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::bilinear, "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &), promote, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::tensordot, "aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef), promote, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::dot, "aten::dot(Tensor self, Tensor tensor) -> Tensor", Tensor (const Tensor &, const Tensor &), promote, wellbehaved)
  KERNEL(at::equal, "aten::equal(Tensor self, Tensor other) -> bool", bool (const Tensor &, const Tensor &), promote, wellbehaved)
  // passthrough
  KERNEL_UNBOXED_ONLY(at::detach_, "aten::detach_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::zero_, "aten::zero_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::eq_out, "aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::eq_out, "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::ge_out, "aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::ge_out, "aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::gt_out, "aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::gt_out, "aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::le_out, "aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::le_out, "aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::lt_out, "aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::lt_out, "aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::ne_out, "aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::ne_out, "aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::add_out, "aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::div_out, "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::mul_out, "aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), passthrough, wellbehaved)
  ;

/**************************************************************************************
Explicit registration for non-well-behaved ops
It's not technically required to register these separately, but it helps organize them.
**************************************************************************************/

/**************************************
1a:  in-place ops with an at:: exposure
**************************************/
auto register_inplace = torch::RegisterOperators()
  // fp16 ops
  KERNEL_UNBOXED_ONLY(at::addmv_, "aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, inplace)
  // fp32 ops
  KERNEL_UNBOXED_ONLY(at::acos_, "aten::acos_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::asin_, "aten::asin_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::cosh_, "aten::cosh_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::exp_, "aten::exp_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::expm1_, "aten::expm1_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::log_, "aten::log_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::log10_, "aten::log10_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::log2_, "aten::log2_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::log1p_, "aten::log1p_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::reciprocal_, "aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::rsqrt_, "aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::sinh_, "aten::sinh_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::tan_, "aten::tan_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  // firstarg ops
  ;

/**************************************************
1b:  in-place ops only accessible as Tensor methods
**************************************************/
auto register_inplace_method_only = torch::RegisterOperators()
  // fp16 ops
  KERNEL_UNBOXED_ONLY(at::autocast::addmm_, "aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::addr_, "aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, inplace)
  // fp32 ops
  KERNEL_UNBOXED_ONLY(at::autocast::erfinv_, "aten::erfinv_(Tensor(a!) self) -> Tensor(a!)", Tensor & (Tensor &), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::pow_scalar_, "aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)", Tensor & (Tensor &, Scalar), fp32, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::pow_tensor_, "aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, inplace)
  // firstarg ops
  KERNEL_UNBOXED_ONLY(at::autocast::addcdiv_, "aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::addcmul_, "aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::atan2_, "aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::eq_scalar_, "aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::eq_tensor_, "aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::ge_scalar_, "aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::ge_tensor_, "aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::gt_scalar_, "aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::gt_tensor_, "aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::le_scalar_, "aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::le_tensor_, "aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::lt_scalar_, "aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::lt_tensor_, "aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::ne_scalar_, "aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::ne_tensor_, "aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::add_tensor_, "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::add_scalar_, "aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)", Tensor & (Tensor &, Scalar, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::div_tensor_, "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::div_scalar_, "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::mul_tensor_, "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), firstarg, inplace)
  KERNEL_UNBOXED_ONLY(at::autocast::mul_scalar_, "aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)", Tensor & (Tensor &, Scalar), firstarg, inplace)
  ;

/******************************************************************************************************
Explicit registration for non-well-behaved ops part 2:  ops that write to a user-supplied output buffer
******************************************************************************************************/
auto register_user_supplied_out = torch::RegisterOperators()
  // fp16 ops
  KERNEL_UNBOXED_ONLY(at::addmm_out, "aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::addmv_out, "aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::addr_out, "aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::matmul_out, "aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), fp16, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::mm_out, "aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), fp16, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::mv_out, "aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), fp16, user_supplied_out)
  // fp32 ops
  KERNEL_UNBOXED_ONLY(at::acos_out, "aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::asin_out, "aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::cosh_out, "aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::erfinv_out, "aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::exp_out, "aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::expm1_out, "aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::log_out, "aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::log10_out, "aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::log2_out, "aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::log1p_out, "aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::reciprocal_out, "aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::rsqrt_out, "aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::sinh_out, "aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::tan_out, "aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::pow_out, "aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, Scalar), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::pow_out, "aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), fp32, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::pow_out, "aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, Scalar, const Tensor &), fp32, user_supplied_out)
  // firstarg ops
  KERNEL_UNBOXED_ONLY(at::addcdiv_out, "aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar), firstarg, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::addcmul_out, "aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar), firstarg, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::atan2_out, "aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), firstarg, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::cross_out, "aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &, c10::optional<int64_t>), firstarg, user_supplied_out)
  KERNEL_UNBOXED_ONLY(at::dot_out, "aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)", Tensor & (Tensor &, const Tensor &, const Tensor &), firstarg, user_supplied_out)
  ;

}
#endif

} // namespace autocast
} // namespace at
