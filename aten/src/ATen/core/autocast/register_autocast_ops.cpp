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
  promote,
  passthrough
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

/******************************************************************
Templates to provide wrapper functions for well-behaved ops

Well-behaved means
- The op has an at:: exposure, AND
- if the op requires casting or promotion, all Tensor arguments are
  received as "const Tensor &", which means the op does not modify
  any Tensor arguments in-place.

CastPolicy::passthrough ops (which don't cast or promote) may
receive non-const Tensor & arguments and remain well-behaved as
long as they have an at:: exposure.
******************************************************************/

// Copying the pattern used in core/boxing/kernel_function.h to extract args and return type
// (see also https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)
template<CastPolicy policy, Behavior behavior, class FuncType, FuncType* F, class Ret, class ArgList> struct WrapFunction_ {};

// Separate struct specializations for the four CastPolicies so the wrapper instantiation for each op
// only compiles with the type selection logic it actually needs (ie, promote_type is only used by
// CastPolicy::promote ops).

// CastPolicy::fp16
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    auto to_type = at::kHalf;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return (*F)(cached_cast(to_type, args)...);
  }
};

// CastPolicy::fp32
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    auto to_type = at::kFloat;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return (*F)(cached_cast(to_type, args)...);
  }
};

// CastPolicy::promote
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::promote, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    auto to_type = promote_type(at::kHalf, args...);
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
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
**********************************************************************************/

// In-place ops. Basically, I try to imitate what VariableType*.cpp is doing:
// forward calls to an at:: function if available and a Tensor method, ie self.method, otherwise.

// 1.a. Most of the time VariableType forwards to an at:: call.
// For these, I can call into the "F" template parameter directly.
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Behavior::inplace, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs> static
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) {
    auto run_as_type = at::kHalf;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
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
    auto run_as_type = at::kFloat;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
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
#define SPECIALIZE_FP16_INPLACE_NO_AT_EXPOSURE(METHOD, OVERLOAD) \
template<class Ret, class... Args> \
struct WrapFunction_<CastPolicy::fp16, Behavior::inplace, decltype(OVERLOAD), OVERLOAD, Ret, guts::typelist::typelist<Args...>> { \
  template<typename... RemainingArgs> static \
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) { \
    auto run_as_type = at::kHalf; \
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId); \
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

#define SPECIALIZE_FP32_INPLACE_NO_AT_EXPOSURE(METHOD, OVERLOAD) \
template<class Ret, class... Args> \
struct WrapFunction_<CastPolicy::fp32, Behavior::inplace, decltype(OVERLOAD), OVERLOAD, Ret, guts::typelist::typelist<Args...>> { \
  template<typename... RemainingArgs> static \
  Tensor & peel_first_arg_and_run(Tensor & self, RemainingArgs... args) { \
    auto run_as_type = at::kFloat; \
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId); \
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

// Define the specializations that will serve each 1.b. METHOD.
// For each specialization, I must also supply a dummy OVERLOAD with the right signature
// (copied from VariableType) to provide the type information for instantiation.
// Most of the time there is only one OVERLOAD so no disambiguation is required.
// pow_ is the exception.  The pow(_) family of overloads is a pain in the neck.
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

// 2.a. Functions with out=... arguments
// According to VariableType, these don't support automatic differentiation.
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Behavior::user_supplied_out, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs>
  static Tensor & peel_first_arg_and_run(Tensor & out, RemainingArgs... args) {
    auto run_as_type = at::kHalf;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
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

template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, Behavior::user_supplied_out, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  template<typename... RemainingArgs>
  static Tensor & peel_first_arg_and_run(Tensor & out, RemainingArgs... args) {
    auto run_as_type = at::kFloat;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
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

// 2.b. Haven't encountered a 2b yet, luckily.


/********************************************************
Wrapper template as used by core/boxing/kernel_function.h
********************************************************/
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

// Macro to reduce boilerplate
#define PATCH(FUNC, POLICY, BEHAVIOR) &WrapFunction<CastPolicy::POLICY, Behavior::BEHAVIOR, decltype( FUNC ), FUNC>::type::call

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

/*****************************************
Explicit registration for well-behaved ops
*****************************************/

// TODO:  Codegen the stuff below?  Ed said
// > you are going to have to write the function definition at some point, I wouldn't try to get clever about it
// Therefore, for the moment, this is all copy pasted in from VariableTypeEverything.cpp with appropriate substitutions.
auto register_well_behaved = torch::RegisterOperators()
  // CastPolicy::fp16
  // TODO (to consider): should the slow_conv*d from python_nn_functions_dispatch.h go here?
  // They use unfolding + admm (which may use cublas) so it's possible they benefit from Tensor Cores,
  // although if that code path is being used, it's probably unsalvageably slow regardless.
  .op(torch::RegisterOperators::options()
    .schema("aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool), PATCH(at::_convolution, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef), PATCH(at::_convolution_nogroup, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), PATCH(at::conv1d, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), PATCH(at::conv2d, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), PATCH(at::conv3d, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>(TensorTypeId::AutocastTensorId, PATCH(at::conv_tbc, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), PATCH(conv_transpose1d, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), PATCH(conv_transpose2d, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), PATCH(conv_transpose3d, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t), PATCH(at::convolution, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), PATCH(at::cudnn_convolution, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), PATCH(at::cudnn_convolution_transpose, fp16, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::prelu(Tensor self, Tensor weight) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::prelu, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(TensorTypeId::AutocastTensorId, PATCH(at::addmm, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(TensorTypeId::AutocastTensorId, PATCH(at::addmv, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(TensorTypeId::AutocastTensorId, PATCH(at::addr, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::matmul(Tensor self, Tensor other) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::matmul, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::mm, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mv(Tensor self, Tensor vec) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::mv, fp16, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // CastPolicy::fp32
  .op(torch::RegisterOperators::options()
    .schema("aten::acos(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::acos, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::asin(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::asin, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cosh(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::cosh, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::erfinv(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::erfinv, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::exp(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::exp, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::expm1(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::expm1, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::log, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log10(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::log10, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log2(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::log2, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log1p(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::log1p, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reciprocal(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::reciprocal, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rsqrt(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::rsqrt, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sinh(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::sinh, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::tan(Tensor self) -> Tensor")
    .kernel<Tensor (const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH(at::tan, fp32, wellbehaved))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor")
    .kernel<Tensor (const Tensor &, Scalar)>(TensorTypeId::AutocastTensorId,
    // The pow overloads don't play well with the decltype in my PATCH helper macro,
    // so I have to write out the full instantiation for WrapFunction.
    &WrapFunction<CastPolicy::fp32, Behavior::wellbehaved, Tensor (const Tensor &, Scalar), at::pow>::type::call)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(TensorTypeId::AutocastTensorId,
    &WrapFunction<CastPolicy::fp32, Behavior::wellbehaved, Tensor (const Tensor &, const Tensor &), at::pow>::type::call)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor")
    .kernel<Tensor (Scalar, const Tensor &)>(TensorTypeId::AutocastTensorId,
    &WrapFunction<CastPolicy::fp32, Behavior::wellbehaved, Tensor (Scalar, const Tensor &), at::pow>::type::call)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // CastPolicy::promote
  // CastPolicy::passthrough
  .op(torch::RegisterOperators::options()
    .schema("aten::detach_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::detach_, passthrough, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::zero_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::zero_, passthrough, wellbehaved)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
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
  .op(torch::RegisterOperators::options()
    .schema("aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), PATCH(at::addmv_, fp16, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // fp32 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::acos_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::acos_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::asin_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::asin_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cosh_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::cosh_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::exp_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::exp_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::expm1_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::expm1_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::log_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log10_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::log10_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log2_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::log2_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log1p_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::log1p_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::reciprocal_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::rsqrt_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sinh_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::sinh_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::tan_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::tan_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // promote ops
  ;

/**************************************************
1b:  in-place ops only accessible as Tensor methods
**************************************************/
auto register_inplace_method_only = torch::RegisterOperators()
  // fp16 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), PATCH(at::autocast::addmm_, fp16, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), PATCH(at::autocast::addr_, fp16, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // fp32 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::erfinv_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH(at::autocast::erfinv_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar), PATCH(at::autocast::pow_scalar_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::autocast::pow_tensor_, fp32, inplace)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // promote ops
  ;

/******************************************************************************************************
Explicit registration for non-well-behaved ops part 2:  ops that write to a user-supplied output buffer
******************************************************************************************************/
auto register_user_supplied_out = torch::RegisterOperators()
  // fp16 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), PATCH(at::addmm_out, fp16, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), PATCH(at::addmv_out, fp16, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), PATCH(at::addr_out, fp16, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), PATCH(at::matmul_out, fp16, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), PATCH(at::mm_out, fp16, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &), PATCH(at::mv_out, fp16, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // fp32 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::acos_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::asin_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::cosh_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::erfinv_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::exp_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::expm1_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::log_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::log10_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::log2_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::log1p_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::reciprocal_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::rsqrt_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::sinh_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &), PATCH(at::tan_out, fp32, user_supplied_out)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, Scalar),
    // The pow overloads don't play well with the decltype in my PATCH helper macro,
    // so I have to write out the full WrapFunction instantiation.
    &WrapFunction<CastPolicy::fp32, Behavior::user_supplied_out, Tensor & (Tensor &, const Tensor &, Scalar), at::pow_out>::type::call
    >(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &),
    &WrapFunction<CastPolicy::fp32, Behavior::user_supplied_out, Tensor & (Tensor &, const Tensor &, const Tensor &), at::pow_out>::type::call
    >(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, Scalar, const Tensor &),
    &WrapFunction<CastPolicy::fp32, Behavior::user_supplied_out, Tensor & (Tensor &, Scalar, const Tensor &), at::pow_out>::type::call
    >(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // promote ops
  ;

#undef PATCH
}
#endif

} // namespace autocast
} // namespace at
