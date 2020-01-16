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
  fp16 = 0, // Cast all inputs to fp16 before running the op.
  fp32, // Cast all inputs to fp32 before running the op.
  fp32_dtype_flag, // Handles functions that should run + output in fp32 and support an output dtype flag.
                   // The policy is:  If the user has explicity specified a dtype, respect it.
                   // Otherwise, cast to the autocast type.
  promote, // Run in the widest dtype among several args.
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

/*******************************************************
Logic to flip an output dtype flag.
Keep it simple for now by assuming only one such flag is
present in the argument list.  If I ever need a function
with more than flag I'll figure out something else.
The policy is:
If the user has explicity specified a dtype, respect it.
Otherwise, cast to the autocast type.
********************************************************/

// Overload to catch dtype flags
c10::optional<ScalarType> flip_dtype_flag(at::ScalarType to_type, const c10::optional<ScalarType> & dtype) {
  return dtype.has_value() ? dtype : to_type;
}

// Template to catch other args
template<typename T>
T flip_dtype_flag(at::ScalarType to_type, T arg) {
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

// CastPolicy::fp32_dtype_flag
template<class FuncType, FuncType* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_dtype_flag, Behavior::wellbehaved, FuncType, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    auto to_type = at::kFloat;
    return (*F)(flip_dtype_flag(to_type, args)...);
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

/****************************************************************
Wrapper to extract args (imitating core/boxing/kernel_function.h)
****************************************************************/
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
  // manually register something for it.
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

Option 2:  Patch the Python-exposed surface of calls, to make 100% sure autograd history
recording can't sneak in ahead of autocast.  This mirrors Apex most closely.

I think Option 2 is the right answer for all ops, not just convolutions.  Option 2 is what I implement here.
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
  KERNEL_UNBOXED_ONLY(at::cudnn_convolution, "aten::cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::cudnn_convolution_transpose, "aten::cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::cudnn_convolution, "aten::cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::cudnn_convolution_transpose, "aten::cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16, wellbehaved)
  KERNEL(at::prelu, "aten::prelu(Tensor self, Tensor weight) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL(at::addmm, "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, wellbehaved)
  KERNEL(at::addmv, "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, wellbehaved)
  KERNEL(at::addr, "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16, wellbehaved)
  KERNEL(at::matmul, "aten::matmul(Tensor self, Tensor other) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL(at::mm, "aten::mm(Tensor self, Tensor mat2) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL(at::mv, "aten::mv(Tensor self, Tensor vec) -> Tensor", Tensor (const Tensor &, const Tensor &), fp16, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::linear, "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &), fp16, wellbehaved)
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
  KERNEL(at::softplus, "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor", Tensor (const Tensor &, Scalar, Scalar), fp32, wellbehaved)
  KERNEL(at::gelu, "aten::gelu(Tensor self) -> Tensor", Tensor (const Tensor &), fp32, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::layer_norm, "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor", Tensor (const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool), fp32, wellbehaved)
  // The macro doesn't like this one so I had to write it out manually.
  .op(torch::RegisterOperators::options()
    .schema("aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double),
    &WrapFunction<CastPolicy::fp32, Behavior::wellbehaved, std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double), at::native_layer_norm>::type::call
    >(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  KERNEL_UNBOXED_ONLY(at::group_norm, "aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor", Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool), fp32, wellbehaved)
  // fp32_dtype_flag
  KERNEL_UNBOXED_ONLY(at::softmax, "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_dtype_flag, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::softmax, "aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_dtype_flag, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::log_softmax, "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_dtype_flag, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::log_softmax, "aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_dtype_flag, wellbehaved)
  // promote
  KERNEL(at::addcdiv, "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), promote, wellbehaved)
  KERNEL(at::addcmul, "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), promote, wellbehaved)
  KERNEL(at::atan2, "aten::atan2(Tensor self, Tensor other) -> Tensor", Tensor (const Tensor &, const Tensor &), promote, wellbehaved)
  KERNEL(at::cross, "aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor", Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>), promote, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::bilinear, "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor", Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &), promote, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::tensordot, "aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef), promote, wellbehaved)
  KERNEL_UNBOXED_ONLY(at::dot, "aten::dot(Tensor self, Tensor tensor) -> Tensor", Tensor (const Tensor &, const Tensor &), promote, wellbehaved)
  KERNEL(at::equal, "aten::equal(Tensor self, Tensor other) -> bool", bool (const Tensor &, const Tensor &), promote, wellbehaved)
  ;
}
#endif

} // namespace autocast
} // namespace at
