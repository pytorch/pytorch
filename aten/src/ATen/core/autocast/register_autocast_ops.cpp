#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
// TODO:  Remove this dependency once callBoxed is possible for all ops
#include <torch/csrc/jit/operator.h>

#include <iostream>

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
Templates for well-behaved ops

Well-behaved means
- The op has an at:: exposure, AND
- if the op requires casting or promotion, all Tensor arguments are
  received as "const Tensor &", which means the op does not modify
  any Tensor arguments in-place.

CastPolicy::passthrough ops (which don't cast or promote) may
receive non-const Tensor & arguments and remain well-behaved as
long as they have an at:: exposure.

Fortunately, most of the ops I need to treat are well-behaved.
******************************************************************/

// Trick to extract args and return type from arbitrary function type:
// https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer
// I could also use c10::guts::function_traits for this as in ATen/core/boxing/kernel_function.h, but
// that seems no less verbose.
template <CastPolicy policy, typename F, F* func> struct Patch {};

// Separate struct specializations for the four CastPolicies so the wrapper instantiation for each op
// only compiles with the type selection logic it actually needs (ie, promote_type is only used by
// CastPolicy::promote ops).

// CastPolicy::fp16
template <typename Ret, typename... Args, auto (*F)(Args...) -> Ret>
struct Patch<CastPolicy::fp16, Ret (Args...), F> {
  static Ret call(Args... args) {
    auto to_type = at::kHalf;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return F(cached_cast(to_type, args)...);
  }
};

// CastPolicy::fp32
template <typename Ret, typename... Args, auto (*F)(Args...) -> Ret>
struct Patch<CastPolicy::fp32, Ret (Args...), F> {
  static Ret call(Args... args) {
    auto to_type = at::kFloat;
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return F(cached_cast(to_type, args)...);
  }
};

// CastPolicy::promote
template <typename Ret, typename... Args, auto (*F)(Args...) -> Ret>
struct Patch<CastPolicy::promote, Ret (Args...), F> {
  static Ret call(Args... args) {
    auto to_type = promote_type(at::kHalf, args...);
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return F(cached_cast(to_type, args)...);
  }
};

// The passthrough specialization supplies explicitly-registerable unboxed kernels
// to serve ops that meet all of the following:
// - don't require casting, so in principle they should use the boxed fallback
// - don't play well with the boxed fallback, unfortunately
// - do play well with the templating logic
template <typename Ret, typename... Args, auto (*F)(Args...) -> Ret>
struct Patch<CastPolicy::passthrough, Ret (Args...), F> {
  static Ret call(Args... args) {
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return F(args...);
  }
};

/*****************************************************************
Special treatment for ops that aren't well-behaved

Generally, these ops fall into three categories:
- Ops that modify Tensor arguments in-place
- Ops that write to a user-supplied `out=...` buffer
- Ops that (for whatever reason) don't have an at:: exposure

There seems to be a correlation (perhaps coincidental) between
not having an at::* exposure and being in-place.

I could try writing additional templates for these black-sheep ops,
but for now, I'll special-case them manually, imitating
VariableType*.cpp.  Hopefully there aren't many.
******************************************************************/

// In-place ops
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
  auto to_type = at::kHalf;
  if (self.scalar_type() == at::kHalf) {
    self.addmm_(cached_cast(to_type, mat1),
                cached_cast(to_type, mat2),
                cached_cast(to_type, beta),
                cached_cast(to_type, alpha));
  } else {
    auto fp16_result = at::addmm(cached_cast(to_type, self),
                                 cached_cast(to_type, mat1),
                                 cached_cast(to_type, mat2),
                                 cached_cast(to_type, beta),
                                 cached_cast(to_type, alpha));
    self.copy_(fp16_result);
  }
  return self;
}

// Functions with out=... arguments
// These don't support automatic differentiation, so that's not something we need to worry about.
Tensor & addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
  auto to_type = at::kHalf;
  if (out.scalar_type() == to_type) {
    at::addmm_out(out,
                  cached_cast(to_type, self),
                  cached_cast(to_type, mat1),
                  cached_cast(to_type, mat2),
                  cached_cast(to_type, beta),
                  cached_cast(to_type, alpha));
  } else {
    AT_ASSERT(!out.requires_grad());
    auto fp16_result = at::empty_like(out, out.options().dtype(to_type));
    at::addmm_out(fp16_result,
                  cached_cast(to_type, self),
                  cached_cast(to_type, mat1),
                  cached_cast(to_type, mat2),
                  cached_cast(to_type, beta),
                  cached_cast(to_type, alpha));
    out.copy_(fp16_result);
  }
  return out;
}

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

/*****************************************
Explicit registration for well-behaved ops
*****************************************/

#define PATCH_FUNCTION(FUNC, POLICY) &Patch<CastPolicy::POLICY, decltype( FUNC ), FUNC>::call
// example:
// PATCH_FUNCTION(at::cudnn_convolution, fp16)
// becomes
// &Patch<CastPolicy::fp16, decltype( at::cudnn_convolution ), at::cudnn_convolution>::call

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
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool), PATCH_FUNCTION(at::_convolution, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef), PATCH_FUNCTION(at::_convolution_nogroup, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), PATCH_FUNCTION(at::conv1d, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), PATCH_FUNCTION(at::conv2d, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), PATCH_FUNCTION(at::conv3d, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>(TensorTypeId::AutocastTensorId, PATCH_FUNCTION(at::conv_tbc, fp16))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), PATCH_FUNCTION(conv_transpose1d, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), PATCH_FUNCTION(conv_transpose2d, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), PATCH_FUNCTION(conv_transpose3d, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t), PATCH_FUNCTION(at::convolution, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), PATCH_FUNCTION(at::cudnn_convolution, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), PATCH_FUNCTION(at::cudnn_convolution_transpose, fp16)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::prelu(Tensor self, Tensor weight) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &)>(TensorTypeId::AutocastTensorId, PATCH_FUNCTION(at::prelu, fp16))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")
    .kernel<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(TensorTypeId::AutocastTensorId, PATCH_FUNCTION(at::addmm, fp16))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // CastPolicy::fp32
  // CastPolicy::promote
  // CastPolicy::passthrough
  .op(torch::RegisterOperators::options()
    .schema("aten::detach_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), PATCH_FUNCTION(at::detach_, passthrough)>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  ;

/**************************************************************************************
Registration for non-well-behaved ops part 1:  in-place ops

It's not technically required to register these separately, but it helps organize them.
**************************************************************************************/
auto register_inplace = torch::RegisterOperators()
  // fp16 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &at::autocast::addmm_>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // fp32 ops
  // promote ops
  ;

/*********************************************************************************************
Registration for non-well-behaved ops part 2:  ops that write to a user-supplied output buffer
*********************************************************************************************/
auto register_user_supplied_output = torch::RegisterOperators()
  // fp16 ops
  .op(torch::RegisterOperators::options()
    .schema("aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), &at::autocast::addmm_out>(TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  // fp32 ops
  // promote ops
  ;
}
#endif

} // namespace autocast
} // namespace at
