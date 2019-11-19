#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
// TODO:  Remove this dependency once callBoxed is possible for all ops
#include <torch/csrc/jit/operator.h>

namespace at {
namespace autocast {

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache FP16 casts of FP32 model weights (see caching_caster below).
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
  promote
};

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

// Overload to catch Tensor args, and cast if necessary.
Tensor caching_caster(at::ScalarType to_type, const Tensor & arg) {
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
T caching_caster(at::ScalarType to_type, T arg) {
  return arg;
}

// https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer
// I could also use c10::guts::function_traits for this as in ATen/core/boxing/kernel_function.h, but
// that seems no less verbose.
template <CastPolicy policy, typename F, F* func> struct Patch {};

template <CastPolicy policy, typename Ret, typename... Args, auto (*F)(Args...) -> Ret>
struct Patch<policy, Ret (Args...), F> {
  static Ret call(Args... args) {
    auto to_type = at::kHalf;
    switch(policy) {
      case CastPolicy::fp16:
        to_type = at::kHalf;
        break;
      case CastPolicy::fp32:
        to_type = at::kFloat;
        break;
      case CastPolicy::promote:
        to_type = promote_type(to_type, args...);
        break;
      default:
        AT_ERROR("Unexpected CastPolicy in autograd::Patch");
    }
    c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
    return F(caching_caster(to_type, args)...);
  }
};

// void callBoxedWorkaround(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
//   auto s = Symbol::fromQualString(op.schema().name());
//   auto operators = torch::jit::getAllOperatorsFor(s);
//   // Find the exact match
//   std::shared_ptr<torch::jit::Operator> jit_op;
//   for (const auto& candidate_op : operators) {
//     auto candidate_schema = candidate_op->schema();
//     // NB: this is a VERY slow equality test
//     if (candidate_schema == op.schema()) {
//       jit_op = candidate_op;
//       break;
//     }
//   }
//   TORCH_INTERNAL_ASSERT(jit_op);
//
//   auto offset = jit_op->getOperation()(*stack);
//   TORCH_INTERNAL_ASSERT(offset == 0);
// }
//
// // The fallback for greylist ops disables autocasting and redispatches.
// void autocast_fallback(c10::OperatorKernel*, const c10::OperatorHandle& op, c10::Stack* stack) {
//   c10::impl::ExcludeTensorTypeIdGuard no_autocasting(TensorTypeId::AutocastTensorId);
//   // Temporary workaround.
//   // TODO:  Replace callBoxedWorkaround with op.callBoxed(stack) once callBoxed is possible for all ops.
//   callBoxedWorkaround(op, stack);
// }
// // TODO:  Once fallthrough is implemented (https://github.com/pytorch/pytorch/issues/29548),
// // autocast_fallback can be deleted entirely.

#ifndef USE_STATIC_DISPATCH
namespace {
// Static registration for kernels that require autocast.

// It's debatable at which level the convolutions should be patched.
//
// Option 1:  Patch at the Python-exposed level, to make 100% sure autograd history
// recording can't sneak in ahead of autocast.  This mirrors Apex most closely.
//
// Option 2:  Patch only at the level of explicit calls into cudnn (cudnn_convolution, etc),
// because those are the code paths that are guaranteed to use Tensor Cores, therefore they're the only ones
// that should be whitelisted.  One difficulty with this approach is that convolutions (and other ops) are wrapped
// in several layers of at::* calls.  Several non-explicitly-registered layers (e.g. at::convolution) could be called
// before we reach cudnn_convolution.  Because they are not explicitly registered, these layers would invoke the
// boxed fallback, which would exclude AutocastTensorId, so by the time at::cudnn_convolution is actually
// called, it wouldn't route back through the autocasting logic at all.
//
// I think Option 1 is the right answer for all ops, not just convolutions.

// Boxed fallback registration

// auto register_fallback = c10::Dispatcher::singleton()
//   .registerBackendFallbackKernel(TensorTypeId::AutocastTensorId,
//                                  KernelFunction::makeFromBoxedFunction(&autocast_fallback));
//                     // change to KernelFunction::makeFromBoxedFunction<&autocast_fallback>()
//                     // once https://github.com/pytorch/pytorch/pull/29337/files goes in

// Explicit registration for fp32/fp16/promote ops

#define PATCH(FUNC, POLICY) &Patch<CastPolicy::POLICY, decltype( FUNC ), FUNC>::call
// example:
// PATCH(at::cudnn_convolution, fp16)
// becomes
// &Patch<CastPolicy::fp16, decltype( at::cudnn_convolution ), at::cudnn_convolution>::call
// I guess I could have used a complex set of macros instead of templates above.

auto register_explicit = torch::RegisterOperators()
  /* FP16 OPS */
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool),
                            PATCH(at::cudnn_convolution, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>),
                            PATCH(at::cudnn_convolution_backward, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_backward_bias(Tensor grad_output) -> Tensor")
    .kernel<Tensor (const Tensor &)>
           (TensorTypeId::AutocastTensorId,
            PATCH(at::cudnn_convolution_backward_bias, fp16))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool),
                            PATCH(at::cudnn_convolution_backward_input,fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool),
                            PATCH(at::cudnn_convolution_backward_weight, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool),
                            PATCH(at::cudnn_convolution_transpose, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
    .impl_unboxedOnlyKernel<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>),
                            PATCH(at::cudnn_convolution_transpose_backward, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose_backward_bias(Tensor grad_output) -> Tensor")
    .kernel<Tensor (const Tensor &)>
           (TensorTypeId::AutocastTensorId,
            PATCH(at::cudnn_convolution_transpose_backward_bias, fp16))
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool),
                            PATCH(at::cudnn_convolution_transpose_backward_input, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op(torch::RegisterOperators::options()
    .schema("aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")
    .impl_unboxedOnlyKernel<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool),
                            PATCH(at::cudnn_convolution_transpose_backward_weight, fp16)>
                           (TensorTypeId::AutocastTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
  /* FP32 OPS */
  /* PROMOTE OPS */
}
#endif

} // namespace autocast
} // namespace at
