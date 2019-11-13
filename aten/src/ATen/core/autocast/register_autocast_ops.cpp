#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

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

// feels good to be writing C++ again

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
    bool can_try_cache = (to_type == at::kHalf && arg.scalar_type() == at::kFloat);
    // If arg is a variable, explicitly check is_leaf().  Non-variable Tensors do not permit an is_leaf() call,
    // so if arg is not a variable, assume it is a frozen leaf.
    if (arg.is_variable()) {
      can_try_cache = (can_try_cache || arg.is_leaf());
    }
    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl()); // See cached_casts declaration for detailed strategy.
      if (it != cached_casts.end()) {
        return std::get<1>(it->second); // Return the cached cast.
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

#ifndef USE_STATIC_DISPATCH
namespace {
// Static registration for kernels that require autocast.

// Debatable at which level the convolutions should be patched.
// I'll patch at the top (Python-exposed) level, because then I'm 100% sure
// autograd history recording can't sneak in ahead of me.

// Or maybe we only want to patch cudnn_convolution, because those are the only ones that will use
// Tensor Cores, and therefore they're the only ones that should be whitelisted?  I think that's
// actually the right answer.  I can include checks for all outputs that their grad_fns saved inputs
// in the expected/desired types.

#define PATCH(FUNC, POLICY) &Patch<CastPolicy::POLICY, decltype( FUNC ), FUNC>::call
// example:
// PATCH(at::cudnn_convolution, fp16)
// becomes
// &Patch<CastPolicy::fp16, decltype( at::cudnn_convolution ), at::cudnn_convolution>::call
// I guess I could have used a complex set of macros instead of templates above.

auto registerer = torch::RegisterOperators()
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
