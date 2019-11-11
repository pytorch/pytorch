#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace autocast {

namespace {
// Establish the same cast-caching policy as Apex (cache FP16 casts of FP32 leaves).
// The key type is a proxy for a Tensor uuid.
// The value type is Tensor to make sure cached Tensors stay alive.
thread_local std::unordered_map<TensorImpl*, Tensor> cached_leaf_casts;
}

std::unordered_map<TensorImpl*, Tensor> & get_cache() {
  return cached_leaf_casts;
}

enum class CastPolicy : uint8_t {
  fp16 = 0,
  fp32,
  promote
};

// feels good to be writing C++ again

// If nextTensor is a floating-point Tensor, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
at::ScalarType prioritize(at::ScalarType current, const Tensor & nextTensor) {
  if (!nextTensor.is_floating_point()) {
    return current;
  } else {
    auto next = nextTensor.scalar_type();
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

// Catchall:: If an arg is not a Tensor, don't bother trying to read its type.
template<typename T>
at::ScalarType prioritize(at::ScalarType current, T nextArg) {
  return current;
}

// Simple overload for the tail case.
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

// Cast Tensor arg if appropriate.
Tensor caching_caster(at::ScalarType to_type, const Tensor & arg) {
  if (arg.is_floating_point() && arg.scalar_type() != to_type) {
    bool can_try_cache = (arg.scalar_type() == at::kFloat && to_type == at::kHalf);
    if (arg.is_variable()) {
      can_try_cache = (can_try_cache || arg.is_leaf());
    }
    if (can_try_cache) {
      auto it = cached_leaf_casts.find(arg.unsafeGetTensorImpl()); // Use the owned TensorImpl* as a Tensor's uuid.
      if (it != cached_leaf_casts.end()) {
        return it->second; // Return the cached value.
      } else {
        auto casted_arg = arg.to(to_type);
        cached_leaf_casts.emplace(arg.unsafeGetTensorImpl(), casted_arg);
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

// Catchall:  If an arg is not a Tensor, don't try to cast.
// Should we bother with perfect forwarding here?  I lean towards no, because types are deduced from the
// patched function's signature/schema, not from the args.  Besides, we know exactly what argument types
// we're dealing with.  Function signatures are specified explicitly in registration below.  If something
// looks like it might benefit from perfect forwarding, we can deal with that when the time comes.
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
}
#endif

} // namespace autocast
} // namespace at
