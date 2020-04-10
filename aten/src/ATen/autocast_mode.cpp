#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <iostream>
#include <exception>

namespace at {
namespace autocast {

bool is_enabled() {
  return c10::impl::tls_is_dispatch_key_included(DispatchKey::Autocast);
}

void set_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_included(DispatchKey::Autocast, new_enabled);
}

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache fp16 casts of fp32 model weights (see cached_cast below).
//
// After discussion with @ezyang, the cache uses the following structure:
// The key is the source tensor's TensorImpl*, a proxy for a Tensor uuid that's unchanged
// across shallow copies.  The value is a tuple with a weakref to the source tensor's
// TensorImpl as the first element and the casted tensor as the second element.
//
// The weakref keeps the source's TensorImpl from being deleted.  We need to because we're
// using the source TensorImpl* as the key.  If it were deleted, another random Tensor could
// be allocated whose TensorImpl* happened to have the same value.  This TensorImpl* would
// then mistakenly hit in cache:  a rare, intermittent, unpredictable bug.
//
// I'm not using the weak_intrusive_ptr as the key because it's more difficult to compare
// directly against incoming TensorImpl*s.
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, Tensor>;
thread_local std::unordered_map<TensorImpl*, val_type> cached_casts;

// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the autocasting region.
thread_local int nesting = 0;
}

void clear_cache() {
  cached_casts.clear();
}

int increment_nesting() {
  return ++nesting;
}

int decrement_nesting() {
  return --nesting;
}

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  fp16 = 0, // Cast all inputs to at::kHalf before running the op.
  fp32, // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //   1. we'd like to run in fp32 and
                      //   2. have a c10::optional<ScalarType> arg that controls the output type.
                      // fp32_set_opt_dtype wrappers' policy is:  if the output type is already set,
                      // don't touch it, otherwise, set it to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //   1. we'd like to run in fp32 and
                     //   2. have some overloads that accept an output type and other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't have an output dtype.
                     // The wrapper policy is:  append at::kFloat to the args, and redispatch to the
                     // type-aware overload.
  promote, // Run in the widest dtype among several args.
};

/********************************************************************
Logic to extract the promote type from any Tensor or TensorList args.
********************************************************************/

// Overload to catch Tensor args.
// If nextArg is floating-point, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
inline at::ScalarType prioritize(at::ScalarType current, const Tensor& nextArg) {
  if (current == at::kDouble) {
    AT_ERROR("promote type is double in at::autocast::prioritize");
    return current;
  }
  if (nextArg.is_cuda() && nextArg.is_floating_point()) {
    auto next = nextArg.scalar_type();
    if (next == at::kDouble) {
      return current; // ignores double tensors
    } else if (current == at::kFloat || next == at::kFloat) {
      return at::kFloat; // prioritizes float over half
    } else if (current == at::kHalf && next == at::kHalf) {
      return at::kHalf;
    } else {
      AT_ERROR("Unexpected floating ScalarType in at::autocast::prioritize");
      return current;
    }
  } else {
    return current;
  }
}

// Overload to catch TensorList args (for e.g. cat, stack).
// Reuses the overload above to process each Tensor in the list.
inline at::ScalarType prioritize(at::ScalarType current, const TensorList& list) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor);
  }
  return current;
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template<typename T>
inline at::ScalarType prioritize(at::ScalarType current, T nextArg) {
  return current;
}

// Overload for the tail case.
inline at::ScalarType promote_type(at::ScalarType current) {
  return current;
}

// Unpack args and determine if incoming float16 tensors need to be promoted to float32.
// Non-Tensor arguments are ignored.
template<typename Arg0, typename... Args>
inline at::ScalarType promote_type(at::ScalarType current, Arg0 arg0, Args... args) {
  auto new_current = prioritize(current, arg0);
  return promote_type(new_current, args...);
}

/****************************************************
Logic to apply cached casting to any Tensor argument.
****************************************************/
inline bool is_eligible(const Tensor& arg) {
  return (arg.is_cuda() && arg.is_floating_point() && (arg.scalar_type() != at::kDouble));
}

// Overload to catch Tensor args
inline Tensor cached_cast(at::ScalarType to_type, const Tensor& arg) {
  if (is_eligible(arg) && (arg.scalar_type() != to_type)) {
    // Heuristic:  Do what Apex does, and cache fp16 casts of fp32 model weights (leaves).
    // See cached_casts declaration above for detailed strategy.
    bool can_try_cache = (to_type == at::kHalf && arg.scalar_type() == at::kFloat && arg.requires_grad() && arg.is_leaf());
    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      } else {
        auto casted_arg = arg.to(to_type);
        cached_casts.emplace(arg.unsafeGetTensorImpl(), val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

// Overload to process TensorLists
std::vector<Tensor> cached_cast(at::ScalarType to_type, const TensorList& arg) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cached_cast(to_type, t));
  }
  return vec;
}

// Template to catch non-Tensor args.
template<typename T>
inline T cached_cast(at::ScalarType to_type, T arg) {
  return arg;
}

/*******************************************************
Logic to flip an output dtype flag.
Keep it simple for now by assuming only one such flag is
present in the argument list.  If I ever need a function
with more than flag I'll figure out something else.
The policy is:
If the user has explicity specified a dtype, respect it.
Otherwise, set it to the autocast type.
********************************************************/

// Overload to catch dtype flags
inline c10::optional<ScalarType> set_opt_dtype(at::ScalarType to_type, const c10::optional<ScalarType>& dtype) {
  return dtype.has_value() ? dtype : to_type;
}

// Template to catch other args
template<typename T>
inline T set_opt_dtype(at::ScalarType to_type, T arg) {
  return arg;
}

template<typename... Args>
inline bool firstarg_is_eligible(const Tensor& arg, Args... args) {
  return is_eligible(arg);
}

template<typename... Args>
inline at::ScalarType type_from_firstarg(at::ScalarType to_type, const Tensor& arg, Args... args) {
  return (is_eligible(arg) ? to_type : arg.scalar_type());
}

/********************************************************************************************************
Templates to provide wrapper functions

I'm copying the pattern used in core/boxing/impl/WrapFunctionIntoFunctor.h to extract args and return type.
(see also https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)

This strategy uses an exterior "WrapFunction" that extracts arguments on behalf of
(in my case several specializations of) an interior "WrapFunction_".
Interior WrapFunction_ specializations are defined for each CastPolicy.
********************************************************************************************************/

// Base template for WrapFunction_, which is specialized to contain a "call" method each CastPolicy
template<CastPolicy policy, class Redispatch, Redispatch* F, class Ret, class ArgList> struct WrapFunction_ {};

// CastPolicy::fp16
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp16, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocasting(DispatchKey::Autocast);
    return (*F)(cached_cast(at::kHalf, args)...);
  }
};

// CastPolicy::fp32
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocasting(DispatchKey::Autocast);
    return (*F)(cached_cast(at::kFloat, args)...);
  }
};

// CastPolicy::fp32_set_opt_dtype
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_set_opt_dtype, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocasting(DispatchKey::Autocast);
    if (firstarg_is_eligible(args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype, because setting
      // opt dtype explicitly may interfere with internal implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_append_dtype, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocasting(DispatchKey::Autocast);
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::promote, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocasting(DispatchKey::Autocast);
    auto to_type = promote_type(at::kHalf, args...);
    return (*F)(cached_cast(to_type, args)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating core/boxing/impl/WrapFunctionIntoFunctor.h)
template<CastPolicy policy,
         class Registered, // The signature for which we're registering.  The dispatcher's calling code invokes our
                           // registered functions with arguments matching Registered, so we register
                           // WrapFunction_::call methods with a matching signature to properly field those arguments.
                           // guts::function_traits below extracts return_type and parameter_types from Registered,
                           // which WrapFunction_ templates above use to declare their call methods.
         class Redispatch, // The signature for the function we're redispatching to.  In most cases this is the same
                           // as Registered, but for some ops (for example, ops where we append a dtype) it's useful
                           // to redispatch to a function with a different signature.
         Redispatch* F>    // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<policy,
                             Redispatch,
                             F,
                             typename guts::function_traits<Registered>::return_type,
                             typename guts::function_traits<Registered>::parameter_types>;
};

/*******************************
Banned functions
*******************************/

Tensor binary_cross_entropy_banned(const Tensor &, const Tensor &, const Tensor &, int64_t) {
  AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
           "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
           "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
           "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
           "safe to autocast.");
}

#ifndef USE_STATIC_DISPATCH
namespace {
/*****************************************************************************************************************
This section performs load-time registration for autocast wrappers.

It's debatable at what level operations should be patched.  We'd like casts to be autograd-exposed
and precede autograd history recording, so that for fp16 ops, input tensors are saved for backward
in fp16 rather than fp32.  Saving inputs in fp16 can significantly reduce a model's memory footprint.

Option 1 (strawman):  Patch only at the level of explicit calls into cudnn/cublas (cudnn_convolution, etc),
because those are the code paths that are guaranteed to use Tensor Cores, therefore they're the ones that
will benefit most from fp16.   Potential pitfall:  convolutions (and other ops) are wrapped in several
layers of at::* calls.  If one of those happens to record autograd history, then we've lost the
opportunity to save inputs in fp16.

Option 2:  Patch the Python-exposed surface of calls, to make 100% sure autograd history
recording can't sneak in ahead of autocast.  This mirrors Apex most closely.

I think Option 2 is the right answer for all ops, not just convolutions.  Option 2 is what I implement here.
*****************************************************************************************************************/

auto register_fallthrough = c10::import()
  .fallback(c10::DispatchKey::Autocast, c10::CppFunction::makeFallthrough());

/********************************************************************************************************************
Explicit registration for out-of-place ops

The stuff below could be codegenned.  Ed said
> you are going to have to write the function definition at some point, I wouldn't try to get clever about it
Therefore, for the moment, this is all copy pasted in from VariableTypeEverything.cpp with appropriate substitutions.
********************************************************************************************************************/

// Workaround for a compiler bug in VS 2017 (versions < 15.8).  See comments in autocast_VS2017_helper.h.
#ifdef _MSC_VER
  #if _MSC_VER >= 1915
    // With VS 15.8+, template directly on at:: functions.
    #define ADD_NS(RAW_OP) at::RAW_OP
  #else
    // If we're compiling with the buggy VS, pull in local wrappers to template on.
    #include <ATen/autocast_VS2017_helper.h>
    #define ADD_NS(RAW_OP) autocastVS2017Helper::RAW_OP
  #endif
#else
  // With other compilers, template directly on at:: functions.
  #define ADD_NS(RAW_OP) at::RAW_OP
#endif

// Common cases where registration signature matches redispatch signature
// (that's why SIGNATURE is repeated in the WrapFunction instantiation)
#define KERNEL(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  .impl(REGISTER_NAME, DispatchKey::Autocast, \
    &WrapFunction<CastPolicy::POLICY, SIGNATURE, SIGNATURE, &FUNC>::type::call)

#define KERNEL_UNBOXED_ONLY(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  .impl_UNBOXED(REGISTER_NAME, DispatchKey::Autocast, \
    &WrapFunction<CastPolicy::POLICY, SIGNATURE, SIGNATURE, &FUNC>::type::call)

// Less-common but still useful case: redispatching to a function with a new signature (e.g. appending a dtype)
#define KERNEL_UNBOXED_ONLY_DIFFERENT_REDISPATCH_SIGNATURE(REDISPATCH_FUNC, REGISTER_NAME, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, POLICY) \
  .impl_UNBOXED(REGISTER_NAME, DispatchKey::Autocast, \
    &WrapFunction<CastPolicy::POLICY, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, &REDISPATCH_FUNC>::type::call)

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
auto register_out_of_place = c10::import()
  // fp16
  KERNEL_UNBOXED_ONLY(ADD_NS(_convolution), "aten::_convolution", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(_convolution_nogroup), "aten::_convolution_nogroup", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv1d), "aten::conv1d", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv2d), "aten::conv2d", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv3d), "aten::conv3d", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv_tbc), "aten::conv_tbc", Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv_transpose1d), "aten::conv_transpose1d", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv_transpose2d), "aten::conv_transpose2d.input", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(conv_transpose3d), "aten::conv_transpose3d.input", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(convolution), "aten::convolution", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(cudnn_convolution), "aten::cudnn_convolution.deprecated", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(cudnn_convolution_transpose), "aten::cudnn_convolution_transpose.deprecated", Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(cudnn_convolution), "aten::cudnn_convolution", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(cudnn_convolution_transpose), "aten::cudnn_convolution_transpose", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), fp16)
  KERNEL(ADD_NS(prelu), "aten::prelu", Tensor (const Tensor &, const Tensor &), fp16)
  KERNEL(ADD_NS(addmm), "aten::addmm", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16)
  KERNEL(ADD_NS(addmv), "aten::addmv", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16)
  KERNEL(ADD_NS(addr), "aten::addr", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16)
  KERNEL(ADD_NS(matmul), "aten::matmul", Tensor (const Tensor &, const Tensor &), fp16)
  KERNEL(ADD_NS(mm), "aten::mm", Tensor (const Tensor &, const Tensor &), fp16)
  KERNEL(ADD_NS(mv), "aten::mv", Tensor (const Tensor &, const Tensor &), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(linear), "aten::linear", Tensor (const Tensor &, const Tensor &, const Tensor &), fp16)
  KERNEL(ADD_NS(addbmm), "aten::addbmm", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16)
  KERNEL(ADD_NS(baddbmm), "aten::baddbmm", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar), fp16)
  KERNEL(ADD_NS(bmm), "aten::bmm", Tensor (const Tensor &, const Tensor &), fp16)
  KERNEL_UNBOXED_ONLY(ADD_NS(chain_matmul), "aten::chain_matmul", Tensor (TensorList), fp16)
  // fp32
  KERNEL(ADD_NS(acos), "aten::acos", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(asin), "aten::asin", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(cosh), "aten::cosh", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(erfinv), "aten::erfinv", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(exp), "aten::exp", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(expm1), "aten::expm1", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log), "aten::log", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log10), "aten::log10", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log2), "aten::log2", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log1p), "aten::log1p", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(reciprocal), "aten::reciprocal", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(rsqrt), "aten::rsqrt", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(sinh), "aten::sinh", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(tan), "aten::tan", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(pow), "aten::pow.Tensor_Scalar", Tensor (const Tensor &, Scalar), fp32)
  KERNEL(ADD_NS(pow), "aten::pow.Tensor_Tensor", Tensor (const Tensor &, const Tensor &), fp32)
  KERNEL(ADD_NS(pow), "aten::pow.Scalar", Tensor (Scalar, const Tensor &), fp32)
  KERNEL(ADD_NS(softplus), "aten::softplus", Tensor (const Tensor &, Scalar, Scalar), fp32)
  KERNEL(ADD_NS(gelu), "aten::gelu", Tensor (const Tensor &), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(layer_norm), "aten::layer_norm", Tensor (const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool), fp32)
  // The macro doesn't like this one so I had to write it out manually.
  .impl_UNBOXED("aten::native_layer_norm", DispatchKey::Autocast,
                &WrapFunction<CastPolicy::fp32, std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double), std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double), &ADD_NS(native_layer_norm)>::type::call)
  KERNEL_UNBOXED_ONLY(ADD_NS(group_norm), "aten::group_norm", Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(frobenius_norm), "aten::frobenius_norm", Tensor (const Tensor &), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(frobenius_norm), "aten::frobenius_norm.dim", Tensor (const Tensor &, IntArrayRef, bool), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(nuclear_norm), "aten::nuclear_norm", Tensor (const Tensor &, bool), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(nuclear_norm), "aten::nuclear_norm.dim", Tensor (const Tensor &, IntArrayRef, bool), fp32)
  KERNEL(ADD_NS(cosine_similarity), "aten::cosine_similarity", Tensor (const Tensor &, const Tensor &, int64_t, double), fp32)
  KERNEL(ADD_NS(poisson_nll_loss), "aten::poisson_nll_loss", Tensor (const Tensor &, const Tensor &, bool, bool, double, int64_t), fp32)
  KERNEL(ADD_NS(cosine_embedding_loss), "aten::cosine_embedding_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(nll_loss), "aten::nll_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(nll_loss2d), "aten::nll_loss2d", Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t), fp32)
  KERNEL(ADD_NS(hinge_embedding_loss), "aten::hinge_embedding_loss", Tensor (const Tensor &, const Tensor &, double, int64_t), fp32)
  KERNEL(ADD_NS(kl_div), "aten::kl_div", Tensor (const Tensor &, const Tensor &, int64_t, bool), fp32)
  KERNEL(ADD_NS(l1_loss), "aten::l1_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(smooth_l1_loss), "aten::smooth_l1_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(mse_loss), "aten::mse_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(margin_ranking_loss), "aten::margin_ranking_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t), fp32)
  KERNEL(ADD_NS(multilabel_margin_loss), "aten::multilabel_margin_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(soft_margin_loss), "aten::soft_margin_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(triplet_margin_loss), "aten::triplet_margin_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(multi_margin_loss), "aten::multi_margin_loss", Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(binary_cross_entropy_with_logits), "aten::binary_cross_entropy_with_logits", Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(dist), "aten::dist", Tensor (const Tensor &, const Tensor &, Scalar), fp32)
  KERNEL(ADD_NS(pdist), "aten::pdist", Tensor (const Tensor &, double), fp32)
  KERNEL_UNBOXED_ONLY(ADD_NS(cdist), "aten::cdist", Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>), fp32)
  KERNEL(ADD_NS(renorm), "aten::renorm", Tensor (const Tensor &, Scalar, int64_t, Scalar), fp32)
  // fp32_set_opt_dtype
  KERNEL_UNBOXED_ONLY(ADD_NS(prod), "aten::prod", Tensor (const Tensor &, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(prod), "aten::prod.dim_int", Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(prod), "aten::prod.dim_Dimname", Tensor (const Tensor &, Dimname, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(softmax), "aten::softmax.int", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(softmax), "aten::softmax.Dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(log_softmax), "aten::log_softmax.int", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(log_softmax), "aten::log_softmax.Dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(cumprod), "aten::cumprod", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(cumprod), "aten::cumprod.dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(cumsum), "aten::cumsum", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(cumsum), "aten::cumsum.dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL_UNBOXED_ONLY(ADD_NS(norm), "aten::norm.ScalarOpt_dtype", Tensor (const Tensor &, c10::optional<Scalar>, ScalarType), fp32_set_opt_dtype)
  // KERNEL_UNBOXED_ONLY(ADD_NS(norm), "aten::norm.ScalarOpt_dim_dtype", Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType), fp32_set_opt_dtype)
  // KERNEL_UNBOXED_ONLY(ADD_NS(norm), "aten::norm.names_ScalarOpt_dim_dtype", Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(sum), "aten::sum", Tensor (const Tensor &, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(sum), "aten::sum.dim_IntList", Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL_UNBOXED_ONLY(ADD_NS(sum), "aten::sum.dim_DimnameList", Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  KERNEL_UNBOXED_ONLY_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "aten::norm.Scalar", Tensor (const Tensor &, Scalar), Tensor (const Tensor &, c10::optional<Scalar>, ScalarType), fp32_append_dtype)
  KERNEL_UNBOXED_ONLY_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "aten::norm.ScalarOpt_dim", Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool), Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType), fp32_append_dtype)
  KERNEL_UNBOXED_ONLY_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "aten::norm.names_ScalarOpt_dim", Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool), Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType), fp32_append_dtype)
  // promote
  KERNEL(ADD_NS(addcdiv), "aten::addcdiv", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), promote)
  KERNEL(ADD_NS(addcmul), "aten::addcmul", Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar), promote)
  KERNEL(ADD_NS(atan2), "aten::atan2", Tensor (const Tensor &, const Tensor &), promote)
  KERNEL(ADD_NS(cross), "aten::cross", Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(bilinear), "aten::bilinear", Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(tensordot), "aten::tensordot", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(dot), "aten::dot", Tensor (const Tensor &, const Tensor &), promote)
  KERNEL(ADD_NS(equal), "aten::equal", bool (const Tensor &, const Tensor &), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(cat), "aten::cat", Tensor (TensorList, int64_t), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(cat), "aten::cat.names", Tensor (TensorList, Dimname), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(_cat), "aten::_cat", Tensor (TensorList, int64_t), promote)
  KERNEL_UNBOXED_ONLY(ADD_NS(stack), "aten::stack", Tensor (TensorList, int64_t), promote)
  ;

auto register_banned = torch::import()
  .impl_UNBOXED("aten::binary_cross_entropy", DispatchKey::Autocast,
                &at::autocast::binary_cross_entropy_banned);
}
#endif

} // namespace autocast
} // namespace at
