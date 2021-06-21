#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <iostream>
#include <exception>

namespace at {
namespace autocast {

bool is_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastCUDA);
}

void set_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastCUDA, !new_enabled);
}

bool is_cpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastCPU);
}

void set_cpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastCPU, !new_enabled);
}

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache lower_precision_fp casts of fp32 model weights (see cached_cast below).
//
// After discussion with @ezyang, the cache uses the following structure:
// The key is the fp32 source tensor's TensorImpl*, a proxy for a Tensor uuid that's
// unchanged across shallow copies.
// The value is a tuple with a weakref to the source tensor's TensorImpl as the first
// element and the casted tensor as the second element.
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::unordered_map<TensorImpl*, val_type> cached_casts;

// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the autocasting region.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local int nesting = 0;

// autocast_cpu_dtype is the lower_precision_fp used by AutocastCPU.
thread_local at::ScalarType autocast_cpu_dtype = at::kBFloat16;
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

at::ScalarType get_autocast_cpu_dtype() {
  return autocast_cpu_dtype;
}

void set_autocast_cpu_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::kBFloat16,
      "Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype");
  autocast_cpu_dtype = dtype;
}

// Overload to catch Tensor args
// TODO (possible optimization):
// Move cast_cache to an inline function in a header with cached_casts declared as
// extern thread_local in the header.
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // Heuristic:  Do what Apex does, and cache lower_precision_fp casts of fp32 model weights (leaves).
    // See cached_casts declaration above for detailed strategy.
    bool can_try_cache = (to_type == get_lower_precision_fp_from_device_type(device_type) &&
                         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
                         arg.is_leaf() && !arg.is_view());
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

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  lower_precision_fp = 0, // Cast all inputs to lower_precision_fp before running the op.
                          // Currently, lower_precision_fp is fp16 for AutocastCUDA, and is defined by user(default bf16) for AutocastCPU.
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

/********************************************************************************************************
Templates to provide wrapper functions

I'm copying the pattern used in core/boxing/impl/WrapFunctionIntoFunctor.h to extract args and return type.
(see also https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)

This strategy uses an exterior "WrapFunction" that extracts arguments on behalf of
(in my case several specializations of) an interior "WrapFunction_".
Interior WrapFunction_ specializations are defined for each CastPolicy.
********************************************************************************************************/

// Base template for WrapFunction_, which is specialized to contain a "call" method each CastPolicy
template<CastPolicy policy, DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class ArgList> struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::lower_precision_fp, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(get_lower_precision_fp_from_device_type(device_type), args, device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::fp32_set_opt_dtype DeviceType::CUDA
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_set_opt_dtype, DeviceType::CUDA, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);
    if (firstarg_is_eligible(args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype, because setting
      // opt dtype explicitly may interfere with internal implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype DeviceType::CUDA
template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_append_dtype, DeviceType::CUDA, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::promote, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type = promote_type(get_lower_precision_fp_from_device_type(device_type), device_type, args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating core/boxing/impl/WrapFunctionIntoFunctor.h)
template<CastPolicy policy,
         DeviceType device_type,
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
                             device_type,
                             Redispatch,
                             F,
                             typename guts::function_traits<Registered>::return_type,
                             typename guts::function_traits<Registered>::parameter_types>;
};

/*******************************
Banned functions
*******************************/

Tensor binary_cross_entropy_banned(const Tensor &, const Tensor &, const c10::optional<Tensor>&, int64_t) {
  AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
           "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
           "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
           "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
           "safe to autocast.");
}

namespace {
/*****************************************************************************************************************
This section performs load-time registration for autocast wrappers.

It's debatable at what level operations should be patched.  We'd like casts to be autograd-exposed
and precede autograd history recording, so that for lower_precision_fp ops, input tensors are saved for backward
in lower_precision_fp rather than fp32.  Saving inputs in lower_precision_fp can significantly reduce
a model's memory footprint.

Option 1 (strawman):  Patch only at the level of explicit calls into cudnn/cublas (cudnn_convolution, etc),
because those are the code paths that are guaranteed to use Tensor Cores, therefore they're the ones that
will benefit most from lower_precision_fp.   Potential pitfall:  convolutions (and other ops) are wrapped in several
layers of at::* calls.  If one of those happens to record autograd history, then we've lost the
opportunity to save inputs in lower_precision_fp.

Option 2:  Patch the Python-exposed surface of calls, to make 100% sure autograd history
recording can't sneak in ahead of autocast.  This mirrors Apex most closely.

I think Option 2 is the right answer for all ops, not just convolutions.  Option 2 is what I implement here.
*****************************************************************************************************************/

/********************************************************************************************************************
Explicit registration for out-of-place ops

The stuff below could be codegenned.  Ed said
> you are going to have to write the function definition at some point, I wouldn't try to get clever about it
Therefore, for the moment, this is all copy pasted in from VariableTypeEverything.cpp with appropriate substitutions.
********************************************************************************************************************/

#define ADD_NS(RAW_OP) at::RAW_OP

// Common cases where registration signature matches redispatch signature
// (that's why SIGNATURE is repeated in the WrapFunction instantiation)
#define KERNEL(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CUDA, SIGNATURE, SIGNATURE, &FUNC>::type::call);

// Less-common but still useful case: redispatching to a function with a new signature (e.g. appending a dtype)
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(REDISPATCH_FUNC, REGISTER_NAME, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CUDA, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, &REDISPATCH_FUNC>::type::call);

// KERNEL_CPU registration for AutocastCPU
#define KERNEL_CPU(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CPU, SIGNATURE, SIGNATURE, &FUNC>::type::call);

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, Autocast, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  // lower_precision_fp
  KERNEL(ADD_NS(_convolution), "_convolution.deprecated", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(_convolution), "_convolution", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(_convolution_nogroup), "_convolution_nogroup", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef), lower_precision_fp)
  KERNEL(ADD_NS(conv1d), "conv1d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL(ADD_NS(conv2d), "conv2d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL(ADD_NS(conv3d), "conv3d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL(ADD_NS(conv_tbc), "conv_tbc", Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t), lower_precision_fp)
  KERNEL(ADD_NS(conv_transpose1d), "conv_transpose1d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), lower_precision_fp)
  KERNEL(ADD_NS(conv_transpose2d), "conv_transpose2d.input", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), lower_precision_fp)
  KERNEL(ADD_NS(conv_transpose3d), "conv_transpose3d.input", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), lower_precision_fp)
  KERNEL(ADD_NS(convolution), "convolution", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL(ADD_NS(cudnn_convolution), "cudnn_convolution.deprecated", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(cudnn_convolution_transpose), "cudnn_convolution_transpose.deprecated", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(cudnn_convolution), "cudnn_convolution.deprecated2", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(cudnn_convolution_transpose), "cudnn_convolution_transpose.deprecated2", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(cudnn_convolution), "cudnn_convolution", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(cudnn_convolution_transpose), "cudnn_convolution_transpose", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, bool), lower_precision_fp)
  KERNEL(ADD_NS(prelu), "prelu", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL(ADD_NS(addmm), "addmm", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL(ADD_NS(addmv), "addmv", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL(ADD_NS(addr), "addr", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL(ADD_NS(matmul), "matmul", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL(ADD_NS(mm), "mm", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL(ADD_NS(mv), "mv", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL(ADD_NS(linear), "linear", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&), lower_precision_fp)
  KERNEL(ADD_NS(addbmm), "addbmm", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL(ADD_NS(baddbmm), "baddbmm", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL(ADD_NS(bmm), "bmm", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL(ADD_NS(chain_matmul), "chain_matmul", Tensor (TensorList), lower_precision_fp)
  KERNEL(ADD_NS(linalg_multi_dot), "linalg_multi_dot", Tensor (TensorList), lower_precision_fp)
  // The macro doesn't like these (I think it chokes on commas inside <>) so write them manually
  m.impl(TORCH_SELECTIVE_NAME("aten::_thnn_fused_lstm_cell"),
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, DeviceType::CUDA,
                                 std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 &ADD_NS(_thnn_fused_lstm_cell)>::type::call)));
  m.impl("_thnn_fused_gru_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, DeviceType::CUDA,
                                 std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 &ADD_NS(_thnn_fused_gru_cell)>::type::call)));
  m.impl("lstm_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, DeviceType::CUDA,
                                 std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 &ADD_NS(lstm_cell)>::type::call)));
  m.impl("gru_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, DeviceType::CUDA,
                                 Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 &ADD_NS(gru_cell)>::type::call)));
  m.impl("rnn_tanh_cell", // tanh unary op is executed as a cuda math library call.
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, DeviceType::CUDA,
                                 Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 &ADD_NS(rnn_tanh_cell)>::type::call)));
  m.impl("rnn_relu_cell",
         TORCH_FN((&WrapFunction<CastPolicy::lower_precision_fp, DeviceType::CUDA,
                                 Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&),
                                 &ADD_NS(rnn_relu_cell)>::type::call)));
  // fp32
  KERNEL(ADD_NS(acos), "acos", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(asin), "asin", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(cosh), "cosh", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(erfinv), "erfinv", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(exp), "exp", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(expm1), "expm1", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log), "log", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log10), "log10", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log2), "log2", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(log1p), "log1p", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(reciprocal), "reciprocal", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(rsqrt), "rsqrt", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(sinh), "sinh", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(tan), "tan", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(pow), "pow.Tensor_Scalar", Tensor (const Tensor &, const Scalar&), fp32)
  KERNEL(ADD_NS(pow), "pow.Tensor_Tensor", Tensor (const Tensor &, const Tensor &), fp32)
  KERNEL(ADD_NS(pow), "pow.Scalar", Tensor (const Scalar&, const Tensor &), fp32)
  KERNEL(ADD_NS(softplus), "softplus", Tensor (const Tensor &, const Scalar&, const Scalar&), fp32)
  KERNEL(ADD_NS(layer_norm), "layer_norm", Tensor (const Tensor &, IntArrayRef, const c10::optional<Tensor>&, const c10::optional<Tensor>&, double, bool), fp32)
  // The macro doesn't like this one (I think it chokes on commas inside <>) so write it manually
  m.impl(TORCH_SELECTIVE_NAME("aten::native_layer_norm"),
         TORCH_FN((&WrapFunction<CastPolicy::fp32, DeviceType::CUDA,
                                 std::tuple<Tensor,Tensor,Tensor> (const Tensor&, IntArrayRef, const c10::optional<Tensor>&, const c10::optional<Tensor>&, double),
                                 std::tuple<Tensor,Tensor,Tensor> (const Tensor&, IntArrayRef, const c10::optional<Tensor>&, const c10::optional<Tensor>&, double),
                                 &ADD_NS(native_layer_norm)>::type::call)));
  KERNEL(ADD_NS(group_norm), "group_norm", Tensor (const Tensor &, int64_t, const c10::optional<Tensor>&, const c10::optional<Tensor>&, double, bool), fp32)
  KERNEL(ADD_NS(frobenius_norm), "frobenius_norm", Tensor (const Tensor &), fp32)
  KERNEL(ADD_NS(frobenius_norm), "frobenius_norm.dim", Tensor (const Tensor &, IntArrayRef, bool), fp32)
  KERNEL(ADD_NS(nuclear_norm), "nuclear_norm", Tensor (const Tensor &, bool), fp32)
  KERNEL(ADD_NS(nuclear_norm), "nuclear_norm.dim", Tensor (const Tensor &, IntArrayRef, bool), fp32)
  KERNEL(ADD_NS(cosine_similarity), "cosine_similarity", Tensor (const Tensor &, const Tensor &, int64_t, double), fp32)
  KERNEL(ADD_NS(poisson_nll_loss), "poisson_nll_loss", Tensor (const Tensor &, const Tensor &, bool, bool, double, int64_t), fp32)
  KERNEL(ADD_NS(cosine_embedding_loss), "cosine_embedding_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t), fp32)
  KERNEL(ADD_NS(nll_loss), "nll_loss", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, int64_t, int64_t), fp32)
  KERNEL(ADD_NS(nll_loss2d), "nll_loss2d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, int64_t, int64_t), fp32)
  KERNEL(ADD_NS(hinge_embedding_loss), "hinge_embedding_loss", Tensor (const Tensor &, const Tensor &, double, int64_t), fp32)
  KERNEL(ADD_NS(kl_div), "kl_div", Tensor (const Tensor &, const Tensor &, int64_t, bool), fp32)
  KERNEL(ADD_NS(l1_loss), "l1_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(smooth_l1_loss), "smooth_l1_loss", Tensor (const Tensor &, const Tensor &, int64_t, double), fp32)
  KERNEL(ADD_NS(huber_loss), "huber_loss", Tensor (const Tensor &, const Tensor &, int64_t, double), fp32)
  KERNEL(ADD_NS(mse_loss), "mse_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(margin_ranking_loss), "margin_ranking_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t), fp32)
  KERNEL(ADD_NS(multilabel_margin_loss), "multilabel_margin_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(soft_margin_loss), "soft_margin_loss", Tensor (const Tensor &, const Tensor &, int64_t), fp32)
  KERNEL(ADD_NS(triplet_margin_loss), "triplet_margin_loss", Tensor (const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t), fp32)
  KERNEL(ADD_NS(multi_margin_loss), "multi_margin_loss", Tensor (const Tensor &, const Tensor &, const Scalar&, const Scalar&, const c10::optional<Tensor>&, int64_t), fp32)
  KERNEL(ADD_NS(binary_cross_entropy_with_logits), "binary_cross_entropy_with_logits", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&, int64_t), fp32)
  KERNEL(ADD_NS(dist), "dist", Tensor (const Tensor &, const Tensor &, const Scalar&), fp32)
  KERNEL(ADD_NS(pdist), "pdist", Tensor (const Tensor &, double), fp32)
  KERNEL(ADD_NS(cdist), "cdist", Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>), fp32)
  KERNEL(ADD_NS(renorm), "renorm", Tensor (const Tensor &, const Scalar&, int64_t, const Scalar&), fp32)
  // fp32_set_opt_dtype
  KERNEL(ADD_NS(prod), "prod", Tensor (const Tensor &, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(prod), "prod.dim_int", Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(prod), "prod.dim_Dimname", Tensor (const Tensor &, Dimname, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(softmax), "softmax.int", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(softmax), "softmax.Dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(log_softmax), "log_softmax.int", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(log_softmax), "log_softmax.Dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(cumprod), "cumprod", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(cumprod), "cumprod.dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(cumsum), "cumsum", Tensor (const Tensor &, int64_t, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(cumsum), "cumsum.dimname", Tensor (const Tensor &, Dimname, c10::optional<ScalarType>), fp32_set_opt_dtype)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL(ADD_NS(norm), "norm.ScalarOpt_dtype", Tensor (const Tensor &, c10::optional<Scalar>, ScalarType), fp32_set_opt_dtype)
  // KERNEL(ADD_NS(norm), "norm.ScalarOpt_dim_dtype", Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType), fp32_set_opt_dtype)
  // KERNEL(ADD_NS(norm), "norm.names_ScalarOpt_dim_dtype", Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType), fp32_set_opt_dtype)
  KERNEL(ADD_NS(sum), "sum", Tensor (const Tensor &, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(sum), "sum.dim_IntList", Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  KERNEL(ADD_NS(sum), "sum.dim_DimnameList", Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>), fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.Scalar", Tensor (const Tensor &, const Scalar&), Tensor (const Tensor &, const c10::optional<Scalar>&, ScalarType), fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool, ScalarType), fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.names_ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool, ScalarType), fp32_append_dtype)
  // promote
  KERNEL(ADD_NS(addcdiv), "addcdiv", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&), promote)
  KERNEL(ADD_NS(addcmul), "addcmul", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&), promote)
  KERNEL(ADD_NS(atan2), "atan2", Tensor (const Tensor &, const Tensor &), promote)
  KERNEL(ADD_NS(bilinear), "bilinear", Tensor (const Tensor &, const Tensor &, const Tensor &, const c10::optional<Tensor>&), promote)
  KERNEL(ADD_NS(cross), "cross", Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>), promote)
  KERNEL(ADD_NS(dot), "dot", Tensor (const Tensor &, const Tensor &), promote)
  KERNEL(ADD_NS(grid_sampler), "grid_sampler", Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool), promote)
  KERNEL(ADD_NS(index_put), "index_put", Tensor (const Tensor &, const torch::List<c10::optional<Tensor>>&, const Tensor &, bool), promote)
  KERNEL(ADD_NS(tensordot), "tensordot", Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef), promote)
  KERNEL(ADD_NS(scatter_add), "scatter_add", Tensor (const Tensor&, int64_t, const Tensor&, const Tensor&), promote)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // lower_precision_fp cast policy
  KERNEL_CPU(ADD_NS(conv1d), "conv1d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_CPU(ADD_NS(conv2d), "conv2d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_CPU(ADD_NS(conv3d), "conv3d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), lower_precision_fp)
  KERNEL_CPU(ADD_NS(_log_softmax), "_log_softmax", Tensor (const Tensor &, int64_t, bool), lower_precision_fp)
  KERNEL_CPU(ADD_NS(bmm), "bmm", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL_CPU(ADD_NS(mm), "mm", Tensor (const Tensor &, const Tensor &), lower_precision_fp)
  KERNEL_CPU(ADD_NS(baddbmm), "baddbmm", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL_CPU(ADD_NS(addmm), "addmm", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL_CPU(ADD_NS(addbmm), "addbmm", Tensor (const Tensor &, const Tensor &, const Tensor &, const Scalar&, const Scalar&), lower_precision_fp)
  KERNEL_CPU(ADD_NS(linear), "linear", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor> &), lower_precision_fp)

  // fp32 cast policy
  KERNEL_CPU(ADD_NS(conv_transpose3d), "conv_transpose3d.input", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor> &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef), fp32)
  KERNEL_CPU(ADD_NS(batch_norm), "batch_norm", Tensor (const Tensor &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, const c10::optional<Tensor> &, bool, double, double, bool), fp32)
  KERNEL_CPU(ADD_NS(max_pool2d), "max_pool2d", Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool), fp32)
  KERNEL_CPU(ADD_NS(adaptive_avg_pool2d), "adaptive_avg_pool2d", Tensor (const Tensor &, IntArrayRef), fp32)

  KERNEL_CPU(ADD_NS(convolution), "convolution", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t), fp32)
  KERNEL_CPU(ADD_NS(dropout), "dropout", Tensor (const Tensor &, double, bool), fp32)
  KERNEL_CPU(ADD_NS(avg_pool2d), "avg_pool2d", Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>), fp32)
  KERNEL_CPU(ADD_NS(avg_pool3d), "avg_pool3d", Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>), fp32)
  KERNEL_CPU(ADD_NS(gelu), "gelu", Tensor (const Tensor &), fp32)
  KERNEL_CPU(ADD_NS(upsample_nearest1d), "upsample_nearest1d", Tensor (const Tensor &, IntArrayRef, c10::optional<double>), fp32)
  KERNEL_CPU(ADD_NS(upsample_nearest1d), "upsample_nearest1d.vec", Tensor (const Tensor &, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>), fp32)
  KERNEL_CPU(ADD_NS(upsample_nearest2d), "upsample_nearest2d", Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>), fp32)
  KERNEL_CPU(ADD_NS(upsample_nearest2d), "upsample_nearest2d.vec", Tensor (const Tensor &, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>), fp32)
  KERNEL_CPU(ADD_NS(upsample_nearest3d), "upsample_nearest3d", Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>), fp32)
  KERNEL_CPU(ADD_NS(upsample_nearest3d), "upsample_nearest3d.vec", Tensor (const Tensor &, c10::optional<IntArrayRef>, c10::optional<ArrayRef<double>>), fp32)
  KERNEL_CPU(ADD_NS(upsample_linear1d), "upsample_linear1d", Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>), fp32)
  KERNEL_CPU(ADD_NS(upsample_linear1d), "upsample_linear1d.vec", Tensor (const Tensor &, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>), fp32)
  KERNEL_CPU(ADD_NS(upsample_bilinear2d), "upsample_bilinear2d", Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>), fp32)
  KERNEL_CPU(ADD_NS(upsample_bilinear2d), "upsample_bilinear2d.vec", Tensor (const Tensor &, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>), fp32)
  KERNEL_CPU(ADD_NS(upsample_trilinear3d), "upsample_trilinear3d", Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>), fp32)
  KERNEL_CPU(ADD_NS(upsample_trilinear3d), "upsample_trilinear3d.vec", Tensor (const Tensor &, c10::optional<IntArrayRef>, bool, c10::optional<ArrayRef<double>>), fp32)
  KERNEL_CPU(ADD_NS(binary_cross_entropy), "binary_cross_entropy", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, int64_t), fp32)
  KERNEL_CPU(ADD_NS(binary_cross_entropy_with_logits), "binary_cross_entropy_with_logits", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&, int64_t), fp32)
  KERNEL_CPU(ADD_NS(pow), "pow.Tensor_Scalar", Tensor (const Tensor &, const Scalar &), fp32)
  KERNEL_CPU(ADD_NS(pow), "pow.Tensor_Tensor", Tensor (const Tensor &, const Tensor &), fp32)
  KERNEL_CPU(ADD_NS(pow), "pow.Scalar", Tensor (const Scalar&, const Tensor &), fp32)
  KERNEL_CPU(ADD_NS(smooth_l1_loss), "smooth_l1_loss", Tensor (const Tensor &, const Tensor &, int64_t, double), fp32)
  KERNEL_CPU(ADD_NS(reflection_pad1d), "reflection_pad1d", Tensor (const Tensor &, IntArrayRef), fp32)
  KERNEL_CPU(ADD_NS(std), "std", Tensor (const Tensor &, bool), fp32)
  KERNEL_CPU(ADD_NS(std), "std.dim", Tensor (const Tensor &, IntArrayRef, bool, bool), fp32)
  KERNEL_CPU(ADD_NS(instance_norm), "instance_norm", Tensor (const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&, const c10::optional<Tensor>&, const c10::optional<Tensor>&, bool, double, double, bool), fp32)
  KERNEL_CPU(ADD_NS(fake_quantize_per_tensor_affine), "fake_quantize_per_tensor_affine", Tensor (const Tensor &, double, int64_t, int64_t, int64_t), fp32)

  // promote
  KERNEL_CPU(ADD_NS(cat), "cat", Tensor (TensorList, int64_t), promote)
  KERNEL_CPU(ADD_NS(stack), "stack", Tensor (TensorList, int64_t), promote)

  m.impl(TORCH_SELECTIVE_NAME("aten::topk"),
         TORCH_FN((&WrapFunction<CastPolicy::fp32, DeviceType::CPU,
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool),
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool),
                                 &ADD_NS(topk)>::type::call)));

  m.impl(TORCH_SELECTIVE_NAME("aten::sort"),
         TORCH_FN((&WrapFunction<CastPolicy::fp32, DeviceType::CPU,
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool),
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool),
                                 &ADD_NS(sort)>::type::call)));

   m.impl(TORCH_SELECTIVE_NAME("aten::kthvalue"),
         TORCH_FN((&WrapFunction<CastPolicy::fp32, DeviceType::CPU,
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool),
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool),
                                 &ADD_NS(kthvalue)>::type::call)));

   m.impl(TORCH_SELECTIVE_NAME("aten::kthvalue.dimname"),
         TORCH_FN((&WrapFunction<CastPolicy::fp32, DeviceType::CPU,
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, at::Dimname, bool),
                                 std::tuple<Tensor,Tensor> (const Tensor &, int64_t, at::Dimname, bool),
                                 &ADD_NS(kthvalue)>::type::call)));
}
} // namespace
} // namespace autocast
} // namespace at
