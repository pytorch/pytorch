#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <ATen/Operators.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <iostream>
#include <exception>
#include <mutex>

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

bool is_xpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastXPU);
}

void set_xpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastXPU, !new_enabled);
}

bool is_hpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastHPU);
}

void set_hpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastHPU, !new_enabled);
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
std::unordered_map<TensorImpl*, val_type> cached_casts;
std::mutex cached_casts_mutex;

// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the autocasting region.
thread_local int nesting = 0;

// autocast_cpu_dtype is the lower_precision_fp used by AutocastCPU.
thread_local at::ScalarType autocast_cpu_dtype = at::kBFloat16;

// autocast_xpu_dtype is the lower_precision_fp used by AutocastXPU.
thread_local at::ScalarType autocast_xpu_dtype = at::kBFloat16;

// autocast_hpu_dtype is the lower_precision_fp used by AutocastHPU.
thread_local at::ScalarType autocast_hpu_dtype = at::kBFloat16;

// should we enabled the cache inside autocast.
thread_local bool cache_enabled = true;

// autocast_gpu_dtype is the lower_precision_fp used by AutocastGPU.
thread_local at::ScalarType autocast_gpu_dtype = at::kHalf;
}

void clear_cache() {
  const std::lock_guard<std::mutex> lock(cached_casts_mutex);
  cached_casts.clear();
}

int increment_nesting() {
  return ++nesting;
}

int decrement_nesting() {
  return --nesting;
}

at::ScalarType get_autocast_gpu_dtype() {
  return autocast_gpu_dtype;
}

at::ScalarType get_autocast_cpu_dtype() {
  return autocast_cpu_dtype;
}

at::ScalarType get_autocast_xpu_dtype() {
  return autocast_xpu_dtype;
}

at::ScalarType get_autocast_hpu_dtype() {
  return autocast_hpu_dtype;
}

void set_autocast_cpu_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::kBFloat16,
      "Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype");
  autocast_cpu_dtype = dtype;
}

void set_autocast_gpu_dtype(at::ScalarType dtype) {
  autocast_gpu_dtype = dtype;
}

void set_autocast_xpu_dtype(at::ScalarType dtype) {
  autocast_xpu_dtype = dtype;
}

void set_autocast_hpu_dtype(at::ScalarType dtype) {
  autocast_hpu_dtype = dtype;
}

bool is_autocast_cache_enabled() {
  return cache_enabled;
}

void set_autocast_cache_enabled(bool enabled) {
  cache_enabled = enabled;
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
                         arg.is_leaf() && !arg.is_view() && cache_enabled);
    if (can_try_cache) {
      const std::lock_guard<std::mutex> lock(cached_casts_mutex);
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
#define KERNEL(OP, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CUDA, decltype(ATEN_FN(OP)), decltype(ATEN_FN(OP)), &ATEN_FN(OP)>::type::call);
#define KERNEL2(OP, OVERLOAD, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CUDA, decltype(ATEN_FN2(OP, OVERLOAD)), decltype(ATEN_FN2(OP, OVERLOAD)), &ATEN_FN2(OP, OVERLOAD)>::type::call);

// Less-common but still useful case: redispatching to a function with a new signature (e.g. appending a dtype)
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(REDISPATCH_FUNC, REGISTER_NAME, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CUDA, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, &REDISPATCH_FUNC>::type::call);

// KERNEL_CPU registration for AutocastCPU
#define KERNEL_CPU(OP, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CPU, decltype(ATEN_FN(OP)), decltype(ATEN_FN(OP)), &ATEN_FN(OP)>::type::call);
#define KERNEL_CPU2(OP, OVERLOAD, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
    &WrapFunction<CastPolicy::POLICY, DeviceType::CPU, decltype(ATEN_FN2(OP, OVERLOAD)), decltype(ATEN_FN2(OP, OVERLOAD)), &ATEN_FN2(OP, OVERLOAD)>::type::call);

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, Autocast, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  // lower_precision_fp
  KERNEL2(_convolution, deprecated, lower_precision_fp)
  KERNEL(_convolution, lower_precision_fp)
  KERNEL(conv1d, lower_precision_fp)
  KERNEL(conv2d, lower_precision_fp)
  KERNEL(conv3d, lower_precision_fp)
  KERNEL(conv_tbc, lower_precision_fp)
  KERNEL(conv_transpose1d, lower_precision_fp)
  KERNEL2(conv_transpose2d, input, lower_precision_fp)
  KERNEL2(conv_transpose3d, input, lower_precision_fp)
  KERNEL(convolution, lower_precision_fp)
  KERNEL(cudnn_convolution, lower_precision_fp)
  KERNEL(cudnn_convolution_transpose, lower_precision_fp)
  KERNEL(prelu, lower_precision_fp)
  KERNEL(addmm, lower_precision_fp)
  KERNEL(addmv, lower_precision_fp)
  KERNEL(addr, lower_precision_fp)
  KERNEL(matmul, lower_precision_fp)
  KERNEL(einsum, lower_precision_fp)
  KERNEL(mm, lower_precision_fp)
  KERNEL(mv, lower_precision_fp)
  KERNEL(linear, lower_precision_fp)
  KERNEL(addbmm, lower_precision_fp)
  KERNEL(baddbmm, lower_precision_fp)
  KERNEL(bmm, lower_precision_fp)
  KERNEL(chain_matmul, lower_precision_fp)
  KERNEL(linalg_multi_dot, lower_precision_fp)
  KERNEL(_thnn_fused_lstm_cell, lower_precision_fp)
  KERNEL(_thnn_fused_gru_cell, lower_precision_fp)
  KERNEL(lstm_cell, lower_precision_fp)
  KERNEL(gru_cell, lower_precision_fp)
  KERNEL(rnn_tanh_cell, lower_precision_fp)
  KERNEL(rnn_relu_cell, lower_precision_fp)

  // fp32
  KERNEL(acos, fp32)
  KERNEL(asin, fp32)
  KERNEL(cosh, fp32)
  KERNEL(erfinv, fp32)
  KERNEL(exp, fp32)
  KERNEL(expm1, fp32)
  KERNEL(log, fp32)
  KERNEL(log10, fp32)
  KERNEL(log2, fp32)
  KERNEL(log1p, fp32)
  KERNEL(reciprocal, fp32)
  KERNEL(rsqrt, fp32)
  KERNEL(sinh, fp32)
  KERNEL(tan, fp32)
  KERNEL2(pow, Tensor_Scalar, fp32)
  KERNEL2(pow, Tensor_Tensor, fp32)
  KERNEL2(pow, Scalar, fp32)
  KERNEL(softplus, fp32)
  KERNEL(layer_norm, fp32)
  KERNEL(native_layer_norm, fp32)
  KERNEL(group_norm, fp32)
  KERNEL(frobenius_norm, fp32)
  KERNEL2(frobenius_norm, dim, fp32)
  KERNEL(nuclear_norm, fp32)
  KERNEL2(nuclear_norm, dim, fp32)
  KERNEL(cosine_similarity, fp32)
  KERNEL(poisson_nll_loss, fp32)
  KERNEL(cosine_embedding_loss, fp32)
  KERNEL(nll_loss, fp32)
  KERNEL(nll_loss2d, fp32)
  KERNEL(hinge_embedding_loss, fp32)
  KERNEL(kl_div, fp32)
  KERNEL(l1_loss, fp32)
  KERNEL(smooth_l1_loss, fp32)
  KERNEL(huber_loss, fp32)
  KERNEL(mse_loss, fp32)
  KERNEL(margin_ranking_loss, fp32)
  KERNEL(multilabel_margin_loss, fp32)
  KERNEL(soft_margin_loss, fp32)
  KERNEL(triplet_margin_loss, fp32)
  KERNEL(multi_margin_loss, fp32)
  KERNEL(binary_cross_entropy_with_logits, fp32)
  KERNEL(dist, fp32)
  KERNEL(pdist, fp32)
  KERNEL(cdist, fp32)
  KERNEL(renorm, fp32)
  KERNEL(logsumexp, fp32)
  // fp32_set_opt_dtype
  KERNEL(prod, fp32_set_opt_dtype)
  KERNEL2(prod, dim_int, fp32_set_opt_dtype)
  KERNEL2(prod, dim_Dimname, fp32_set_opt_dtype)
  KERNEL2(softmax, int, fp32_set_opt_dtype)
  KERNEL2(softmax, Dimname, fp32_set_opt_dtype)
  KERNEL2(log_softmax, int, fp32_set_opt_dtype)
  KERNEL2(log_softmax, Dimname, fp32_set_opt_dtype)
  KERNEL(cumprod, fp32_set_opt_dtype)
  KERNEL2(cumprod, dimname, fp32_set_opt_dtype)
  KERNEL(cumsum, fp32_set_opt_dtype)
  KERNEL2(cumsum, dimname, fp32_set_opt_dtype)
  KERNEL(linalg_vector_norm, fp32_set_opt_dtype)
  KERNEL(linalg_matrix_norm, fp32_set_opt_dtype)
  KERNEL2(linalg_matrix_norm, str_ord, fp32_set_opt_dtype)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL2(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL2(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  // KERNEL2(norm, names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  KERNEL(sum, fp32_set_opt_dtype)
  KERNEL2(sum, dim_IntList, fp32_set_opt_dtype)
  KERNEL2(sum, dim_DimnameList, fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.Scalar", Tensor (const Tensor &, const Scalar&), Tensor (const Tensor &, const c10::optional<Scalar>&, ScalarType), fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool, ScalarType), fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.names_ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool, ScalarType), fp32_append_dtype)
  // promote
  KERNEL(addcdiv, promote)
  KERNEL(addcmul, promote)
  KERNEL(atan2, promote)
  KERNEL(bilinear, promote)
  KERNEL(cross, promote)
  KERNEL(dot, promote)
  KERNEL(grid_sampler, promote)
  KERNEL(index_put, promote)
  KERNEL(tensordot, promote)
  KERNEL(scatter_add, promote)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // lower_precision_fp cast policy
  KERNEL_CPU(conv1d, lower_precision_fp)
  KERNEL_CPU2(conv1d, padding, lower_precision_fp)
  KERNEL_CPU(conv2d, lower_precision_fp)
  KERNEL_CPU2(conv2d, padding, lower_precision_fp)
  KERNEL_CPU(conv3d, lower_precision_fp)
  KERNEL_CPU2(conv3d, padding, lower_precision_fp)
  KERNEL_CPU(bmm, lower_precision_fp)
  KERNEL_CPU(mm, lower_precision_fp)
  KERNEL_CPU(baddbmm, lower_precision_fp)
  KERNEL_CPU(addmm, lower_precision_fp)
  KERNEL_CPU(addbmm, lower_precision_fp)
  KERNEL_CPU(linear, lower_precision_fp)
  KERNEL_CPU2(_convolution, deprecated, lower_precision_fp)
  KERNEL_CPU(matmul, lower_precision_fp)
  KERNEL_CPU(conv_tbc, lower_precision_fp)

  // fp32 cast policy
  KERNEL_CPU(conv_transpose1d, fp32)
  KERNEL_CPU2(conv_transpose2d, input, fp32)
  KERNEL_CPU2(conv_transpose3d, input, fp32)
  KERNEL_CPU(avg_pool3d, fp32)
  KERNEL_CPU(binary_cross_entropy, fp32)
  KERNEL_CPU(grid_sampler, fp32)
  KERNEL_CPU(polar, fp32)
  KERNEL_CPU(prod, fp32)
  KERNEL_CPU2(prod, dim_int, fp32)
  KERNEL_CPU2(prod, dim_Dimname, fp32)
  KERNEL_CPU(quantile, fp32)
  KERNEL_CPU2(quantile, scalar, fp32)
  KERNEL_CPU(nanquantile, fp32)
  KERNEL_CPU2(nanquantile, scalar, fp32)
  KERNEL_CPU(stft, fp32)
  KERNEL_CPU2(stft, center, fp32)
  KERNEL_CPU(cdist, fp32)
  KERNEL_CPU(grid_sampler_2d, fp32)
  KERNEL_CPU(_grid_sampler_2d_cpu_fallback, fp32)
  KERNEL_CPU(grid_sampler_3d, fp32)
  KERNEL_CPU(trace, fp32)
  KERNEL_CPU(view_as_complex, fp32)
  KERNEL_CPU(cholesky, fp32)
  KERNEL_CPU(cholesky_inverse, fp32)
  KERNEL_CPU(cholesky_solve, fp32)
  KERNEL_CPU(inverse, fp32)
  KERNEL_CPU(lu_solve, fp32)
  KERNEL_CPU(orgqr, fp32)
  KERNEL_CPU(ormqr, fp32)
  KERNEL_CPU(pinverse, fp32)
  KERNEL_CPU(max_pool3d, fp32)
  KERNEL_CPU(max_unpool2d, fp32)
  KERNEL_CPU(max_unpool3d, fp32)
  KERNEL_CPU(adaptive_avg_pool3d, fp32)
  KERNEL_CPU(reflection_pad1d, fp32)
  KERNEL_CPU(reflection_pad2d, fp32)
  KERNEL_CPU(replication_pad1d, fp32)
  KERNEL_CPU(replication_pad2d, fp32)
  KERNEL_CPU(replication_pad3d, fp32)
  KERNEL_CPU(mse_loss, fp32)
  KERNEL_CPU(cosine_embedding_loss, fp32)
  KERNEL_CPU(nll_loss, fp32)
  KERNEL_CPU(nll_loss2d, fp32)
  KERNEL_CPU(hinge_embedding_loss, fp32)
  KERNEL_CPU(poisson_nll_loss, fp32)
  KERNEL_CPU(smooth_l1_loss, fp32)
  KERNEL_CPU(cross_entropy_loss, fp32)
  KERNEL_CPU(l1_loss, fp32)
  KERNEL_CPU(huber_loss, fp32)
  KERNEL_CPU(margin_ranking_loss, fp32)
  KERNEL_CPU(soft_margin_loss, fp32)
  KERNEL_CPU(triplet_margin_loss, fp32)
  KERNEL_CPU(multi_margin_loss, fp32)
  KERNEL_CPU2(ctc_loss, IntList, fp32)
  KERNEL_CPU2(ctc_loss, Tensor, fp32)
  KERNEL_CPU(kl_div, fp32)
  KERNEL_CPU(multilabel_margin_loss, fp32)
  KERNEL_CPU(binary_cross_entropy_with_logits, fp32)
  KERNEL_CPU(fft_fft, fp32)
  KERNEL_CPU(fft_ifft, fp32)
  KERNEL_CPU(fft_fft2, fp32)
  KERNEL_CPU(fft_ifft2, fp32)
  KERNEL_CPU(fft_fftn, fp32)
  KERNEL_CPU(fft_ifftn, fp32)
  KERNEL_CPU(fft_rfft, fp32)
  KERNEL_CPU(fft_irfft, fp32)
  KERNEL_CPU(fft_rfft2, fp32)
  KERNEL_CPU(fft_irfft2, fp32)
  KERNEL_CPU(fft_rfftn, fp32)
  KERNEL_CPU(fft_irfftn, fp32)
  KERNEL_CPU(fft_hfft, fp32)
  KERNEL_CPU(fft_ihfft, fp32)
  KERNEL_CPU(linalg_cond, fp32)
  KERNEL_CPU2(linalg_cond, p_str, fp32)
  KERNEL_CPU(linalg_matrix_rank, fp32)
  KERNEL_CPU2(linalg_matrix_rank, tol_tensor, fp32)
  KERNEL_CPU2(linalg_matrix_rank, atol_rtol_tensor, fp32)
  KERNEL_CPU2(linalg_matrix_rank, atol_rtol_float, fp32)
  KERNEL_CPU(linalg_solve, fp32)
  KERNEL_CPU(linalg_cholesky, fp32)
  KERNEL_CPU(linalg_svdvals, fp32)
  KERNEL_CPU(linalg_eigvals, fp32)
  KERNEL_CPU(linalg_eigvalsh, fp32)
  KERNEL_CPU(linalg_inv, fp32)
  KERNEL_CPU(linalg_householder_product, fp32)
  KERNEL_CPU(linalg_tensorinv, fp32)
  KERNEL_CPU(linalg_tensorsolve, fp32)
  KERNEL_CPU(fake_quantize_per_tensor_affine, fp32)
  KERNEL_CPU(geqrf, fp32)
  KERNEL_CPU(_lu_with_info, fp32)
  KERNEL_CPU(qr, fp32)
  KERNEL_CPU(svd, fp32)
  KERNEL_CPU(symeig, fp32)
  KERNEL_CPU(triangular_solve, fp32)
  KERNEL_CPU(fractional_max_pool2d, fp32)
  KERNEL_CPU(fractional_max_pool3d, fp32)
  KERNEL_CPU(adaptive_max_pool3d, fp32)
  KERNEL_CPU(multilabel_margin_loss_forward, fp32)
  KERNEL_CPU(linalg_qr, fp32)
  KERNEL_CPU(linalg_cholesky_ex, fp32)
  KERNEL_CPU(linalg_svd, fp32)
  KERNEL_CPU(linalg_eig, fp32)
  KERNEL_CPU(linalg_eigh, fp32)
  KERNEL_CPU(linalg_lstsq, fp32)
  KERNEL_CPU(linalg_inv_ex, fp32)

  // promote
  KERNEL_CPU(stack, promote)
  KERNEL_CPU(cat, promote)
  KERNEL_CPU(index_copy, promote)
  KERNEL_CPU2(index_copy, dimname, promote)

}
} // namespace
} // namespace autocast
} // namespace at
