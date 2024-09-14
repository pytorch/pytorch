#pragma once

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <torch/library.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

namespace at::autocast {

TORCH_API bool is_autocast_enabled(at::DeviceType device_type);
TORCH_API void set_autocast_enabled(at::DeviceType device_type, bool enabled);
TORCH_API at::ScalarType get_autocast_dtype(at::DeviceType device_type);
TORCH_API void set_autocast_dtype(
    at::DeviceType device_type,
    at::ScalarType dtype);
TORCH_API void clear_cache();
TORCH_API int increment_nesting();
TORCH_API int decrement_nesting();
TORCH_API bool is_autocast_cache_enabled();
TORCH_API void set_autocast_cache_enabled(bool enabled);

// deprecated CUDA-specific autocast APIs
C10_DEPRECATED_MESSAGE(
    "at::autocast::is_enabled() is deprecated. Please use at::autocast::is_autocast_enabled(at::kCUDA) instead.")
TORCH_API inline bool is_enabled() {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "() is deprecated. Please use at::autocast::is_autocast_enabled(at::kCUDA) instead.")
  return is_autocast_enabled(at::kCUDA);
}
C10_DEPRECATED_MESSAGE(
    "at::autocast::set_enabled(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(at::kCUDA, enabled) instead.")
TORCH_API inline void set_enabled(bool enabled) {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(at::kCUDA, enabled) instead.")
  set_autocast_enabled(at::kCUDA, enabled);
}
C10_DEPRECATED_MESSAGE(
    "at::autocast::get_autocast_gpu_dtype() is deprecated. Please use at::autocast::get_autocast_dtype(at::kCUDA) instead.")
TORCH_API inline at::ScalarType get_autocast_gpu_dtype() {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "() is deprecated. Please use at::autocast::get_autocast_dtype(at::kCUDA) instead.")
  return get_autocast_dtype(at::kCUDA);
}
C10_DEPRECATED_MESSAGE(
    "at::autocast::set_autocast_gpu_dtype(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(at::kCUDA, dtype) instead.")
TORCH_API inline void set_autocast_gpu_dtype(at::ScalarType dtype) {
  TORCH_WARN_DEPRECATION(
      "at::autocast::",
      __func__,
      "(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(at::kCUDA, dtype) instead.")
  set_autocast_dtype(at::kCUDA, dtype);
}

#define DECLARE_DEPRECATED_AUTOCAST_APIS(name, device_type)                                          \
  C10_DEPRECATED_MESSAGE(                                                                            \
      "at::autocast::is_" #name                                                                      \
      "_enabled() is deprecated. Please use at::autocast::is_autocast_enabled(" #device_type         \
      ") instead.")                                                                                  \
  TORCH_API inline bool is_##name##_enabled() {                                                      \
    TORCH_WARN_DEPRECATION(                                                                          \
        "at::autocast::",                                                                            \
        __func__,                                                                                    \
        "() is deprecated. Please use at::autocast::is_autocast_enabled(" #device_type               \
        ") instead.")                                                                                \
    return is_autocast_enabled(device_type);                                                         \
  }                                                                                                  \
                                                                                                     \
  C10_DEPRECATED_MESSAGE(                                                                            \
      "at::autocast::set_" #name                                                                     \
      "_enabled(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(" #device_type \
      ", enabled) instead.")                                                                         \
  TORCH_API inline void set_##name##_enabled(bool enabled) {                                         \
    TORCH_WARN_DEPRECATION(                                                                          \
        "at::autocast::",                                                                            \
        __func__,                                                                                    \
        "(enabled) is deprecated. Please use at::autocast::set_autocast_enabled(" #device_type       \
        ", enabled) instead.")                                                                       \
    set_autocast_enabled(device_type, enabled);                                                      \
  }                                                                                                  \
                                                                                                     \
  C10_DEPRECATED_MESSAGE(                                                                            \
      "at::autocast::get_autocast_" #name                                                            \
      "_dtype() is deprecated. Please use at::autocast::get_autocast_dtype(" #device_type            \
      ") instead.")                                                                                  \
  TORCH_API inline at::ScalarType get_autocast_##name##_dtype() {                                    \
    TORCH_WARN_DEPRECATION(                                                                          \
        "at::autocast::",                                                                            \
        __func__,                                                                                    \
        "() is deprecated. Please at::autocast::get_autocast_dtype(" #device_type                    \
        ") instead.")                                                                                \
    return get_autocast_dtype(device_type);                                                          \
  }                                                                                                  \
                                                                                                     \
  C10_DEPRECATED_MESSAGE(                                                                            \
      "at::autocast::set_autocast_" #name                                                            \
      "_dtype(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(" #device_type       \
      ", dtype) instead.")                                                                           \
  TORCH_API inline void set_autocast_##name##_dtype(at::ScalarType dtype) {                          \
    TORCH_WARN_DEPRECATION(                                                                          \
        "at::autocast::",                                                                            \
        __func__,                                                                                    \
        "(dtype) is deprecated. Please use at::autocast::set_autocast_dtype(" #device_type           \
        ", dtype) instead.")                                                                         \
    set_autocast_dtype(device_type, dtype);                                                          \
  }

#define AT_FORALL_DEPRECATED_AUTOCAST_BAKCNEDS(_) \
  _(cpu, at::kCPU)                                \
  _(xpu, at::kXPU)                                \
  _(xla, at::kXLA)                                \
  _(hpu, at::kHPU)                                \
  _(ipu, at::kIPU)                                \
  _(privateuseone, at::kPrivateUse1)

// deprecated other backend specific autocast APIs
AT_FORALL_DEPRECATED_AUTOCAST_BAKCNEDS(DECLARE_DEPRECATED_AUTOCAST_APIS)

namespace {
inline bool is_autocast_eligible(
    const Tensor& tensor,
    c10::DeviceType device_type) {
  switch (device_type) {
    case c10::DeviceType::CUDA:
      return (tensor.is_cuda() || tensor.is_xla()) &&
          tensor.is_floating_point();
    case c10::DeviceType::CPU:
      return (tensor.is_cpu() || tensor.is_mkldnn()) &&
          tensor.is_floating_point();
    case c10::DeviceType::XPU:
      return tensor.is_xpu() && tensor.is_floating_point();
    case c10::DeviceType::IPU:
      return tensor.is_ipu() && tensor.is_floating_point();
    case c10::DeviceType::HPU:
      return tensor.is_hpu() && tensor.is_floating_point();
    case c10::DeviceType::XLA:
      return tensor.is_xla() && tensor.is_floating_point();
    case c10::DeviceType::PrivateUse1:
      return tensor.is_privateuseone() && tensor.is_floating_point();
    case c10::DeviceType::MPS:
      return tensor.is_mps() && tensor.is_floating_point();
    default:
      return false;
  }
}
} // namespace

inline DispatchKey get_autocast_dispatch_key_from_device_type(
    c10::DeviceType device_type) {
  switch (device_type) {
    case c10::DeviceType::CUDA:
      return DispatchKey::Autocast;
    case c10::DeviceType::CPU:
      return DispatchKey::AutocastCPU;
    case c10::DeviceType::XPU:
      return DispatchKey::AutocastXPU;
    case c10::DeviceType::IPU:
      return DispatchKey::AutocastIPU;
    case c10::DeviceType::HPU:
      return DispatchKey::AutocastHPU;
    case c10::DeviceType::XLA:
      return DispatchKey::AutocastXLA;
    case c10::DeviceType::PrivateUse1:
      return DispatchKey::AutocastPrivateUse1;
    case c10::DeviceType::MPS:
      return DispatchKey::AutocastMPS;
    default:
      throw std::runtime_error(
          "unknown device type for autocast in get_autocast_dispatch_key_from_device_type");
  }
}

inline bool is_autocast_available(c10::DeviceType device_type) {
  if (device_type == at::kCPU || device_type == at::kCUDA ||
      device_type == at::kXPU || device_type == at::kIPU ||
      device_type == at::kHPU || device_type == at::kXLA ||
      device_type == at::kPrivateUse1 || device_type == at::kMPS) {
    return true;
  } else {
    return false;
  }
}

inline at::ScalarType get_lower_precision_fp_from_device_type(
    c10::DeviceType device_type) {
  if (is_autocast_available(device_type)) {
    return get_autocast_dtype(device_type);
  } else {
    throw std::runtime_error(
        "unknown device type for autocast in get_lower_precision_fp_from_device_type");
  }
}

/********************************************************************
Logic to extract the promote type from any Tensor or TensorList args.
********************************************************************/

// Overload to catch Tensor args.
// If nextArg is floating-point, compare its scalar_type with our
// current best guess for the promote type, and update if necessary.
inline at::ScalarType prioritize(
    at::ScalarType current,
    const Tensor& nextArg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  if (current == at::kDouble) {
    AT_ERROR("promote type is double in at::autocast::prioritize");
    return current;
  }
  at::ScalarType lower_precision_fp =
      get_lower_precision_fp_from_device_type(device_type);
  if (is_autocast_eligible(nextArg, device_type)) {
    auto next = nextArg.scalar_type();
    if (next == at::kDouble) {
      return current; // ignores double tensors
    } else if (current == at::kFloat || next == at::kFloat) {
      return at::kFloat; // prioritizes float over lower_precision_fp
    } else if (current == lower_precision_fp && next == lower_precision_fp) {
      return lower_precision_fp;
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
inline at::ScalarType prioritize(
    at::ScalarType current,
    const TensorList& list,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor, device_type);
  }
  return current;
}

inline at::ScalarType prioritize(
    at::ScalarType current,
    const ITensorListRef& list,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor, device_type);
  }
  return current;
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template <typename T>
inline at::ScalarType prioritize(
    at::ScalarType current,
    T nextArg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  return current;
}

// Overload for the tail case.
inline at::ScalarType promote_type(
    at::ScalarType current,
    c10::DeviceType device_type) {
  return current;
}

// Unpack args and determine if incoming lower_precision_fp tensors need to be
// promoted to float32. Non-Tensor arguments are ignored.
template <typename Arg0, typename... Args>
inline at::ScalarType promote_type(
    at::ScalarType current,
    c10::DeviceType device_type,
    Arg0 arg0,
    Args... args) {
  auto new_current = prioritize(current, arg0, device_type);
  return promote_type(new_current, device_type, args...);
}

/****************************************************
Logic to apply cached casting to any Tensor argument.
****************************************************/
inline bool is_eligible(
    const Tensor& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  return (
      arg.defined() && is_autocast_eligible(arg, device_type) &&
      (arg.scalar_type() != at::kDouble));
}

// Overload to catch Tensor args
TORCH_API Tensor cached_cast(
    at::ScalarType to_type,
    const Tensor& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA);

// Overload to process std::optional<Tensor>
inline std::optional<Tensor> cached_cast(
    at::ScalarType to_type,
    const std::optional<Tensor>& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  if (arg.has_value()) {
    return cached_cast(to_type, *arg, device_type);
  } else {
    return std::nullopt;
  }
}

// Overload to process TensorLists
inline std::vector<Tensor> cached_cast(
    at::ScalarType to_type,
    const TensorList& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.emplace_back(cached_cast(to_type, t, device_type));
  }
  return vec;
}

inline std::vector<Tensor> cached_cast(
    at::ScalarType to_type,
    const ITensorListRef& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.emplace_back(cached_cast(to_type, t, device_type));
  }
  return vec;
}

// Template to catch non-Tensor args.
template <typename T>
inline T cached_cast(
    at::ScalarType to_type,
    T arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
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
std::optional<ScalarType> inline set_opt_dtype(
    at::ScalarType to_type,
    const std::optional<ScalarType>& dtype) {
  return dtype.has_value() ? dtype : to_type;
}

// Template to catch other args
template <typename T>
inline T set_opt_dtype(at::ScalarType to_type, T arg) {
  return arg;
}

template <typename... Args>
inline bool firstarg_is_eligible(
    c10::DeviceType device_type,
    const Tensor& arg,
    Args... args) {
  return is_eligible(arg, device_type);
}

template <typename... Args>
inline at::ScalarType type_from_firstarg(
    c10::DeviceType device_type,
    at::ScalarType to_type,
    const Tensor& arg,
    Args... args) {
  return (is_eligible(arg, device_type) ? to_type : arg.scalar_type());
}

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  lower_precision_fp = 0, // Cast all inputs to lower_precision_fp before
                          // running the op. Currently, lower_precision_fp is
                          // fp16 for AutocastCUDA, and is defined by user
                          // (default bf16) for AutocastCPU or other device.
  fp32, // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //  1. we'd like to run in fp32 and
                      //  2. have a std::optional<ScalarType> arg that controls
                      //  the output type.
                      // fp32_set_opt_dtype wrappers' policy is: if the output
                      // type is already set, don't touch it, otherwise, set
                      // it to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //  1. we'd like to run in fp32 and
                     //  2. have some overloads that accept an output type and
                     //  other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't
                     // have an output dtype.
                     // The wrapper policy is:  append at::kFloat to the args,
                     // and redispatch to the type-aware overload.
  promote, // Run in the widest dtype among several args.
};

/********************************************************************************************************
Templates to provide wrapper functions

I'm copying the pattern used in core/boxing/impl/WrapFunctionIntoFunctor.h to
extract args and return type. (see also
https://stackoverflow.com/questions/46533698/how-to-deduce-argument-list-from-function-pointer)

This strategy uses an exterior "WrapFunction" that extracts arguments on behalf
of (in my case several specializations of) an interior "WrapFunction_".
Interior WrapFunction_ specializations are defined for each CastPolicy.
********************************************************************************************************/

// Base template for WrapFunction_, which is specialized to contain a "call"
// method each CastPolicy
template <
    CastPolicy policy,
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template <
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::lower_precision_fp,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(
        get_lower_precision_fp_from_device_type(device_type),
        args,
        device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template <
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::fp32,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::fp32_set_opt_dtype General_DeviceType
template <
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::fp32_set_opt_dtype,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    if (firstarg_is_eligible(device_type, args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype,
      // because setting opt dtype explicitly may interfere with internal
      // implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype General_DeviceType
template <
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::fp32_append_dtype,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    at::ScalarType out_type =
        type_from_firstarg(device_type, at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template <
    c10::DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::promote,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type = promote_type(
        get_lower_precision_fp_from_device_type(device_type),
        device_type,
        args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating
// core/boxing/impl/WrapFunctionIntoFunctor.h)
template <
    CastPolicy policy,
    c10::DeviceType device_type,
    class Registered, // The signature for which we're registering.  The
                      // dispatcher's calling code invokes our registered
                      // functions with arguments matching Registered, so we
                      // register WrapFunction_::call methods with a matching
                      // signature to properly field those arguments.
    // guts::function_traits below extracts return_type and
    // parameter_types from Registered, which WrapFunction_
    // templates above use to declare their call methods.
    class Redispatch, // The signature for the function we're redispatching to.
                      // In most cases this is the same as Registered, but for
                      // some ops (for example, ops where we append a dtype)
                      // it's useful to redispatch to a function with a
                      // different signature.
    Redispatch* F> // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<
      policy,
      device_type,
      Redispatch,
      F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

/*****************************************************************************************************************
This section performs load-time registration for autocast wrappers.

It's debatable at what level operations should be patched.  We'd like casts to
be autograd-exposed and precede autograd history recording, so that for
lower_precision_fp ops, input tensors are saved for backward in
lower_precision_fp rather than fp32.  Saving inputs in lower_precision_fp
can significantly reduce a model's memory footprint.

Option 1 (strawman):  Patch only at the level of explicit calls into
cudnn/cublas (cudnn_convolution, etc), because those are the code paths that are
guaranteed to use Tensor Cores, therefore they're the ones that will benefit
most from lower_precision_fp.   Potential pitfall:  convolutions (and other ops)
are wrapped in several layers of at::* calls.  If one of those happens to record
autograd history, then we've lost the opportunity to save inputs in
lower_precision_fp.

Option 2:  Patch the Python-exposed surface of calls, to make 100% sure autograd
history recording can't sneak in ahead of autocast.  This mirrors Apex most
closely.

I think Option 2 is the right answer for all ops, not just convolutions. Option
2 is what I implement here.
*****************************************************************************************************************/

/********************************************************************************************************************
Explicit registration for out-of-place ops

The stuff below could be codegenned.  Ed said
> you are going to have to write the function definition at some point, I
wouldn't try to get clever about it Therefore, for the moment, this is all
copy pasted in from VariableTypeEverything.cpp with appropriate substitutions.
********************************************************************************************************************/

} // namespace at::autocast

#define ADD_NS(RAW_OP) at::RAW_OP

#define _KERNEL_OVERLOAD_NARG_IMPL(_0, _1, _2, N, ...) N
#define _KERNEL_OVERLOAD_NARG(...) \
  C10_EXPAND_MSVC_WORKAROUND(_KERNEL_OVERLOAD_NARG_IMPL(__VA_ARGS__, 2, 1))

// Common cases where registration signature matches redispatch signature
// (that's why SIGNATURE is repeated in the WrapFunction instantiation)
#define KERNEL1(DISPATCHKEY, OP, POLICY)      \
  m.impl(                                     \
      TORCH_SELECTIVE_NAME("aten::" #OP),     \
      &::at::autocast::WrapFunction<          \
          ::at::autocast::CastPolicy::POLICY, \
          DISPATCHKEY,                        \
          decltype(ATEN_FN(OP)),              \
          decltype(ATEN_FN(OP)),              \
          &ATEN_FN(OP)>::type::call);

#define KERNEL2(DISPATCHKEY, OP, OVERLOAD, POLICY)      \
  m.impl(                                               \
      TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
      &::at::autocast::WrapFunction<                    \
          ::at::autocast::CastPolicy::POLICY,           \
          DISPATCHKEY,                                  \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          &ATEN_FN2(OP, OVERLOAD)>::type::call);

#define _KERNEL_DISPATCH(DISPATCHKEY, NARG, ...) \
  C10_CONCATENATE(KERNEL, NARG)(DISPATCHKEY, __VA_ARGS__)

#define _KERNEL_IMPL(DISPATCHKEY, ...) \
  _KERNEL_DISPATCH(DISPATCHKEY, _KERNEL_OVERLOAD_NARG(__VA_ARGS__), __VA_ARGS__)

// It will dispatch to KERNEL1 or KERNEL2 based on its inputs.
#define KERNEL(DISPATCHKEY, ...) _KERNEL_IMPL(DISPATCHKEY, __VA_ARGS__)

// Less-common but still useful case: redispatching to a function
// with a new signature (e.g. appending a dtype)
#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(      \
    DISPATCHKEY,                                    \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  m.impl(                                           \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
      &::at::autocast::WrapFunction<                \
          ::at::autocast::CastPolicy::POLICY,       \
          DISPATCHKEY,                              \
          REGISTER_SIGNATURE,                       \
          REDISPATCH_SIGNATURE,                     \
          &REDISPATCH_FUNC>::type::call);

// KERNEL_CPU/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CPU
// registration (OP, POLICY) or (OP, OVERLOAD, POLICY) for AutocastCPU
#define KERNEL_CPU(...) KERNEL(c10::DeviceType::CPU, __VA_ARGS__)

#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CPU( \
    REDISPATCH_FUNC,                               \
    REGISTER_NAME,                                 \
    REGISTER_SIGNATURE,                            \
    REDISPATCH_SIGNATURE,                          \
    POLICY)                                        \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(           \
      c10::DeviceType::CPU,                        \
      REDISPATCH_FUNC,                             \
      REGISTER_NAME,                               \
      REGISTER_SIGNATURE,                          \
      REDISPATCH_SIGNATURE,                        \
      POLICY)

// KERNEL_CUDA/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA
// registration (OP, POLICY) or (OP, OVERLOAD, POLICY) for AutocastCUDA
#define KERNEL_CUDA(...) KERNEL(c10::DeviceType::CUDA, __VA_ARGS__)

#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA( \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(            \
      c10::DeviceType::CUDA,                        \
      REDISPATCH_FUNC,                              \
      REGISTER_NAME,                                \
      REGISTER_SIGNATURE,                           \
      REDISPATCH_SIGNATURE,                         \
      POLICY)

// KERNEL_XPU/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU
// registration (OP, POLICY) or (OP, OVERLOAD, POLICY) for AutocastXPU
#define KERNEL_XPU(...) KERNEL(c10::DeviceType::XPU, __VA_ARGS__)

#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU( \
    REDISPATCH_FUNC,                               \
    REGISTER_NAME,                                 \
    REGISTER_SIGNATURE,                            \
    REDISPATCH_SIGNATURE,                          \
    POLICY)                                        \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(           \
      c10::DeviceType::XPU,                        \
      REDISPATCH_FUNC,                             \
      REGISTER_NAME,                               \
      REGISTER_SIGNATURE,                          \
      REDISPATCH_SIGNATURE,                        \
      POLICY)

// KERNEL_PRIVATEUSEONE/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE
// registration (OP, POLICY) or (OP, OVERLOAD, POLICY) for AutocastPrivateUse1
#define KERNEL_PRIVATEUSEONE(...) \
  KERNEL(c10::DeviceType::PrivateUse1, __VA_ARGS__)

#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE( \
    REDISPATCH_FUNC,                                         \
    REGISTER_NAME,                                           \
    REGISTER_SIGNATURE,                                      \
    REDISPATCH_SIGNATURE,                                    \
    POLICY)                                                  \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(                     \
      c10::DeviceType::PrivateUse1,                          \
      REDISPATCH_FUNC,                                       \
      REGISTER_NAME,                                         \
      REGISTER_SIGNATURE,                                    \
      REDISPATCH_SIGNATURE,                                  \
      POLICY)

// KERNEL_MPS registration for AutocastMPS
#define KERNEL_MPS(OP, POLICY)            \
  m.impl(                                 \
      TORCH_SELECTIVE_NAME("aten::" #OP), \
      &WrapFunction<                      \
          CastPolicy::POLICY,             \
          DeviceType::MPS,                \
          decltype(ATEN_FN(OP)),          \
          decltype(ATEN_FN(OP)),          \
          &ATEN_FN(OP)>::type::call);

#define KERNEL_MPS2(OP, OVERLOAD, POLICY)               \
  m.impl(                                               \
      TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
      &WrapFunction<                                    \
          CastPolicy::POLICY,                           \
          DeviceType::MPS,                              \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          &ATEN_FN2(OP, OVERLOAD)>::type::call);

// Op lists for different policies.
// To make sure other backends can reuse the policy op list.
#define AT_FORALL_LOWER_PRECISION_FP(_)  \
  _(_convolution, deprecated)            \
  _(_convolution)                        \
  _(conv1d)                              \
  _(conv2d)                              \
  _(conv3d)                              \
  _(conv_tbc)                            \
  _(conv_transpose1d)                    \
  _(conv_transpose2d, input)             \
  _(conv_transpose3d, input)             \
  _(convolution)                         \
  _(prelu)                               \
  _(addmm)                               \
  _(addmv)                               \
  _(addr)                                \
  _(matmul)                              \
  _(einsum)                              \
  _(mm)                                  \
  _(mv)                                  \
  _(linalg_vecdot)                       \
  _(linear)                              \
  _(addbmm)                              \
  _(baddbmm)                             \
  _(bmm)                                 \
  _(chain_matmul)                        \
  _(linalg_multi_dot)                    \
  _(_thnn_fused_lstm_cell)               \
  _(_thnn_fused_gru_cell)                \
  _(lstm_cell)                           \
  _(gru_cell)                            \
  _(rnn_tanh_cell)                       \
  _(rnn_relu_cell)                       \
  _(_scaled_dot_product_flash_attention) \
  _(scaled_dot_product_attention)

#define AT_FORALL_FP32(_)             \
  _(acos)                             \
  _(asin)                             \
  _(cosh)                             \
  _(erfinv)                           \
  _(exp)                              \
  _(expm1)                            \
  _(log)                              \
  _(log10)                            \
  _(log2)                             \
  _(log1p)                            \
  _(reciprocal)                       \
  _(rsqrt)                            \
  _(sinh)                             \
  _(tan)                              \
  _(pow, Tensor_Scalar)               \
  _(pow, Tensor_Tensor)               \
  _(pow, Scalar)                      \
  _(softplus)                         \
  _(layer_norm)                       \
  _(native_layer_norm)                \
  _(group_norm)                       \
  _(frobenius_norm, dim)              \
  _(nuclear_norm)                     \
  _(nuclear_norm, dim)                \
  _(cosine_similarity)                \
  _(poisson_nll_loss)                 \
  _(cosine_embedding_loss)            \
  _(nll_loss)                         \
  _(nll_loss2d)                       \
  _(hinge_embedding_loss)             \
  _(kl_div)                           \
  _(l1_loss)                          \
  _(smooth_l1_loss)                   \
  _(huber_loss)                       \
  _(mse_loss)                         \
  _(margin_ranking_loss)              \
  _(multilabel_margin_loss)           \
  _(soft_margin_loss)                 \
  _(triplet_margin_loss)              \
  _(multi_margin_loss)                \
  _(binary_cross_entropy_with_logits) \
  _(dist)                             \
  _(pdist)                            \
  _(cdist)                            \
  _(renorm)                           \
  _(logsumexp)                        \
  _(upsample_nearest1d)               \
  _(_upsample_nearest_exact1d)        \
  _(upsample_nearest2d)               \
  _(_upsample_nearest_exact2d)        \
  _(upsample_nearest3d)               \
  _(_upsample_nearest_exact3d)        \
  _(upsample_linear1d)                \
  _(upsample_bilinear2d)              \
  _(_upsample_bilinear2d_aa)          \
  _(upsample_trilinear3d)             \
  _(upsample_bicubic2d)               \
  _(_upsample_bicubic2d_aa)

#define AT_FORALL_FP32_SET_OPT_DTYPE(_) \
  _(prod)                               \
  _(prod, dim_int)                      \
  _(prod, dim_Dimname)                  \
  _(softmax, int)                       \
  _(softmax, Dimname)                   \
  _(log_softmax, int)                   \
  _(log_softmax, Dimname)               \
  _(cumprod)                            \
  _(cumprod, dimname)                   \
  _(cumsum)                             \
  _(cumsum, dimname)                    \
  _(linalg_vector_norm)                 \
  _(linalg_matrix_norm)                 \
  _(linalg_matrix_norm, str_ord)        \
  _(sum)                                \
  _(sum, dim_IntList)                   \
  _(sum, dim_DimnameList)

#define AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(_)                         \
  _(ADD_NS(norm),                                                           \
    "norm.Scalar",                                                          \
    Tensor(const Tensor&, const Scalar&),                                   \
    Tensor(const Tensor&, const std::optional<Scalar>&, ScalarType),        \
    fp32_append_dtype)                                                      \
  _(ADD_NS(norm),                                                           \
    "norm.ScalarOpt_dim",                                                   \
    Tensor(const Tensor&, const std::optional<Scalar>&, IntArrayRef, bool), \
    Tensor(                                                                 \
        const Tensor&,                                                      \
        const std::optional<Scalar>&,                                       \
        IntArrayRef,                                                        \
        bool,                                                               \
        ScalarType),                                                        \
    fp32_append_dtype)                                                      \
  _(ADD_NS(norm),                                                           \
    "norm.names_ScalarOpt_dim",                                             \
    Tensor(const Tensor&, const std::optional<Scalar>&, DimnameList, bool), \
    Tensor(                                                                 \
        const Tensor&,                                                      \
        const std::optional<Scalar>&,                                       \
        DimnameList,                                                        \
        bool,                                                               \
        ScalarType),                                                        \
    fp32_append_dtype)

#define AT_FORALL_PROMOTE(_) \
  _(addcdiv)                 \
  _(addcmul)                 \
  _(atan2)                   \
  _(bilinear)                \
  _(cross)                   \
  _(dot)                     \
  _(vdot)                    \
  _(grid_sampler)            \
  _(index_put)               \
  _(tensordot)               \
  _(scatter_add)
