#pragma once

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <torch/library.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

namespace at::autocast {

TORCH_API bool is_enabled();
TORCH_API void set_enabled(bool enabled);
TORCH_API void clear_cache();
TORCH_API int increment_nesting();
TORCH_API int decrement_nesting();
TORCH_API bool is_cpu_enabled();
TORCH_API void set_cpu_enabled(bool enabled);
TORCH_API at::ScalarType get_autocast_gpu_dtype();
TORCH_API at::ScalarType get_autocast_cpu_dtype();
TORCH_API void set_autocast_gpu_dtype(at::ScalarType dtype);
TORCH_API void set_autocast_cpu_dtype(at::ScalarType dtype);
TORCH_API bool is_xpu_enabled();
TORCH_API void set_xpu_enabled(bool enabled);
TORCH_API at::ScalarType get_autocast_xpu_dtype();
TORCH_API void set_autocast_xpu_dtype(at::ScalarType dtype);
TORCH_API bool is_ipu_enabled();
TORCH_API void set_ipu_enabled(bool enabled);
TORCH_API at::ScalarType get_autocast_ipu_dtype();
TORCH_API void set_autocast_ipu_dtype(at::ScalarType dtype);
TORCH_API bool is_hpu_enabled();
TORCH_API void set_hpu_enabled(bool enabled);
TORCH_API at::ScalarType get_autocast_hpu_dtype();
TORCH_API void set_autocast_hpu_dtype(at::ScalarType dtype);
TORCH_API bool is_xla_enabled();
TORCH_API void set_xla_enabled(bool enabled);
TORCH_API at::ScalarType get_autocast_xla_dtype();
TORCH_API void set_autocast_xla_dtype(at::ScalarType dtype);
TORCH_API bool is_privateuseone_enabled();
TORCH_API void set_privateuseone_enabled(bool enabled);
TORCH_API at::ScalarType get_autocast_privateuseone_dtype();
TORCH_API void set_autocast_privateuseone_dtype(at::ScalarType dtype);
TORCH_API bool is_autocast_cache_enabled();
TORCH_API void set_autocast_cache_enabled(bool enabled);

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
      return tensor.device().type() == c10::DeviceType::PrivateUse1 &&
          tensor.is_floating_point();
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
    default:
      throw std::runtime_error(
          "unknown device type for autocast in get_autocast_dispatch_key_from_device_type");
  }
}

inline at::ScalarType get_lower_precision_fp_from_device_type(
    c10::DeviceType device_type) {
  switch (device_type) {
    case c10::DeviceType::CUDA:
      return get_autocast_gpu_dtype();
    case c10::DeviceType::CPU:
      return get_autocast_cpu_dtype();
    case c10::DeviceType::XPU:
      return get_autocast_xpu_dtype();
    case c10::DeviceType::IPU:
      return get_autocast_ipu_dtype();
    case c10::DeviceType::HPU:
      return get_autocast_hpu_dtype();
    case c10::DeviceType::XLA:
      return get_autocast_xla_dtype();
    case c10::DeviceType::PrivateUse1:
      return get_autocast_privateuseone_dtype();
    default:
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

// Overload to process optional<Tensor>
inline c10::optional<Tensor> cached_cast(
    at::ScalarType to_type,
    const c10::optional<Tensor>& arg,
    c10::DeviceType device_type = c10::DeviceType::CUDA) {
  if (arg.has_value()) {
    return cached_cast(to_type, *arg, device_type);
  } else {
    return c10::nullopt;
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
c10::optional<ScalarType> inline set_opt_dtype(
    at::ScalarType to_type,
    const c10::optional<ScalarType>& dtype) {
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
                      //  2. have a c10::optional<ScalarType> arg that controls
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

// Common cases where registration signature matches redispatch signature
// (that's why SIGNATURE is repeated in the WrapFunction instantiation)
#define KERNEL(DISPATCHKEY, OP, POLICY)       \
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

// KERNEL_CPU/KERNEL_CPU2/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CPU
// registration for AutocastCPU
#define KERNEL_CPU(OP, POLICY) KERNEL(c10::DeviceType::CPU, OP, POLICY)

#define KERNEL_CPU2(OP, OVERLOAD, POLICY) \
  KERNEL2(c10::DeviceType::CPU, OP, OVERLOAD, POLICY)

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

// KERNEL_CUDA/KERNEL_CUDA2/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA
// registration for AutocastCUDA
#define KERNEL_CUDA(OP, POLICY) KERNEL(c10::DeviceType::CUDA, OP, POLICY)

#define KERNEL_CUDA2(OP, OVERLOAD, POLICY) \
  KERNEL2(c10::DeviceType::CUDA, OP, OVERLOAD, POLICY)

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

// KERNEL_PRIVATEUSEONE/KERNEL_PRIVATEUSEONE2/
// KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE
// registration for AutocastPrivateUse1
#define KERNEL_PRIVATEUSEONE(OP, POLICY) \
  KERNEL(c10::DeviceType::PrivateUse1, OP, POLICY)

#define KERNEL_PRIVATEUSEONE2(OP, OVERLOAD, POLICY) \
  KERNEL2(c10::DeviceType::PrivateUse1, OP, OVERLOAD, POLICY)

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
