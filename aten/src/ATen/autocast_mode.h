#pragma once

namespace at {
namespace autocast {

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
TORCH_API bool is_autocast_cache_enabled();
TORCH_API void set_autocast_cache_enabled(bool enabled);


namespace {
  bool is_autocast_eligible(const Tensor& tensor, DeviceType device_type) {
    return device_type == DeviceType::CUDA
        ? (tensor.is_cuda() || tensor.is_xla()) && tensor.is_floating_point()
        : (tensor.is_cpu() || tensor.is_mkldnn()) && tensor.is_floating_point();
  }
} // namespace

inline DispatchKey get_autocast_dispatch_key_from_device_type(
    DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CUDA:
      return DispatchKey::Autocast;
    case DeviceType::CPU:
      return DispatchKey::AutocastCPU;
    default:
      throw std::runtime_error(
          "unknown device type for autocast in get_autocast_dispatch_key_from_device_type");
  }
}

inline at::ScalarType get_lower_precision_fp_from_device_type(
    DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CUDA:
      return get_autocast_gpu_dtype();
    case DeviceType::CPU:
      return get_autocast_cpu_dtype();
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
    DeviceType device_type=DeviceType::CUDA) {
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
    DeviceType device_type=DeviceType::CUDA) {
  for (const auto& tensor : list) {
    current = prioritize(current, tensor, device_type);
  }
  return current;
}

// Template to catch non-Tensor args (no-op that returns current best guess)
template<typename T>
inline at::ScalarType prioritize(
    at::ScalarType current,
    T nextArg,
    DeviceType device_type=DeviceType::CUDA) {
  return current;
}

// Overload for the tail case.
inline at::ScalarType promote_type(
    at::ScalarType current,
    DeviceType device_type) {
  return current;
}

// Unpack args and determine if incoming lower_precision_fp tensors need to be promoted to float32.
// Non-Tensor arguments are ignored.
template<typename Arg0, typename... Args>
inline at::ScalarType promote_type(
    at::ScalarType current,
    DeviceType device_type,
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
    DeviceType device_type=DeviceType::CUDA) {
  return (arg.defined() &&
          is_autocast_eligible(arg, device_type) &&
          (arg.scalar_type() != at::kDouble));
}

// Overload to catch Tensor args
TORCH_API Tensor cached_cast(
    at::ScalarType to_type,
    const Tensor& arg,
    DeviceType device_type=DeviceType::CUDA);

// Overload to process optional<Tensor>
inline c10::optional<Tensor> cached_cast(
    at::ScalarType to_type,
    const c10::optional<Tensor>& arg,
    DeviceType device_type=DeviceType::CUDA) {
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
    DeviceType device_type=DeviceType::CUDA) {
  std::vector<Tensor> vec;
  vec.reserve(arg.size());
  for (const auto& t : arg) {
    vec.push_back(cached_cast(to_type, t, device_type));
  }
  return vec;
}

// Template to catch non-Tensor args.
template<typename T>
inline T cached_cast(
    at::ScalarType to_type,
    T arg,
    DeviceType device_type=DeviceType::CUDA) {
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
c10::optional<ScalarType> inline set_opt_dtype(at::ScalarType to_type, const c10::optional<ScalarType>& dtype) {
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

} // namespace autocast
} // namespace at
