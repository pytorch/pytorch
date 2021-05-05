#pragma once

namespace at {
namespace autocast {

namespace {
  bool is_autocast_eligible(const Tensor& tensor) {
    return (tensor.is_cuda() || tensor.is_xla()) && tensor.is_floating_point();
  }
} // namespace

TORCH_API bool is_enabled();
TORCH_API void set_enabled(bool enabled);
TORCH_API void clear_cache();
TORCH_API int increment_nesting();
TORCH_API int decrement_nesting();

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
  if (is_autocast_eligible(nextArg)) {
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
  return (arg.defined() && is_autocast_eligible(arg) && (arg.scalar_type() != at::kDouble));
}

// Overload to catch Tensor args
TORCH_API Tensor cached_cast(at::ScalarType to_type, const Tensor& arg);

// Overload to process optional<Tensor>
inline c10::optional<Tensor> cached_cast(at::ScalarType to_type, const c10::optional<Tensor>& arg) {
  if (arg.has_value()) {
    return cached_cast(to_type, *arg);
  } else {
    return c10::nullopt;
  }
}

// Overload to process TensorLists
inline std::vector<Tensor> cached_cast(at::ScalarType to_type, const TensorList& arg) {
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
