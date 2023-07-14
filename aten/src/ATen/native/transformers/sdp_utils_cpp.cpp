#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <c10/core/SymInt.h>
#include <c10/util/string_view.h>
#include <cmath>
#include <functional>

namespace sdp {
namespace {
// This helper function creates a constexpr std::array
// From a compile time list of values
template <typename V, typename... T>
constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
  return {{std::forward<T>(t)...}};
}

bool input_requires_grad_cpp(sdp_params params) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

bool has_for_nested_inputs_cpp(sdp_params params) {
  return (
      params.query.is_nested() || params.key.is_nested() ||
      params.value.is_nested());
}

std::array<SDPBackend, num_backends> priority_order_cpp(sdp_params params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      // SDPBackend::efficient_attention,
      SDPBackend::math};

  return default_order;
}

template <typename dtype_vector>
bool check_tensor_dtype_cpp(
    sdp_params params,
    dtype_vector allowed_dtypes,
    bool debug) {
  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), query_dtype) !=
         allowed_dtypes.end()))) {
    if (debug) {
      TORCH_WARN(
          "Expected query, key and value to all be of dtype: {",
          c10::Join(", ", allowed_dtypes),
          "}. Got ",
          "Query dtype: ",
          params.query.dtype(),
          ", Key dtype: ",
          params.key.dtype(),
          ", and Value dtype: ",
          params.value.dtype(),
          " instead.");
    }
    return false;
  }
  return true;
}


bool try_broadcast_param_size_cpp(
    const c10::SymInt q_size,
    const c10::SymInt k_size,
    const c10::SymInt v_size,
    c10::string_view param_name,
    bool debug) {
  auto max_size = std::max({q_size, k_size, v_size});
  if ((q_size != max_size && q_size != 1) ||
      (k_size != max_size && k_size != 1) ||
      (v_size != max_size && v_size != 1)) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels require query, key and value to have broadcastable ",
          param_name,
          "got Query ",
          param_name,
          q_size,
          ", Key ",
          param_name,
          k_size,
          ", Value ",
          param_name,
          v_size,
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper_cpp(
    at::Tensor param,
    c10::string_view param_name,
    bool debug) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  const at::Tensor& sizes = nt_tensor_impl->get_nested_sizes();
  auto num_head_dims = nt_tensor_impl->opt_size(1);
  if (!num_head_dims.has_value()) {
    // num_head_dims is ragged
    if (debug) {
      TORCH_WARN(
          "Fused kernels do not support ragged num_head_dims, ",
          param_name,
          "has a ragged num_heads.");
    }
    return false;
  }

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = param.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] == 0) {
      if (debug) {
        TORCH_WARN(
            "Fused kernels do not support seq_len == 0, ",
            param_name,
            "has a seq len of 0.");
      }
      return false;
    }
  }
  return true;
}

bool check_for_seq_len_0_nested_tensor_cpp(sdp_params params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!has_for_nested_inputs_cpp(params)) {
    return true;
  }

  bool q_is_safe = params.query.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper_cpp(
            params.query, "query ", debug)
      : true;
  // short circuit if any is unsafe
  if (!q_is_safe) {
    return false;
  }

  bool k_is_safe = params.key.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper_cpp(
            params.key, "key ", debug)
      : true;
  if (!k_is_safe) {
    return false;
  }

  bool v_is_safe = params.value.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper_cpp(
            params.value, "value ", debug)
      : true;
  if (!v_is_safe) {
    return false;
  }

  // We now know none of the inputs have ragged num_heads, so we can safely
  // access .size(1)
  auto q_num_heads = params.query.size(1);
  auto k_num_heads = params.key.size(1);
  auto v_num_heads = params.value.size(1);
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!same_num_heads) {
    return try_broadcast_param_size_cpp(
        q_num_heads, k_num_heads, v_num_heads, "num heads ", debug);
  }

  return true;
}

bool check_nested_tensor_cpp(sdp_params params, bool debug) {
  // Return false if have nested tensor
  if (has_for_nested_inputs_cpp(params)) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels of cpp version currently do support Nested Tensor inputs.");
    }
    return false;
  }
  return true;
}

bool check_for_dropout_cpp(sdp_params params, bool debug) {
  if (params.dropout > 0.00001) {
    if (debug) {
      TORCH_WARN("Both fused kernels do not support non-zero dropout.");
    }
    return false;
  }
  return true;
}

bool check_requires_grad_and_nested_cpp(sdp_params params, bool debug) {
  // If we fail both checks then we return false
  if (has_for_nested_inputs_cpp(params) && input_requires_grad_cpp(params)) {
    if (debug) {
      TORCH_WARN(
          "Memory efficient attention currently doesn't support training with NT inputs.");
    }
    return false;
  }
  return true;
}

bool check_for_attn_mask_cpp(sdp_params params, bool debug) {
  if (params.has_attn_mask) {
    if (debug) {
      TORCH_WARN("Both fused kernels do not support non-null attn_mask.");
    }
    return false;
  }
  return true;
}

bool check_tensor_shapes_cpp(sdp_params params, bool debug) {
  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
        (query_dim == 4))) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels requires query, key and value to be 4 dimensional, but got Query dim: ",
          query_dim,
          ", Key dim: ",
          params.key.dim(),
          ", Value dim: ",
          params.value.dim(),
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_safe_kv_broadcast_cpp(at::Tensor param, bool debug) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  auto seq_len = nt_tensor_impl->opt_size(2);
  if (!seq_len.has_value()) {
    if (debug) {
      TORCH_WARN(
          "For both fused kernels, if one of key/value batch_size requires "
          "broadcasting and the other does not, then the other must have a ",
          "consistent seq_len dim.")
    }
    return false;
  }
  return true;
}

bool check_batch_size_and_num_heads_cpp(sdp_params params, bool debug) {
  // This is expected to be called after check_tensor_shapes ensuring that the
  // size() calls won't error since the inputs are all 4 dimensional
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  auto q_num_heads = params.query.sym_size(1);
  auto k_num_heads = params.key.sym_size(1);
  auto v_num_heads = params.value.sym_size(1);
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!(same_batch_size && same_num_heads)) {
    if (debug) {
      TORCH_WARN(
          "For dense inputs, both fused kernels require query, key and value to have the same batch_size and num_heads. ",
          "Query.sizes(): ",
          params.query.sizes(),
          ", Key sizes(): ",
          params.key.sizes(),
          ", Value sizes(): ",
          params.value.sizes(),
          " instead. To broadcast dense inputs, try using unsqueeze and expand_to before passing them into the kernel.");
    }
    return false;
  }
  return true;
}

bool check_head_dim_size_cpp(sdp_params params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (!(query_size_last == key_size_last &&
        query_size_last == value_size_last)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_head_dim_size_mem_efficient_cpp(sdp_params params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  const int64_t alignment = 1;
  if (!(query_size_last == params.key.sym_size(-1) &&
        query_size_last % alignment == 0 && query_size_last > 0 &&
        value_size_last % alignment == 0 && value_size_last > 0)) {
    if (debug) {
      TORCH_WARN(
          "Mem efficient attention requires last dimension of inputs to be divisible by ",
          alignment,
          ". ",
          "Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_runtime_disabled_flash_cpp(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of flash
  // sdp kernels
  if (!at::globalContext().userEnabledFlashSDP()) {
    if (debug) {
      TORCH_WARN("Flash attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

bool check_runtime_disabled_mem_efficient_cpp(sdp_params params, bool debug) {
  // We check the global context to see if user has explicitly turned of
  // mem_efficient sdp kernels
  if (!at::globalContext().userEnabledMemEfficientSDP()) {
    if (debug) {
      TORCH_WARN("Memory Efficient attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

bool use_flash_attention_cpp(sdp_params params, bool debug) {
  constexpr auto cpp_supported_flash_dtypes =
      array_of<at::ScalarType>(at::kFloat, at::kDouble, at::kBFloat16);

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = array_of<bool (*)(sdp_params, bool)>(
      check_runtime_disabled_flash_cpp,
      check_nested_tensor_cpp,
      check_for_dropout_cpp,
      check_tensor_shapes_cpp,
      check_batch_size_and_num_heads_cpp,
      check_for_attn_mask_cpp,
      check_head_dim_size_cpp);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  return check_tensor_dtype_cpp(params, cpp_supported_flash_dtypes, debug);
}

bool use_mem_efficient_attention_cpp(sdp_params params, bool debug) {
  constexpr auto cpp_supported_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kFloat, at::kDouble, at::kBFloat16);

  //  Define gate functions that determine if a mem efficient kernel can be run
  constexpr auto constraints = array_of<bool (*)(sdp_params, bool)>(
      check_runtime_disabled_mem_efficient_cpp,
      check_nested_tensor_cpp,
      check_for_dropout_cpp,
      check_tensor_shapes_cpp,
      check_batch_size_and_num_heads_cpp,
      check_for_attn_mask_cpp,
      check_head_dim_size_mem_efficient_cpp);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  return check_tensor_dtype_cpp(params, cpp_supported_mem_efficient_dtypes, debug);
}
} // namespace

SDPBackend select_sdp_backend_cpp(sdp_params kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order_cpp(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::flash_attention:
        if (use_flash_attention_cpp(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      // case SDPBackend::efficient_attention:
      //   if (use_mem_efficient_attention_cpp(kernel_params, print_debug)) {
      //     return SDPBackend::efficient_attention;
      //   }
      //   break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention or use_mem_efficient did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Memory efficient kernel not used because:");
  use_mem_efficient_attention_cpp(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  use_flash_attention_cpp(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  return SDPBackend::error;
}
} // namespace sdp
