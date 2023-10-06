#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sdp {
namespace {

std::array<SDPBackend, num_backends> priority_order_cpp(sdp_params params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::math};

  return default_order;
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

bool use_flash_attention_cpp(sdp_params params, bool debug) {
  constexpr auto cpp_supported_flash_dtypes =
      array_of<at::ScalarType>(at::kFloat, at::kDouble, at::kBFloat16);

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = array_of<bool (*)(sdp_params, bool)>(
      check_runtime_disabled_flash,
      check_nested_tensor,
      check_for_dropout,
      check_tensor_shapes,
      check_batch_size_and_num_heads,
      check_for_attn_mask,
      check_head_dim_size_cpp,
      check_nonzero_sequence_lengths,
      check_last_dim_stride_equals_1);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  return check_tensor_dtype(params, cpp_supported_flash_dtypes, debug);
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
  TORCH_WARN("Flash attention kernel not used because:");
  use_flash_attention_cpp(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  return SDPBackend::error;
}
} // namespace sdp
