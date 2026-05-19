#pragma once
#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/OptionalRef.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymFloat.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include <string_view>

namespace sdp {

constexpr int32_t num_backends = at::num_sdp_backends;
using SDPBackend = at::SDPBackend;

struct SDPDiagnostics {
  explicit SDPDiagnostics(bool emit_warnings) : emit_warnings(emit_warnings) {}

  void set_current_backend(std::string_view backend) const {
    if (emit_warnings) {
      return;
    }
    backend_failures.emplace_back(std::string(backend), std::vector<std::string>{});
    current_backend_index = backend_failures.size() - 1;
  }

  template <typename... Args>
  void report_failure(Args&&... args) const {
    if (emit_warnings) {
      TORCH_WARN(std::forward<Args>(args)...);
      return;
    }
    add_failure(format(std::forward<Args>(args)...), false);
  }

  template <typename... Args>
  void report_failure_once(Args&&... args) const {
    if (emit_warnings) {
      TORCH_WARN_ONCE(std::forward<Args>(args)...);
      return;
    }
    add_failure(format(std::forward<Args>(args)...), true);
  }

  [[noreturn]] void raise_error() const {
    std::ostringstream message;
    message << "No available kernel. Aborting execution.\nRejected backends:";
    for (const auto& [backend, reasons] : backend_failures) {
      if (reasons.empty()) {
        continue;
      }
      message << "\n" << backend << ": " << reasons.front();
      for (const auto reason_index : c10::irange<size_t>(1, reasons.size())) {
        message << "\n  - " << reasons[reason_index];
      }
    }
    TORCH_CHECK(false, message.str());
  }

 private:
  template <typename... Args>
  static std::string format(Args&&... args) {
    std::ostringstream message;
    (message << ... << std::forward<Args>(args));
    return message.str();
  }

  void add_failure(std::string message, bool once) const {
    TORCH_INTERNAL_ASSERT(current_backend_index < backend_failures.size());
    auto& reasons = backend_failures[current_backend_index].second;
    if (once && std::find(reasons.begin(), reasons.end(), message) != reasons.end()) {
      return;
    }
    reasons.push_back(std::move(message));
  }

 public:
  bool emit_warnings;
  mutable std::vector<std::pair<std::string, std::vector<std::string>>> backend_failures;
  mutable size_t current_backend_index = 0;
};

template <typename... Args>
inline void report_failure(
    c10::OptionalRef<SDPDiagnostics> diagnostics,
    Args&&... args) {
  if (diagnostics.has_value()) {
    diagnostics.get().report_failure(std::forward<Args>(args)...);
  }
}

template <typename... Args>
inline void report_failure_once(
    c10::OptionalRef<SDPDiagnostics> diagnostics,
    Args&&... args) {
  if (diagnostics.has_value()) {
    diagnostics.get().report_failure_once(std::forward<Args>(args)...);
  }
}

// Note that if this changed make sure to update
// the templated enum in mem_eff/kernel_forward.h and mem_eff/kernel_backward.h
enum class CustomMaskType {
  NoCustomMask = 0,
  CausalFromTopLeft = 1,
  CausalFromBottomRight = 2,
  NumCustomMaskTypes,
};

struct sdp_params {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  std::optional<at::Tensor> attn_mask;
  double dropout;
  bool is_causal;
  bool enable_gqa;
};

SDPBackend select_sdp_backend_cpp(sdp_params const& kernel_params);

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

inline bool input_requires_grad(sdp_params const& params) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

inline bool has_for_nested_inputs(sdp_params const& params) {
  return
      (params.query.is_nested() && params.query.layout() == c10::kStrided) ||
      (params.key.is_nested() && params.key.layout() == c10::kStrided) ||
      (params.value.is_nested() && params.value.layout() == c10::kStrided);
}

inline bool has_for_dense_inputs(sdp_params const& params) {
  return !params.query.is_nested() || !params.key.is_nested() || !params.value.is_nested();
}

inline bool has_only_dense_inputs(sdp_params const& params) {
  return !params.query.is_nested() && !params.key.is_nested() && !params.value.is_nested();
}

template <typename dtype_vector>
inline bool check_tensor_dtype(
    sdp_params const& params,
    dtype_vector allowed_dtypes,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), query_dtype) !=
         allowed_dtypes.end()))) {
    report_failure(
        diagnostics,
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
    return false;
  }
  return true;
}


inline bool try_broadcast_param_size(
    const c10::SymInt q_size,
    const c10::SymInt k_size,
    const c10::SymInt v_size,
    std::string_view param_name,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  auto max_size = std::max({q_size, k_size, v_size});
  if ((q_size != max_size && q_size != 1) ||
      (k_size != max_size && k_size != 1) ||
      (v_size != max_size && v_size != 1)) {
    report_failure(
        diagnostics,
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
    return false;
  }
  return true;
}

inline bool check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
    at::Tensor const& param,
    std::string_view param_name,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  const at::Tensor& sizes = nt_tensor_impl->get_nested_sizes();
  auto num_head_dims = nt_tensor_impl->opt_size(1);
  if (!num_head_dims.has_value()) {
    report_failure(
        diagnostics,
        "Fused kernels do not support ragged num_head_dims, ",
        param_name,
        "has a ragged num_heads.");
    return false;
  }

  auto* sizes_ptr = sizes.const_data_ptr<int64_t>();
  const int64_t n_tensors = param.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] == 0) {
      report_failure(
          diagnostics,
          "Fused kernels do not support seq_len == 0, ",
          param_name,
          "has a seq len of 0.");
      return false;
    }
  }
  return true;
}

inline bool check_for_seq_len_0_nested_tensor(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // When this function is called we are assured that the nt is dim==4
  bool q_is_safe = params.query.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.query, "query ", diagnostics)
      : true;
  // short circuit if any is unsafe
  if (!q_is_safe) {
    return false;
  }

  bool k_is_safe = params.key.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.key, "key ", diagnostics)
      : true;
  if (!k_is_safe) {
    return false;
  }

  bool v_is_safe = params.value.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.value, "value ", diagnostics)
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
    if (input_requires_grad(params)){
      report_failure(
          diagnostics,
          "Both fused kernels do not support training with broadcasted NT inputs.");
      return false;
    }
    return try_broadcast_param_size(
        q_num_heads, k_num_heads, v_num_heads, "num heads ", diagnostics);
  }

  return true;
}

inline bool check_nested_tensor(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // Return false if have nested tensor
  if (!has_only_dense_inputs(params)) {
    report_failure(
        diagnostics,
        "Both fused kernels of cpp version currently do not support Nested Tensor inputs.");
    return false;
  }
  return true;
}

inline bool check_for_dropout(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  if (params.dropout > 0.0) {
    report_failure(diagnostics, "Both fused kernels do not support non-zero dropout.");
    return false;
  }
  return true;
}

inline bool check_requires_grad_and_nested(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  if (input_requires_grad(params)) {
    report_failure(
        diagnostics,
        "Memory efficient attention currently doesn't support training with NT inputs.");
    return false;
  }
  return true;
}

inline bool check_for_attn_mask(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  if (params.attn_mask.has_value()) {
    report_failure(diagnostics, "Flash Attention does not support non-null attn_mask.");
    return false;
  }
  return true;
}

inline bool check_attn_mask_shape(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  auto attn_mask = params.attn_mask;
  if (!attn_mask.has_value()) {
    return true;
  }
  if (attn_mask.value().requires_grad()) {
    return false;
  }
  auto batchSize = params.query.sym_size(0);
  auto qSize = params.query.sym_size(2);
  auto kvSize = params.key.sym_size(2);
  auto num_head = params.query.sym_size(1);

  // Helper to check if a mask dim is compatible with a target dim.
  // Compatible means: symbolically equal, or the mask dim is concretely 1
  // (broadcast). Returns false (conservatively reject) when neither can be
  // determined without guarding on unbacked symbolic ints.
  auto dim_compatible = [](const c10::SymInt& mask_dim,
                           const c10::SymInt& target_dim) -> bool {
    if (TORCH_STATICALLY_KNOWN_TRUE(mask_dim == target_dim)) {
      return true;
    }
    auto mask_int = mask_dim.maybe_as_int();
    return mask_int.has_value() && *mask_int == 1;
  };

  auto mask_qsize = attn_mask.value().sym_size(-2);
  if (!dim_compatible(mask_qsize, qSize)) {
    return false;
  }
  auto mask_kvsize = attn_mask.value().sym_size(-1);
  if (!dim_compatible(mask_kvsize, kvSize)) {
    return false;
  }
  if (attn_mask.value().dim() == 2) {
    return true;
  } else if (attn_mask.value().dim() == 4) {
    auto mask_b = attn_mask.value().sym_size(0);
    auto mask_h = attn_mask.value().sym_size(1);
    if (dim_compatible(mask_b, batchSize) &&
        dim_compatible(mask_h, num_head)) {
      return true;
    }
  }
  report_failure(
      diagnostics,
      "Please use the following attn mask shapes: ",
      "2d - ({Q_seq_len, 1}  x {KV_seq_len, 1}); ",
      "4d - ({Batch, 1} x {Num_heads, 1} x {Q_seq_len, 1}  x {KV_seq_len, 1})");
  return false;
}

inline bool check_tensor_shapes(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
        (query_dim == 4))) {
    report_failure(
        diagnostics,
        "All fused kernels requires query, key and value to be 4 dimensional, but got Query dim: ",
        query_dim,
        ", Key dim: ",
        params.key.dim(),
        ", Value dim: ",
        params.value.dim(),
        " instead.");
    return false;
  }
  return true;
}

inline bool check_safe_kv_broadcast(
    at::Tensor const& param,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  auto seq_len = nt_tensor_impl->opt_size(2);
  if (!seq_len.has_value()) {
    report_failure(
        diagnostics,
        "For both fused kernels, if one of key/value batch_size requires "
        "broadcasting and the other does not, then the other must have a ",
        "consistent seq_len dim.");
    return false;
  }
  return true;
}

template <bool requires_same_num_heads=true>
inline bool check_grouped_query_attention(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  const auto q_num_heads = params.query.sym_size(-3);
  const auto k_num_heads = params.key.sym_size(-3);
  const auto v_num_heads = params.value.sym_size(-3);
  const bool same_kv_heads = k_num_heads == v_num_heads;

  if (requires_same_num_heads && !same_kv_heads){
    report_failure(
        diagnostics,
        "Both fused kernels require key and value to have the same num_heads and batch_size but got: ",
        "Key sizes: ",
        params.key.sizes(),
        ", Value sizes: ",
        params.value.sizes(),
        ", Query sizes: ",
        params.query.sizes(),
        " instead.");
    return false;
  }
  // Check if grouped query attention is supported and validate the number of
  // heads
  if (q_num_heads % k_num_heads != 0 || (!requires_same_num_heads && (q_num_heads % v_num_heads != 0))) {
    report_failure(
        diagnostics,
        "The number of heads in key/value must divide number of heads in query.",
        "Got input Key sizes(): ",
        params.key.sym_size(-3),
        ", Value sizes(): ",
        params.value.sym_size(-3),
        ", Query sizes(): ",
        params.query.sym_size(-3),
        " instead.");
    return false;
  }
  return true;
}

template <bool supports_gqa, bool requires_same_num_heads=true>
inline bool check_batch_size_and_num_heads_dense(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // This is expected to be called after check_tensor_shapes ensuring that the
  // size() calls won't error since the inputs are all 4 dimensional

  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  auto q_num_heads = params.query.sym_size(-3);
  auto k_num_heads = params.key.sym_size(-3);
  auto v_num_heads = params.value.sym_size(-3);

  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!same_batch_size){
    report_failure(
        diagnostics,
        "For dense inputs, both fused kernels require query, key and value to have the same batch_size. ",
        "Query.sizes(): ",
        params.query.sizes(),
        ", Key.sizes(): ",
        params.key.sizes(),
        ", Value.sizes(): ",
        params.value.sizes(),
        " instead. To broadcast dense inputs, try using unsqueeze and expand_to before passing them into the kernel.");
    return false;
  }

  if(params.enable_gqa && supports_gqa){
    return check_grouped_query_attention<requires_same_num_heads>(params, diagnostics);
  }

  // same num heads condition for non-gqa case
  if (!same_num_heads){
    report_failure(
        diagnostics,
        "For dense input, both fused kernels require query, key and value to have the same num_heads. ",
        "Query.sizes(): ",
        params.query.sizes(),
        ", Key sizes(): ",
        params.key.sizes(),
        ", Value sizes(): ",
        params.value.sizes(),
        " instead. To broadcast dense inputs, try using unsqueeze and expand_to before passing them into the kernel.");
    return false;
  }
  // If all checks pass, return true
  return true;
}

inline bool check_batch_size_nested(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // This is expected to be called after check_tensor_shapes ensuring that the
  // size() calls won't error since the inputs are all 4 dimensional
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  // num_heads logic for nested input is checked in
  // check_for_seq_len_0_nested_tensor as there is handling there to make sure
  // num_heads is not ragged
  bool broadcastable_batch_size = true;
  if (!same_batch_size) {
    if (input_requires_grad(params)){
      report_failure(
          diagnostics,
          "Both fused kernels do not support training with broadcasted NT inputs.");
      return false;
    }
    // try to broadcast batchsize
    broadcastable_batch_size = try_broadcast_param_size(
        q_batch_size, k_batch_size, v_batch_size, "batch size ", diagnostics);

    // if only one of k or v require broadcasting of batch size, the other
    // must have a consistent seq_len dim
    if (broadcastable_batch_size) {
      if (k_batch_size == 1 && v_batch_size != 1 &&
          !check_safe_kv_broadcast(params.value, diagnostics)) {
        return false;
      }
      if (v_batch_size == 1 && k_batch_size != 1 &&
          !check_safe_kv_broadcast(params.key, diagnostics)) {
        return false;
      }
    }
  }
  return broadcastable_batch_size;
}

inline bool check_nonzero_sequence_lengths_dense(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // In some cases people will pass in 0 sized tensors, this will
  // cause the fused path to error with unaligned mask
  bool zero_seq_len_q = params.query.sym_size(-2) == 0;
  bool zero_seq_len_k = params.key.sym_size(-2) == 0;
  if (zero_seq_len_q || zero_seq_len_k) {
    report_failure(
        diagnostics,
        "All fused kernels do not support zero seq_len_q or seq_len_kv.");
    return false;
  }
  return true;
}

template<bool ignore_singleton_dim>
inline bool check_last_dim_stride_equals_1_dense(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // The stride checking for NestedTensors is done within the kernel
  // And .contiguous will be called if needed

  // This function checks that the last dimension of the inputs to
  // fused_attention have stride 1
  bool qkv_strides_equal_1 = params.query.sym_stride(-1) == 1 &&
      params.key.sym_stride(-1) == 1 && params.value.sym_stride(-1) == 1;

  // https://github.com/pytorch/pytorch/issues/116333
  // If the head_dim is size 1 the stride won't matter, but we
  // check this condition before padding the head_dim to 1
  if (ignore_singleton_dim){
    qkv_strides_equal_1 = qkv_strides_equal_1 || params.query.sym_size(-1) == 1;
  }
  bool is_cpu = params.query.device().type() == c10::DeviceType::CPU;
  bool mask_stride_equal_1 = params.attn_mask.has_value()
      ? params.attn_mask.value().sym_stride(-1) == 1
      : true;
  bool mask_stride_valid = is_cpu ? true : mask_stride_equal_1;
  if (!(qkv_strides_equal_1 && mask_stride_valid)) {
    if (diagnostics.has_value()) {
      std::ostringstream message;
      message
          << "All fused kernels require the last dimension of the input to have stride 1. ";
      message << "Got Query.stride(-1): " << params.query.sym_stride(-1)
              << ", Key.stride(-1): " << params.key.sym_stride(-1)
              << ", Value.stride(-1): " << params.value.sym_stride(-1);

      if (params.attn_mask.has_value()) {
        message
            << ", Attn_mask.stride(-1): "
            << params.attn_mask.value().sym_stride(-1)
            << " (GPU backends require attn_mask's last dimension to have stride 1 while the CPU does not).";
      }
      diagnostics.get().report_failure(message.str());
    }

    return false;
  }
  return true;
}

inline bool check_runtime_disabled_flash(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // We check the global context to see if user has explicitly turned of flash
  // sdp kernels
  if (!at::globalContext().userEnabledFlashSDP()) {
    report_failure(diagnostics, "Flash attention has been runtime disabled.");
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_mem_efficient(
    sdp_params const& params,
    c10::OptionalRef<SDPDiagnostics> diagnostics = {}) {
  // We check the global context to see if user has explicitly turned of
  // mem_efficient sdp kernels
  if (!at::globalContext().userEnabledMemEfficientSDP()) {
    report_failure(diagnostics, "Memory Efficient attention has been runtime disabled.");
    return false;
  }
  return true;
}


} // namespace sdp
