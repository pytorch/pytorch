#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/core/ScalarType.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <c10/util/Array.h>
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/cudnn-wrapper.h>
#endif

#include <c10/core/SymInt.h>

#if USE_ROCM
#if defined(USE_FLASH_ATTENTION) || defined(USE_MEM_EFF_ATTENTION)
#include <ATen/native/transformers/hip/aotriton_versions.h>
#include <aotriton/flash.h>
#define USE_ROCM_ATTENTION 1
#endif
#else
#define USE_ROCM_ATTENTION 0
#endif

// Avoid potential compiler -Wall -Werror complains undefined macro
#ifndef AOTRITON_VERSION_MINOR
#define AOTRITON_VERSION_MINOR 0
#endif

/**
* Note [SDPA Runtime Dispatch]
* SDPA relies on a runtime dispatch mechanism to select the appropriate
* kernel. This file contains exposes this through the `select_sdp_backend`
* The basic structure of this function is to call `priority_order` to get a
* list of backends to try, and then iterate through them until one succeeds.
* Each backend defines a use_<backend> function that returns true if the
* backend can be run with the given SDP parameters. The use_<backend> function
* will iterate over a list of "filters" that check for specific properties of
* the SDP parameters. If all filters pass, the backend can be used and use_<backend>
* returns true. If any filter fails, then use_<backend> returns false.
*
* In order to aid in debugging, each filter takes sdp_params and a debug flag.
* If the debug flag is set, the filter will print a warning message if it fails.
* The behavior of select_sdp_backend is to return the first backend that
* succeeds. If no backend is viable then it will run each use_<backend> function
* with debug=true and return SDPBackend::error.
*/

namespace sdp {
namespace {

// tracks whether we've set the default priority order once, to avoid setting
// it redundantly or overwriting a user-specified priority order
// when the priority order context manager is used before the default priority
// order is initialized the following happens:
// (1) the current priority order is queried
// (2) priority_order() is called, which initializes it to the default as init_ is false
// (3) the user-specified priority order is set
// (3.1) we are in the priority context...
// (3.2) we exit the priority context...
// (4) the previous priority order (default) is restored
bool priority_order_init_ = false;

// TODO(eqy): more benchmarking to determine whether this should include sm86/89
// Needs to be kept in-sync with test_fused_chocie in test_transformers.py
bool check_prefer_cudnn_attention() {
  static const bool prefer_cudnn = c10::utils::check_env("TORCH_CUDNN_SDPA_PREFERRED") != false;
  if (!prefer_cudnn) {
    return false;
  }
#if (defined(CUDNN_VERSION) && (CUDNN_VERSION >= 90900))
  auto dprops = at::cuda::getCurrentDeviceProperties();
  auto major = dprops->major;
  return (major == 9 || major == 10) && !dprops->minor;
#else
  return false;
#endif
}

// flash_attention V2 is universally faster than efficient_attention and Math
std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
  if (!priority_order_init_) {
    priority_order_init_ = true;
    if (check_prefer_cudnn_attention()) {
        const std::vector<int64_t> cudnn_order = {static_cast<int64_t>(at::SDPBackend::cudnn_attention),
                                                  static_cast<int64_t>(at::SDPBackend::flash_attention),
                                                  static_cast<int64_t>(at::SDPBackend::efficient_attention),
                                                  static_cast<int64_t>(at::SDPBackend::math)};
        at::globalContext().setSDPPriorityOrder(cudnn_order);
    }
  }
  return at::globalContext().sDPPriorityOrder();
}

bool use_tensor_cores(sdp_params const& params, cudaDeviceProp* dprops, bool is_half) {
  if (dprops->major >= 8) {
    return true;
  }
  if (dprops->major >= 7) {
    return is_half;
  }
  return false;
}
int64_t minimum_gemm_alignment(sdp_params const& params) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_half = (params.query.dtype() == at::kHalf) ||
      (params.query.dtype() == at::kBFloat16);
  bool use_tc = use_tensor_cores(params, dprops, is_half);
  int64_t matmul_alignment_mn = 1;
  if (dprops->major >= 8) {
    matmul_alignment_mn = 4;
  }
  int64_t bits_per_scalar = is_half ? 16 : 32;
  if (use_tc) {
    matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar);
  }
  return matmul_alignment_mn;
}

// On ROCM, ME and FA share the backend, and hence they share the checking
// function for fundamental limitations by the GPU kernel
// caller_is_meff is added to make the TORCH_WARN message showing the correct result
template<bool caller_is_meff = false>
bool check_head_dim_size_flash(sdp_params const& params, bool debug) {
#if USE_ROCM_ATTENTION
  // AOTriton 0.9+ supports head_dim up to 512
  const static auto max_hdim = []() {
#if AOTRITON_VERSION_CURRENT == AOTRITON_VERSION_INT(0, 11)
    // gfx11xx only support hdim <= 256 on AOTriton 0.11
    auto dprops = at::cuda::getCurrentDeviceProperties();
    const c10::basic_string_view<char> arch(dprops->gcnArchName);
    if (arch.starts_with("gfx11")) {
      return 256;
    }
#endif // AOTriton 0.11
#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 9)
    return 512;
#else
    return 256;
#endif
  }();
  const auto max_size = c10::SymInt(max_hdim);
#else
  // All head_dim sizes must be equal and less than 256
  const auto max_size = c10::SymInt(256);
#endif
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          caller_is_meff ? "Efficient attention on ROCM" : "Flash attention",
          " requires q,k,v to have the same last dimension and to be less than or equal to 256.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }
  if constexpr(caller_is_meff) {
    bool is_half = (params.query.dtype() == at::kHalf) ||
      (params.query.dtype() == at::kBFloat16);
    const int64_t alignment = is_half ? 8 : 4;
    if (!(query_size_last % alignment == 0 && query_size_last > 0 &&
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
  }
  return true;
}

// See check_head_dim_size_flash above for the purpose of caller_is_meff
template<bool caller_is_meff = false>
bool check_head_dim_size_flash_nested(sdp_params const& params, bool debug) {
  const auto max_size = c10::SymInt(256);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last % 8 == 0) &&
        (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "For NestedTensor inputs,",
          caller_is_meff ? " Efficient attention on ROCM " : " Flash attention",
          " requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 256.",
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

bool check_head_dim_size_mem_efficient(sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  const int64_t alignment = minimum_gemm_alignment(params);
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

template <int Major, int Minor>
struct SMVersion {
  static constexpr int major = Major;
  static constexpr int minor = Minor;
  constexpr SMVersion() = default;
};

/**
 * Checks if the current CUDA device architecture is inclusively within the specified range.
 *
 * @param lower_bound The lower bound of the CUDA device architecture range.
 * @param upper_bound The upper bound of the CUDA device architecture range.
 * @param params The parameters for the current operation.
 * @return True if the current CUDA device architecture is within the specified range, false otherwise.
 */
template <typename lower_bound, typename upper_bound>
bool check_sm_version(cudaDeviceProp * dprops) {
  bool is_gte_lower_bound = dprops->major > lower_bound::major ||
      (dprops->major == lower_bound::major &&
       dprops->minor >= lower_bound::minor);
  bool is_lte_upper_bound = dprops->major < upper_bound::major ||
      (dprops->major == upper_bound::major &&
       dprops->minor <= upper_bound::minor);
  return is_gte_lower_bound && is_lte_upper_bound;
}

bool check_flash_attention_hardware_support(sdp_params const& params, bool debug) {
  // Check that the gpu is capable of running flash attention
  using sm80 = SMVersion<8, 0>;
  using sm121 = SMVersion<12, 1>;
#if USE_ROCM
#if USE_ROCM_ATTENTION
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    // User explicitly set CK as the flash attention backend. Return true for now
    // TODO: Flesh out sanity checks
    return true;
  } else {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (hipSuccess != aotriton::v2::flash::check_gpu(stream)) {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (debug) {
            TORCH_WARN(
                    "Flash attention was not compiled for current AMD GPU architecture. Attempting to run on architecture ", dprops->gcnArchName);
        }
        return false;
    }
#if AOTRITON_VERSION_MINOR >= 9
    if (aotriton::isArchExperimentallySupported(stream)) {
      static const bool enable_experimental = c10::utils::check_env("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == true;
      if (!enable_experimental) {
        TORCH_WARN_ONCE("Flash Efficient attention on Current AMD GPU is still experimental."
            " Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.");
        return false;
      }
    }
#endif
  }
#else
  return false;
#endif
#else
  if (!at::cuda::is_available()) {
    if (debug) {
      TORCH_WARN("flash attention requires a CUDA device, which is not available.");
    }
    return false;
  }
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm80, sm121>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
#endif
  return true;
}

bool check_mem_efficient_hardware_support(sdp_params const& params, bool debug) {
  // Mem Efficient attention supports hardware in the range [sm_50, sm_90]
  using sm50 = SMVersion<5, 0>;
  using sm121 = SMVersion<12, 1>;
#if USE_ROCM
#if USE_ROCM_ATTENTION
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    // User explicitly set CK as the flash attention backend. Return true for now
    // TODO: Flesh out sanity checks
    return true;
  } else {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (hipSuccess != aotriton::v2::flash::check_gpu(stream)) {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (debug) {
            TORCH_WARN(
                    "Mem Efficient attention was not compiled for current AMD GPU architecture. Attempting to run on architecture ", dprops->gcnArchName);
        }
        return false;
    }
#if AOTRITON_VERSION_MINOR >= 9
    if (aotriton::isArchExperimentallySupported(stream)) {
      static const bool enable_experimental = c10::utils::check_env("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == true;
      if (!enable_experimental) {
        TORCH_WARN_ONCE("Mem Efficient attention on Current AMD GPU is still experimental."
            " Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.");
        return false;
      }
    }
#endif
  }
#else
  return false;
#endif
#else
  if (!at::cuda::is_available()) {
    if (debug) {
      TORCH_WARN("Mem Efficient attention requires a CUDA device, which is not available.");
    }
    return false;
  }
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm50, sm121>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "Mem Efficient Attention only supports gpu architectures in the range [sm50, sm121]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
#endif
  return true;
}

bool check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120(
    sdp_params const& params,
    bool debug) {
  // Flash Attention will raise an error in the backward pass if the head_dim
  // size is greater than 192 And the device is between in the range [sm86, sm89]
  using sm86 = SMVersion<8, 6>;
  using sm89 = SMVersion<8, 9>;
  using sm120 = SMVersion<12, 0>;
  using sm121 = SMVersion<12, 1>;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm86_or_sm89 = check_sm_version<sm86, sm89>(dprops);
  bool is_sm120_or_sm121 = check_sm_version<sm120, sm121>(dprops);
  bool is_head_dim_gt192 = params.query.sym_size(-1) > 192;
  bool is_head_dim_lte224 = params.query.sym_size(-1) <= 224;
  bool is_dropout = params.dropout > 0.0;
  //  head_dim size  in (192, 224] is not supported on sm86 and sm89
  bool cond1 = is_head_dim_gt192 && is_head_dim_lte224;
  // head_dim size > 224 and is_dropout is not supported on sm86 and sm89
  bool cond2 = params.query.sym_size(-1) > 224 && is_dropout;
  if (input_requires_grad(params) && (is_sm86_or_sm89 || is_sm120_or_sm121) && (cond1 || cond2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention currently doesn't support training with head_dim ∈ (192, 224] or "
          "(head_dim ∈ (224, 256] and dropout > 0.0) on gpu architectures in the range[sm86, sm89].",
          "Attempting to run with dropout set to: ", params.dropout,
          "and head_dim: ",
          params.query.sym_size(-1), " on a sm ", dprops->major, ".",
          dprops->minor, " gpu.");
    }
    return false;
  }
  return true;
}

bool check_flash_causal_non_square_seqlens(sdp_params const& params, bool debug) {
  // FlashAttention 2 updated the default mask meaning for causal in this PR:
  // 9e5e8bc91e it is now aligned to lower_right which would be a BC break
  // for non-square masks. We will not support non-square masks for causal w/ FAV2
  if (params.is_causal &&
      !params.query.is_nested() && !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ", params.query.sym_size(-2), " seqlen_k: ",
          params.key.sym_size(-2), ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}

bool check_all_tensors_on_device(sdp_params const& params, bool debug) {
  // Check that all tensors are on the GPU device
  // This should be handled by the stub dispatch, but we call can_use_*_attention
  // directly from python we need to ensure that the tensors are on cuda
  if (params.query.device().type() != at::DeviceType::CUDA) {
    if (debug) {
      TORCH_WARN(
          "All tensors need to be on cuda device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  return true;
}

bool check_cudnn_tensor_shapes(sdp_params const& params, bool debug) {
  const auto s_q = params.query.sym_size(2);
  const auto s_k = params.key.sym_size(2);
  const auto d_qk = params.query.sym_size(3);
  const auto d_v = params.value.sym_size(3);
  long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
  if (cudnn_version < 8903) {
    if (debug) {
      TORCH_WARN("SDPA fprop requires cudnn 8.9.3 or higher");
    }
    return false;
  }
  if (cudnn_version < 8906 && params.dropout != 0.0) {
    if (debug) {
      TORCH_WARN("Dropout reference is only supported on 8.9.6 onwards.");
    }
    return false;
  }
  auto head_dim_limit = 128;
  if (cudnn_version >= 91000) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    if (dprops->major == 9 && !dprops->minor) {
      head_dim_limit = 256;
    }
  }
  if (d_qk > head_dim_limit || d_v > head_dim_limit) {
    if (debug) {
      TORCH_WARN("head_dim should be no more than ", head_dim_limit);
    }
    return false;
  }
  if (d_qk % 8 != 0 || d_v % 8 != 0) {
    if (debug) {
      TORCH_WARN("head_dim should be a multiple of 8");
    }
    return false;
  }
  if (cudnn_version < 8906 && s_k % 64 != 0 ) {
    if (debug) {
      TORCH_WARN("not-multiple-of-64 seq_kv is not supported below 8.9.6");
    }
    return false;
  }
  if (cudnn_version < 90000) {
    if (s_q < 64) {
      if (debug) {
        TORCH_WARN("s_q less than 64 is not supported before cudnn 9.0.0");
      }
      return false;
    }
    if (params.dropout != 0.0 && (s_q % 64 != 0 || s_k % 64 != 0)) {
      if (debug) {
        TORCH_WARN(
            "s_q not a multiple of 64 with padding/dropout is not supported with cudnn version 9.0.0");
      }
      return false;
    }
  }
  if (s_k == 1) {
    if (debug) {
      TORCH_WARN_ONCE("cudnn SDPA does not support key/value sequence length 1.");
    }
    return false;
  }
  if (s_q == 1 && params.dropout != 0.0) {
    if (debug) {
      TORCH_WARN_ONCE("cudnn SDPA does not support query sequence length 1 with dropout.");
    }
    return false;
  }
  return true;
}

bool check_cudnn_layout(sdp_params const& params, bool debug) {
  const int64_t h = params.query.size(1);
  const int64_t s_q = params.query.size(2);
  const int64_t d = params.query.size(3);
  const int64_t s_k = params.key.size(2);
  const int64_t s_v = params.value.size(2);
  // corresponds to cuDNN's "packed QKV" layout
  const bool packed_query_layout_ok = (params.query.stride(0) == s_q * 3 * h * d) &&
                                 (params.query.stride(1) == d) &&
                                 (params.query.stride(2) == 3 * h * d) &&
                                 (params.query.stride(3) == 1);
  const bool packed_key_layout_ok = (params.key.stride(0) == s_k * 3 * h * d) &&
                               (params.key.stride(1) == d) &&
                               (params.key.stride(2) == 3 * h * d) &&
                               (params.key.stride(3) == 1);
  const bool packed_value_layout_ok = (params.value.stride(0) == s_v * 3 * h * d) &&
                                 (params.value.stride(1) == d) &&
                                 (params.value.stride(2) == 3 * h * d) &&
                                 (params.value.stride(3) == 1);

  const bool packed_layout_ok = packed_query_layout_ok && packed_key_layout_ok && packed_value_layout_ok;

  const bool query_layout_ok = (params.query.stride(0) == s_q * h * d) &&
                               (params.query.stride(1) == d) &&
                               (params.query.stride(2) == h * d) &&
                               (params.query.stride(3) == 1);
  const bool key_layout_ok = (params.key.stride(0) == s_k * h * d) &&
                              (params.key.stride(1) == d) &&
                              (params.key.stride(2) == h * d) &&
                              (params.key.stride(3) == 1);
  const bool value_layout_ok = (params.value.stride(0) == s_v * h * d) &&
                               (params.value.stride(1) == d) &&
                               (params.value.stride(2) == h * d) &&
                               (params.value.stride(3) == 1);

  const bool layout_ok = query_layout_ok && key_layout_ok && value_layout_ok;

  if (!packed_value_layout_ok && !layout_ok) {
    if (debug) {
      if (!packed_layout_ok) {
        if (!packed_query_layout_ok) {
          TORCH_WARN("Query tensor was not in cuDNN-supported packed QKV layout", params.query.strides());
        }
        if (!packed_key_layout_ok) {
          TORCH_WARN("Key tensor was not in cuDNN-supported packed QKV layout", params.key.strides());
        }
        if (!packed_value_layout_ok) {
          TORCH_WARN("Value tensor was not in cuDNN-supported packed QKV layout", params.value.strides());
        }
      }
      if (!layout_ok) {
        if (!query_layout_ok) {
          TORCH_WARN("Query tensor was not in cuDNN-supported unpacked QKV layout", params.query.strides());
        }
        if (!key_layout_ok) {
          TORCH_WARN("Key tensor was not in cuDNN-supported unpacked QKV layout", params.key.strides());
        }
        if (!value_layout_ok) {
          TORCH_WARN("Value tensor was not in cuDNN-supported unpacked QKV layout", params.value.strides());
        }
      }
    }
    return false;
  }
  return true;
}

bool check_cudnn_hardware_support(sdp_params const& params, bool debug) {
  using sm80 = SMVersion<8, 0>;
  using sm121 = SMVersion<12, 1>;
  if (!at::cuda::is_available()) {
    if (debug) {
      TORCH_WARN("cuDNN SDPA requires a CUDA device, which is not available.");
    }
    return false;
  }
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (!check_sm_version<sm80, sm121>(dprops)) {
    if (debug) {
      TORCH_WARN(
          "cuDNN MHA only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

bool check_for_nested_inputs(sdp_params const& params, bool debug) {
  static const bool enable_cudnn_nested = c10::utils::check_env("TORCH_CUDNN_SDPA_NESTED_TENSOR_ENABLED") == true;
  if (has_for_nested_inputs(params) && !enable_cudnn_nested) {
    if (debug) {
      TORCH_WARN("Experimental cuDNN SDPA nested tensor support is not enabled.");
    }
    return false;
  } else if (has_for_nested_inputs(params) && (params.query.requires_grad() || params.key.requires_grad() || params.value.requires_grad())) {
    if (debug) {
      TORCH_WARN("Experimental cuDNN SDPA nested tensor support does not support backward.");
      return false;
    }
  }

  const auto dprop = at::cuda::getCurrentDeviceProperties();
  // Check that the input is nested
  if (!(dprop->major == 9 || dprop->major == 10) && has_for_nested_inputs(params)) {
    if (debug) {
      TORCH_WARN("cuDNN SDPA supports nested tensors on SM 9.0, SM 10.0.");
    }
    return false;
  }
  return true;
}

bool check_dtypes_low_precision(sdp_params const& params, bool debug) {
  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    constexpr auto sm80_dtypes =
        c10::array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
    return check_tensor_dtype(params, sm80_dtypes, debug);
  } else {
    constexpr auto default_dtypes = c10::array_of<at::ScalarType>(at::kHalf);
    return check_tensor_dtype(params, default_dtypes, debug);
  }
}

bool check_runtime_disabled_cudnn(sdp_params const& params, bool debug) {
  // We check the global context to see if user has explicitly turned of cudnn
  // sdp kernels
  if (!at::globalContext().userEnabledCuDNNSDP()) {
    if (debug) {
      TORCH_WARN("cuDNN attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

bool check_cudnn_deterministic(const sdp_params& params, bool debug) {
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (!ctx.deterministicAlgorithmsWarnOnly()) {
      if (debug) {
        TORCH_WARN("cuDNN SDPA is not deterministic.");
      }
      return false;
    }
  }
  return true;
}

} // namespace

bool can_use_cudnn_attention(const sdp_params& params, bool debug) {
#if defined(USE_ROCM) || !AT_CUDNN_ENABLED() || !defined(CUDNN_VERSION)
  if (debug) {
    TORCH_WARN("Torch was not compiled with cuDNN attention.");
  }
  return false;
#endif
#if defined(CUDNN_VERSION) && CUDNN_VERSION < 90000
  if (debug) {
    TORCH_WARN(CUDNN_VERSION, " cuDNN version too old to use cuDNN Attention (< v9.0.0)");
  }
  return false;
#endif
#if defined(CUDNN_VERSION)
  static auto cudnn_version = cudnnGetVersion();
  if (params.dropout > 0.0 && cudnn_version > 91100 && cudnn_version < 91400) {
    if (debug) {
      TORCH_WARN(CUDNN_VERSION, " cuDNN version does not support droppout in SDPA (9.11 - 9.13).");
    }
    return false;
  }
#endif
  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints =
      c10::array_of<bool (*)(sdp_params const&, bool)>(
          check_runtime_disabled_cudnn,
          check_for_nested_inputs,
          check_all_tensors_on_device,
          check_tensor_shapes,
          check_cudnn_deterministic,
          check_dtypes_low_precision,
          check_attn_mask_shape,
          check_cudnn_hardware_support
          );
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  constexpr auto dense_constraints =
      c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_nonzero_sequence_lengths_dense,
      check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>,
      check_batch_size_and_num_heads_dense<true /*enable_gqa*/, false /*requires_same_num_heads*/>,
      check_cudnn_tensor_shapes
  );

  if (has_only_dense_inputs(params)) {
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
}

bool is_flash_attention_available() {
#ifdef USE_FLASH_ATTENTION
  return true;
#else
  return false;
#endif
}

bool can_use_flash_attention(sdp_params const& params, bool debug) {
#ifndef USE_FLASH_ATTENTION
  if (debug) {
    TORCH_WARN("Torch was not compiled with flash attention.");
  }
  return false;
#else // defined(USE_FLASH_ATTENTION)
  // Define gate functions that determine if a flash kernel can be ran
  // Replace with std::to_array when we migrate to c++20
  constexpr auto general_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_flash,
      check_all_tensors_on_device,
      check_tensor_shapes,
      check_for_attn_mask,
      check_head_dim_size_flash<false /*caller_is_meff*/>,
      check_flash_attention_hardware_support,
      check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120,
      check_flash_causal_non_square_seqlens,
      check_dtypes_low_precision);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_nested,
        check_head_dim_size_flash_nested<false /*caller_is_meff*/>,
        check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  constexpr bool backend_supports_grouped_query_attention = true;
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense<backend_supports_grouped_query_attention>,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
#endif // defined(USE_FLASH_ATTENTION)
}

bool can_use_mem_efficient_attention(sdp_params const& params, bool debug) {
#ifndef USE_MEM_EFF_ATTENTION
  TORCH_WARN_ONCE(!debug, "Torch was not compiled with memory efficient attention.");
  return false;
#endif
  // Constraints specific to mem efficient attention
  constexpr auto less_than_sm80_mem_efficient_dtypes =
      c10::array_of<at::ScalarType>(at::kHalf, at::kFloat);
#ifdef USE_ROCM
  constexpr auto aotriton_mem_efficient_dtypes =
      c10::array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);
  constexpr auto ck_mem_efficient_dtypes =
      c10::array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
#else
  constexpr auto greater_than_or_equal_sm80_mem_efficient_dtypes =
      c10::array_of<at::ScalarType>(at::kHalf, at::kFloat, at::kBFloat16);
#endif

  //  Define gate functions that determine if a mem efficient kernel can be ran
  constexpr auto general_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_mem_efficient,
      check_all_tensors_on_device,
      check_mem_efficient_hardware_support,
      check_tensor_shapes,
#ifdef USE_ROCM
      check_head_dim_size_flash<true /* caller_is_meff */>
#else
      check_head_dim_size_mem_efficient
#endif
  );
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
#ifndef USE_ROCM  // ME and FA shares backend on ROCM and thus supports training
        check_requires_grad_and_nested,
#else // Meanwhile ME on ROCM share the limits of FA about head dimensions
        check_head_dim_size_flash_nested<true /* caller_is_meff */>,
#endif
        check_batch_size_nested,
        check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim=*/>,
        check_batch_size_and_num_heads_dense<false /*supports_grouped_query_attention=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

#ifdef USE_ROCM
  if (params.attn_mask.has_value()) {
    const auto q_dtype = params.query.dtype();
    const auto bias_dtype = params.attn_mask.value().dtype();
    if (bias_dtype != at::kBool && bias_dtype != q_dtype) {
      TORCH_WARN("Efficient attention on ROCM requires attn_mask be boolean, or has the same datatype as of q,k,v");
      return false;
    }
  }
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    return check_tensor_dtype(params, ck_mem_efficient_dtypes, debug);
  }
  return check_tensor_dtype(params, aotriton_mem_efficient_dtypes, debug);
#else
  auto dprop = at::cuda::getCurrentDeviceProperties();
  if (dprop->major >= 8) {
    return check_tensor_dtype(params, greater_than_or_equal_sm80_mem_efficient_dtypes, debug);
  }
#endif
  return check_tensor_dtype(params, less_than_sm80_mem_efficient_dtypes, debug);
}

SDPBackend select_sdp_backend(sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP() && !ctx.userEnabledCuDNNSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::cudnn_attention:
        if (sdp::can_use_cudnn_attention(kernel_params, print_debug)) {
              return SDPBackend::cudnn_attention;
        }
        break;
      case SDPBackend::flash_attention:
        if (sdp::can_use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (sdp::can_use_mem_efficient_attention(kernel_params, print_debug)) {
          return SDPBackend::efficient_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      case SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP()) {
          TORCH_CHECK(false, "Invalid backend");
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
  sdp::can_use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  sdp::can_use_flash_attention(kernel_params, print_debug);
  TORCH_WARN("cuDNN attention kernel not used because:");
  sdp::can_use_cudnn_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return SDPBackend::error;
}

bool check_for_seq_len_1_nested_tensor(sdp_params const& params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

} // namespace sdp
