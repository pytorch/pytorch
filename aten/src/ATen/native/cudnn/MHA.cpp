#include <limits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/cudnn-wrapper.h>
#if defined(__has_include) && __has_include(<cudnn_frontend_version.h>)
#include <cudnn_frontend_version.h>
#endif
#endif

namespace at::native {

// Check the pointer and stride alignment cuDNN requires for varlen tensors.
bool has_aligned_varlen_layout(const Tensor& tensor) {
  constexpr int64_t alignment_bytes = 16;
  if (!tensor.numel()) {
    return true;
  }
  if (tensor.dim() == 0 || tensor.stride(-1) != 1 ||
      reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % alignment_bytes !=
          0) {
    return false;
  }
  const int64_t alignment = alignment_bytes / tensor.element_size();
  for (int64_t dim = 0; dim < tensor.dim() - 1; ++dim) {
    if (tensor.size(dim) > 1 &&
        (tensor.stride(dim) <= 0 || tensor.stride(dim) % alignment != 0)) {
      return false;
    }
  }
  return true;
}

} // namespace at::native

#if defined(USE_ROCM) || !AT_CUDNN_ENABLED() || !defined(CUDNN_VERSION) || \
    (defined(CUDNN_VERSION) && CUDNN_VERSION < 8900) ||                    \
    !defined(CUDNN_FRONTEND_VERSION) ||                                    \
    (defined(CUDNN_FRONTEND_VERSION) && CUDNN_FRONTEND_VERSION < 10100)
namespace at {
namespace native {

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  TORCH_CHECK(
      false, "PyTorch was not compiled with cuDNN Flash Attention enabled!");
}

void run_cudnn_SDP_fprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const std::optional<Tensor>& seqused_k,
    const std::optional<Tensor>& page_table,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  TORCH_CHECK(
      false, "PyTorch was not compiled with cuDNN Flash Attention enabled!");
}

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  TORCH_CHECK(
      false, "PyTorch was not compiled with cuDNN Flash Attention enabled!");
}

void run_cudnn_SDP_bprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,

    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  TORCH_CHECK(
      false, "PyTorch was not compiled with cuDNN Flash Attention enabled!");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED && CUDNN_VERSION >= 8900 && CUDNN_FRONTEND_VERSION >=
      // 10100
#include <cudnn_frontend.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/cudnn/MHA.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils.h>

#include <ATen/cuda/Exceptions.h>
#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/TypeCast.h>
#include <cudnn.h>

#include <cstdint>
#include <iostream>

#if CUDNN_FRONTEND_VERSION >= 12500 && CUDNN_VERSION >= 92400
#define AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS 1
#else
#define AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS 0
#endif

namespace at::native {

namespace fe = cudnn_frontend;

constexpr uint8_t MAX_MHA_DIM = 4;

static void check_cudnn_sdpa_execution(fe::error_t err) {
  if (C10_LIKELY(err.is_good())) {
    return;
  }

  const auto error_message = err.get_message();
  const bool is_cuda_oom =
      error_message.find("err 2 != CUDA_SUCCESS") != std::string::npos ||
      error_message.find("CUDA_ERROR_OUT_OF_MEMORY") != std::string::npos ||
      error_message.find("cudaErrorMemoryAllocation") != std::string::npos;
  TORCH_CHECK(
      false,
      "cuDNN SDPA execution failed with error code ",
      err.get_code(),
      ": ",
      error_message,
      is_cuda_oom
          ? "\nCUDA ran out of memory outside PyTorch's allocator. If this "
            "workload uses many dynamic shapes, cuDNN may need additional "
            "device memory to JIT-compile shape-specialized kernels. Consider "
            "calling torch.cuda.memory.set_per_process_memory_fraction(fraction) "
            "early in the process to leave memory available for cuDNN."
          : "");
}

// See #193893 and #194927 for reasoning
// TODO: remove this and all associated calls/imports when fixed
void check_cudnn_sdpa_decode(int64_t s_q) {
  TORCH_CHECK(
      s_q != 1 || !sdp::is_cudnn_attention_decode_disabled(),
      "cuDNN SDPA decode is disabled for cuDNN versions 9.19-9.25.0 (except 9.24.1) on SM 10.x and 11.x.");
}

// Whether we will use ragged offsets in the dense (non-nested) path
// to avoid recompilation
bool use_ragged_in_dense(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& o,
    bool has_bias) {
  static bool flag =
      c10::utils::check_env("TORCH_CUDNN_SDPA_AVOID_RECOMPILE") == true;
  if (!flag) {
    return flag;
  }
  TORCH_WARN_ONCE(
      "TORCH_CUDNN_SDPA_AVOID_RECOMPILE=1 is currently experimental. "
      "Please report any issues to https://github.com/pytorch/pytorch/issues.");
  if (has_bias) {
    TORCH_WARN_ONCE(
        "TORCH_CUDNN_SDPA_AVOID_RECOMPILE=1 only works without bias."
        "Consider using the is_causal hint instead of bias for causal masking."
        "Falling back to regular dense case, which may trigger excessive recompilation.");
    return !has_bias;
  }
  bool all_bshd = q.dim() == 4 && q.transpose(1, 2).is_contiguous() &&
      k.dim() == 4 && k.transpose(1, 2).is_contiguous() && v.dim() == 4 &&
      v.transpose(1, 2).is_contiguous() && o.dim() == 4 &&
      o.transpose(1, 2).is_contiguous();
  if (!all_bshd) {
    TORCH_WARN_ONCE(
        "TORCH_CUDNN_SDPA_AVOID_RECOMPILE=1 only works with Q, K, V, and output in BSHD memory layout,"
        "e.g., Q, K, V must be allocated with torch.randn((B, S, H, D).transpose(1, 2)."
        "Falling back to regular dense case, which may trigger excessive recompilation.");
  }
  return all_bshd;
}

int roundup_power2(int dim) {
  if (!dim) {
    return 1;
  }
  dim--;
  dim |= dim >> 1;
  dim |= dim >> 2;
  dim |= dim >> 4;
  dim |= dim >> 8;
  dim |= dim >> 16;
  dim++;
  return dim;
}

// scaled_dot_product_attention accepts an attn_mask whose dtype differs from
// query (validate_sdpa_input allows float or query.dtype; bool masks are
// converted to query.dtype before we get here), so the bias cannot inherit the
// graph-wide io data type.
static fe::DataType_t bias_data_type(const Tensor& attn_bias) {
  switch (attn_bias.scalar_type()) {
    case kHalf:
      return fe::DataType_t::HALF;
    case kBFloat16:
      return fe::DataType_t::BFLOAT16;
    case kFloat:
      return fe::DataType_t::FLOAT;
    default:
      TORCH_CHECK(
          false,
          "cuDNN SDPA got attn_bias of unsupported dtype ",
          attn_bias.scalar_type(),
          ", expected one of float, half, bfloat16.");
  }
}

enum class SequenceLengthMode : uint8_t {
  PER_SEQUENCE = 0,
  CUMULATIVE = 1,
};

// Which causal diagonal cuDNN masks against. TOP_LEFT is the dense SDPA
// convention; BOTTOM_RIGHT aligns the diagonal to the last query and key of
// each sequence, which is the FlashAttention varlen/KV-cache convention.
enum class CausalMask : uint8_t {
  NONE = 0,
  TOP_LEFT = 1,
  BOTTOM_RIGHT = 2,
};

struct MHAParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  // the mask dtype is not implied by dataType, and it selects a different graph
  fe::DataType_t biasDataType;
  std::array<int, MAX_MHA_DIM> q_dim;
  std::array<int, MAX_MHA_DIM> k_dim;
  std::array<int, MAX_MHA_DIM> v_dim;
  std::array<int, MAX_MHA_DIM> q_stride;
  std::array<int, MAX_MHA_DIM> k_stride;
  std::array<int, MAX_MHA_DIM> v_stride;
  std::array<int, MAX_MHA_DIM> bias_dim;
  std::array<int, MAX_MHA_DIM> bias_stride;
  // Block tables have shape (batch_size, max_pages_per_sequence).
  std::array<int64_t, 2> page_table_dim;
  std::array<int64_t, 2> page_table_stride;
  std::array<int64_t, MAX_MHA_DIM> o_dim;
  std::array<int64_t, MAX_MHA_DIM> o_stride;
  std::array<int64_t, MAX_MHA_DIM> do_dim;
  std::array<int64_t, MAX_MHA_DIM> do_stride;
  std::array<int64_t, MAX_MHA_DIM> softmaxstats_dim;
  std::array<int64_t, MAX_MHA_DIM> softmaxstats_stride;
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d_qk;
  int64_t d_v;
  double dropout_probability;
  CausalMask causal_mask;
  bool return_softmaxstats;
  // might be redundant if we take 0 dim/stride
  // as signaling no-bias
  bool has_attn_bias;
  bool use_ragged;
  bool is_paged;
  bool is_nested;
  SequenceLengthMode sequence_length_mode;
};

namespace {

template <typename T>
concept HasSetAlignment = requires(T& attributes, int64_t alignment) {
  attributes.set_alignment(alignment);
};

template <typename T>
void setAlignmentIfSupported(T& attributes, int64_t alignment) {
  if constexpr (HasSetAlignment<T>) {
    attributes.set_alignment(alignment);
  }
}

constexpr int64_t kInt32Alignment = alignof(int32_t);
// Frontend versions without set_alignment hardcode 16-byte descriptors.
constexpr int64_t kLegacyTensorAlignment = 16;
constexpr int64_t kRequiredInt32Alignment =
    HasSetAlignment<fe::graph::Tensor_attributes> ? kInt32Alignment
                                                  : kLegacyTensorAlignment;

void checkInt32Alignment(const Tensor& tensor, const char* name) {
  const auto address = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
  TORCH_CHECK(
      address % kRequiredInt32Alignment == 0,
      name,
      " data pointer must be aligned to ",
      kRequiredInt32Alignment,
      " bytes for the selected cuDNN Frontend");
}

// Record an auxiliary tensor layout in the zero-initialized cache key.
void setMHAParamLayout(
    const Tensor& tensor,
    std::array<int64_t, MAX_MHA_DIM>& dim,
    std::array<int64_t, MAX_MHA_DIM>& stride) {
  if (!tensor.defined()) {
    return;
  }
  TORCH_INTERNAL_ASSERT(tensor.dim() <= MAX_MHA_DIM);
  std::copy(tensor.sizes().begin(), tensor.sizes().end(), dim.begin());
  std::copy(tensor.strides().begin(), tensor.strides().end(), stride.begin());
}

} // namespace

void setMHAParams(
    MHAParams& params,
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    double dropout_probability,
    CausalMask causal_mask,
    bool return_softmaxstats,
    bool is_nested,
    const std::optional<Tensor>& page_table,
    SequenceLengthMode sequence_length_mode) {
  memset(&params, 0, sizeof(MHAParams));
  params.device_id = at::cuda::current_device();
  params.dataType = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    params.dataType = fe::DataType_t::BFLOAT16;
  }
  params.b = b;
  params.h = h;
  params.d_qk = d_qk;
  params.d_v = d_v;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.dropout_probability = dropout_probability;
  params.causal_mask = causal_mask;
  params.return_softmaxstats = return_softmaxstats;
  params.has_attn_bias = attn_bias.has_value();
  params.is_paged = page_table.has_value();
  params.is_nested = is_nested;
  params.sequence_length_mode = sequence_length_mode;
  // Paged K/V remain 4D page pools in the nested path.
  const uint8_t q_rank = (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested);
  const uint8_t kv_rank = params.is_paged ? MAX_MHA_DIM : q_rank;
  TORCH_INTERNAL_ASSERT(
      q.sizes().size() == q_rank,
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      q.strides().size() == q_rank,
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.sizes().size() == kv_rank,
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.strides().size() == kv_rank,
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.sizes().size() == kv_rank,
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.strides().size() == kv_rank,
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim.begin());
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride.begin());
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim.begin());
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride.begin());
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim.begin());
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride.begin());
  setMHAParamLayout(o, params.o_dim, params.o_stride);
  setMHAParamLayout(dO, params.do_dim, params.do_stride);
  setMHAParamLayout(
      softmaxstats, params.softmaxstats_dim, params.softmaxstats_stride);
  bool use_ragged = use_ragged_in_dense(q, k, v, o, params.has_attn_bias);
  params.use_ragged = use_ragged;
  if (use_ragged) {
    // ignore B - stride in BSHD (THD) avoid-recompile
    params.q_stride[0] = INT_MAX;
    params.k_stride[0] = INT_MAX;
    params.v_stride[0] = INT_MAX;
    // fix seqlen to rounded value
    params.s_q = roundup_power2(params.s_q);
    params.s_kv = roundup_power2(params.s_kv);
    params.q_dim[2] = roundup_power2(params.q_dim[2]);
    params.k_dim[2] = roundup_power2(params.k_dim[2]);
    params.v_dim[2] = roundup_power2(params.v_dim[2]);
  }
  if (params.is_paged) {
    const auto& table = page_table.value();
    params.page_table_dim[0] = table.size(0);
    params.page_table_dim[1] = table.size(1);
    params.page_table_stride[0] = table.stride(0);
    params.page_table_stride[1] = table.stride(1);
  }
  // uninit is OK as the struct is memset 0'd
  if (params.has_attn_bias) {
    params.biasDataType = bias_data_type(attn_bias.value());
    std::copy(
        attn_bias.value().sizes().begin(),
        attn_bias.value().sizes().end(),
        params.bias_dim.begin());
    std::copy(
        attn_bias.value().strides().begin(),
        attn_bias.value().strides().end(),
        params.bias_stride.begin());
  }
}

struct MHACacheKeyWrapper : ParamsWrapper<MHAParams> {
  MHACacheKeyWrapper(
      int64_t b,
      int64_t h,
      int64_t s_q,
      int64_t s_kv,
      int64_t d_qk,
      int64_t d_v,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      const std::optional<Tensor>& attn_bias,
      const Tensor& o,
      const Tensor& dO,
      const Tensor& softmaxstats,
      double dropout_probability,
      CausalMask causal_mask,
      bool return_softmaxstats,
      bool is_nested,
      const std::optional<Tensor>& page_table = std::nullopt,
      SequenceLengthMode sequence_length_mode =
          SequenceLengthMode::PER_SEQUENCE) {
    setMHAParams(
        this->pod,
        b,
        h,
        s_q,
        s_kv,
        d_qk,
        d_v,
        q,
        k,
        v,
        attn_bias,
        o,
        dO,
        softmaxstats,
        dropout_probability,
        causal_mask,
        return_softmaxstats,
        is_nested,
        page_table,
        sequence_length_mode);
  }
};

struct MHAGraphCache {
  using KeyType = MHACacheKeyWrapper;
  using ValueType = std::unique_ptr<fe::graph::Graph>;
  using MapType =
      std::unordered_map<KeyType, ValueType, ParamsWrapperHash<KeyType>>;
  using iterator = typename MapType::iterator;
  using const_iterator = typename MapType::const_iterator;

  MapType engine_cache;
  int count = 0;
  int hits = 0;

  // no mutexes here as caches are now thread local for v8, can also return a
  // pointer to the Execution Plan if we know it will not be invalidated by
  // another thread
  iterator find(const KeyType& key) {
    static bool flag =
        c10::utils::check_env("TORCH_CUDNN_SDPA_CACHE_DEBUG") == true;
    if (flag && count) {
      TORCH_WARN(
          "SDPA Cache Called ",
          count,
          " times. Hit rate: ",
          100 * hits / count,
          "%");
    }
    count++;
    auto it = engine_cache.find(key);
    if (it != engine_cache.end()) {
      hits++;
    }
    return it;
  }

  const_iterator end() const {
    return engine_cache.end();
  }

  template <typename... Args>
  std::pair<iterator, bool> try_emplace(const KeyType& key, Args&&... args) {
    return engine_cache.try_emplace(key, std::forward<Args>(args)...);
  }
};

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to
// be thread safe across all engines see Limitations in
// https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html
// We also leak the caches to workaround potential teardown race issues.

MHAGraphCache& getMHAGraphCache_() {
  thread_local MHAGraphCache* instance{new MHAGraphCache()};
  return *instance;
}

MHAGraphCache& getMHAGraphBackwardCache_() {
  thread_local MHAGraphCache* instance{new MHAGraphCache()};
  return *instance;
}

namespace {

enum UIDS {
  Q,
  K,
  V,
  O,
  BIAS,
  SCALE,
  SEED,
  OFFSET,
  LSE,
  DO,
  DQ,
  DK,
  DV,
  SEQ_LEN_Q,
  SEQ_LEN_KV,
  CU_SEQ_LEN_Q,
  CU_SEQ_LEN_KV,
  RAG_Q_OFF,
  RAG_K_OFF,
  RAG_V_OFF,
  RAG_DQ_OFF,
  RAG_DK_OFF,
  RAG_DV_OFF,
  RAG_O_OFF,
  RAG_DO_OFF,
  RAG_LSE_OFF,
  PAGE_TABLE_K,
  PAGE_TABLE_V
};

// cuDNN describes packed THD storage as nominal BHSD. Ragged offsets provide
// each sequence base, so the unused batch stride is an INT_MAX placeholder.
std::vector<int64_t> thd_to_bhsd_strides(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.dim() == 3);
  return {INT_MAX, tensor.stride(1), tensor.stride(0), tensor.stride(2)};
}

// Ragged offsets are declared int32 to cuDNN, so the largest offset
// (packed extent * token stride) must not wrap.
void check_ragged_offset_capacity(const Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.size(-3) * tensor.stride(-3) <= std::numeric_limits<int>::max(),
      "cuDNN varlen attention requires the packed extent of ",
      name,
      " times its token stride to fit in int32, got ",
      tensor.size(-3) * tensor.stride(-3));
}

// A ragged offset is cum_seqlen * token_stride, so equal token strides can
// share one offset tensor instead of launching another multiply.
Tensor ragged_offset(
    const Tensor& cum_seqlen,
    int64_t token_stride,
    const Tensor& reuse,
    int64_t reuse_token_stride) {
  return token_stride == reuse_token_stride ? reuse
                                            : cum_seqlen.mul(token_stride);
}

// analogous to the same function in Descriptors.h for cuDNN Convolutions...
auto fixSizeOneDimStrideSDPA(
    const IntArrayRef sizes,
    std::vector<int64_t> strides) {
  int dims = sizes.size();
  for (int d = 0; d < dims; d++) {
    int64_t curr_stride = strides[d];
    if (sizes[d] == 1 && !curr_stride) {
      curr_stride = 1;
      for (int d2 = d + 1; d2 < dims; d2++) {
        curr_stride *= strides[d2];
      }
      strides[d] = curr_stride;
    }
  }
  return strides;
}

} // namespace

std::unique_ptr<fe::graph::Graph> build_graph(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset,
    cudnnHandle_t& handle) {
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }
  auto mha_graph = std::make_unique<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(SCALE)
                            .set_name("Attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));
  auto scaled_dot_product_flash_attention_options =
      fe::graph::SDPA_attributes()
          .set_name("CUDNN_SDPA")
#if CUDNN_FRONTEND_VERSION <= 11200
          .set_is_inference(!return_softmaxstats)
#else
          .set_generate_stats(return_softmaxstats)
#endif
          .set_causal_mask(is_causal)
          .set_attn_scale(attn_scale);
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    auto SEQ_LEN_Q_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(SEQ_LEN_Q)
                              .set_name("Seq_q")
                              .set_dim({b, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto SEQ_LEN_KV_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(SEQ_LEN_KV)
                              .set_name("Seq_kv")
                              .set_dim({b, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    scaled_dot_product_flash_attention_options.set_seq_len_q(SEQ_LEN_Q_)
        .set_seq_len_kv(SEQ_LEN_KV_)
        .set_padding_mask(true);
  }
  if (dropout_probability != 0.0f) {
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_uid(SEED)
                                      .set_name("Seed")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(
                                          dropoutseed.dtype() == kInt
                                              ? fe::DataType_t::INT32
                                              : fe::DataType_t::INT64));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_uid(OFFSET)
                                        .set_name("Offset")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(
                                            dropoutoffset.dtype() == kInt
                                                ? fe::DataType_t::INT32
                                                : fe::DataType_t::INT64));
    scaled_dot_product_flash_attention_options.set_dropout(
        dropout_probability, seed, offset);
  }
  auto Q_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(Q).set_name("Q"));
  auto K_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(K).set_name("K"));
  auto V_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(V).set_name("V"));
  if (attn_bias.has_value()) {
    scaled_dot_product_flash_attention_options.set_bias(mha_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_uid(BIAS)
            .set_name("bias")
            .set_dim(attn_bias.value().sizes().vec())
            .set_stride(attn_bias.value().strides().vec())
            .set_data_type(bias_data_type(attn_bias.value()))));
  }

  auto [O_, Stats] =
      mha_graph->sdpa(Q_, K_, V_, scaled_dot_product_flash_attention_options);
  O_->set_uid(O).set_output(true);
  if (Stats) {
    Stats->set_uid(LSE)
        .set_output(true)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_stride(softmaxstats.strides().vec());
  }
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    auto RAG_Q_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_Q_OFF)
                              .set_name("cum_seq_q")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_K_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_K_OFF)
                              .set_name("cum_seq_k")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_V_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_V_OFF)
                              .set_name("cum_seq_v")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_O_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_O_OFF)
                              .set_name("cum_seq_o")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_STATS_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_LSE_OFF)
                              .set_name("cum_seq_stats")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    O_->set_ragged_offset(RAG_O_OFF_);
    Q_->set_ragged_offset(RAG_Q_OFF_);
    K_->set_ragged_offset(RAG_K_OFF_);
    V_->set_ragged_offset(RAG_V_OFF_);
    auto qsizevec = q.sizes().vec();
    auto ksizevec = k.sizes().vec();
    auto vsizevec = v.sizes().vec();
    auto osizevec = o.sizes().vec();
    qsizevec[2] = roundup_power2(qsizevec[2]);
    ksizevec[2] = roundup_power2(ksizevec[2]);
    vsizevec[2] = roundup_power2(vsizevec[2]);
    osizevec[2] = roundup_power2(osizevec[2]);
    // we checked for BSHD contig., set fake strides as cuDNN will complain
    // if e.g., a ragged dim is smaller than a non-ragged one:
    // consider HBSD tensor where H is 1
    Q_->set_dim(qsizevec).set_stride(
        {INT_MAX, qsizevec[3], qsizevec[1] * qsizevec[3], 1});
    K_->set_dim(ksizevec).set_stride(
        {INT_MAX, ksizevec[3], ksizevec[1] * ksizevec[3], 1});
    V_->set_dim(vsizevec).set_stride(
        {INT_MAX, vsizevec[3], vsizevec[1] * vsizevec[3], 1});
    O_->set_dim(osizevec).set_stride(
        {INT_MAX, osizevec[3], osizevec[1] * osizevec[3], 1});
    if (Stats) {
      Stats->set_ragged_offset(RAG_STATS_OFF_);
      auto statssizevec = softmaxstats.sizes().vec();
      statssizevec[2] = roundup_power2(statssizevec[2]);
      Stats->set_dim(statssizevec);
    }
  } else {
    Q_->set_dim(q.sizes().vec())
        .set_stride(fixSizeOneDimStrideSDPA(q.sizes(), q.strides().vec()));
    K_->set_dim(k.sizes().vec())
        .set_stride(fixSizeOneDimStrideSDPA(k.sizes(), k.strides().vec()));
    V_->set_dim(v.sizes().vec())
        .set_stride(fixSizeOneDimStrideSDPA(v.sizes(), v.strides().vec()));
    O_->set_dim(o.sizes().vec())
        .set_stride(fixSizeOneDimStrideSDPA(o.sizes(), o.strides().vec()));
    if (Stats) {
      Stats->set_dim(softmaxstats.sizes().vec());
    }
  }

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));

  return mha_graph;
}

std::unique_ptr<fe::graph::Graph> build_graph_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    CausalMask causal_mask,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const std::optional<Tensor>& page_table,
    SequenceLengthMode sequence_length_mode,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset,
    cudnnHandle_t& handle) {
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }
  const bool is_paged = page_table.has_value();
  auto mha_graph = std::make_unique<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(SCALE)
                            .set_name("Attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));
  auto index_tensor = [&](UIDS uid, const char* name, int64_t size) {
    auto attributes = fe::graph::Tensor_attributes();
    attributes.set_uid(uid)
        .set_name(name)
        .set_dim({size, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32);
    setAlignmentIfSupported(attributes, kInt32Alignment);
    return mha_graph->tensor(attributes);
  };

  auto scaled_dot_product_flash_attention_options =
      fe::graph::SDPA_attributes()
          .set_name("CUDNN_SDPA_NESTEDTENSOR")
#if CUDNN_FRONTEND_VERSION <= 11200
          .set_is_inference(!return_softmaxstats)
#else
          .set_generate_stats(return_softmaxstats)
#endif
          .set_causal_mask(causal_mask == CausalMask::TOP_LEFT)
          .set_causal_mask_bottom_right(causal_mask == CausalMask::BOTTOM_RIGHT)
          .set_attn_scale(attn_scale)
          .set_padding_mask(true);
#if AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS
  if (sequence_length_mode == SequenceLengthMode::CUMULATIVE) {
    auto CU_SEQ_LEN_Q_ = index_tensor(CU_SEQ_LEN_Q, "Cu_seq_q", b + 1);
    auto CU_SEQ_LEN_KV_ = index_tensor(CU_SEQ_LEN_KV, "Cu_seq_kv", b + 1);
    scaled_dot_product_flash_attention_options.set_cu_seq_len_q(CU_SEQ_LEN_Q_)
        .set_cu_seq_len_kv(CU_SEQ_LEN_KV_)
        .set_implementation(fe::AttentionImplementation_t::UNIFIED);
  }
#endif
  if (sequence_length_mode == SequenceLengthMode::PER_SEQUENCE) {
    scaled_dot_product_flash_attention_options
        .set_seq_len_q(index_tensor(SEQ_LEN_Q, "Seq_q", b))
        .set_seq_len_kv(index_tensor(SEQ_LEN_KV, "Seq_kv", b));
  }
  if (dropout_probability != 0.0f) {
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_uid(SEED)
                                      .set_name("Seed")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(
                                          dropoutseed.dtype() == kInt
                                              ? fe::DataType_t::INT32
                                              : fe::DataType_t::INT64));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_uid(OFFSET)
                                        .set_name("Offset")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(
                                            dropoutoffset.dtype() == kInt
                                                ? fe::DataType_t::INT32
                                                : fe::DataType_t::INT64));
    scaled_dot_product_flash_attention_options.set_dropout(
        dropout_probability, seed, offset);
  }
  auto Q_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(Q)
                                  .set_name("Q")
                                  .set_dim({b, h_q, s_q, d_qk})
                                  .set_stride(thd_to_bhsd_strides(q)));
  std::shared_ptr<fe::graph::Tensor_attributes> K_, V_;
  if (is_paged) {
    // Reinterpret (pages, page_size, H, D) as cuDNN's (pages, H, page_size, D).
    K_ = mha_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_uid(K)
            .set_name("container_K")
            .set_dim({k.size(0), h_k, k.size(1), d_qk})
            .set_stride({k.stride(0), k.stride(2), k.stride(1), k.stride(3)}));
    V_ = mha_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_uid(V)
            .set_name("container_V")
            .set_dim({v.size(0), h_v, v.size(1), d_v})
            .set_stride({v.stride(0), v.stride(2), v.stride(1), v.stride(3)}));
    const auto& table = page_table.value();
    const int64_t table_size = table.size(1);
    const int64_t max_seq_len_kv = table_size * k.size(1);
    auto page_table_tensor = [&](UIDS uid, const char* name) {
      auto attributes = fe::graph::Tensor_attributes();
      attributes.set_uid(uid)
          .set_name(name)
          .set_dim({b, 1, table_size, 1})
          .set_stride({table.stride(0), 1, table.stride(1), 1})
          .set_data_type(fe::DataType_t::INT32);
      setAlignmentIfSupported(attributes, kInt32Alignment);
      return mha_graph->tensor(attributes);
    };
    // K and V share the same block table.
    scaled_dot_product_flash_attention_options
        .set_paged_attention_k_table(
            page_table_tensor(PAGE_TABLE_K, "page_table_k"))
        .set_paged_attention_v_table(
            page_table_tensor(PAGE_TABLE_V, "page_table_v"))
        // cuDNN derives its maximum KV length from the page-table width.
        .set_paged_attention_max_seq_len_kv(c10::checked_convert<int>(
            max_seq_len_kv, "paged attention maximum KV sequence length"));
  } else {
    K_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                               .set_uid(K)
                               .set_name("K")
                               .set_dim({b, h_k, s_kv, d_qk})
                               .set_stride(thd_to_bhsd_strides(k)));
    V_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                               .set_uid(V)
                               .set_name("V")
                               .set_dim({b, h_v, s_kv, d_v})
                               .set_stride(thd_to_bhsd_strides(v)));
  }
  if (attn_bias.has_value()) {
    TORCH_CHECK(
        false,
        "attn_bias not yet supported with cuDNN Attention and NestedTensor");
    scaled_dot_product_flash_attention_options.set_bias(
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec())));
  }
  auto ragged_offset_tensor = [&](UIDS uid, const char* name) {
    auto attributes = fe::graph::Tensor_attributes();
    attributes.set_uid(uid)
        .set_name(name)
        .set_dim({b + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32);
    setAlignmentIfSupported(attributes, kInt32Alignment);
    return mha_graph->tensor(attributes);
  };
  auto RAG_Q_OFF_ = ragged_offset_tensor(RAG_Q_OFF, "cum_seq_q");
  auto RAG_O_OFF_ = ragged_offset_tensor(RAG_O_OFF, "cum_seq_o");
  Q_->set_ragged_offset(RAG_Q_OFF_);
  if (!is_paged) {
    K_->set_ragged_offset(ragged_offset_tensor(RAG_K_OFF, "cum_seq_k"));
    V_->set_ragged_offset(ragged_offset_tensor(RAG_V_OFF, "cum_seq_v"));
  }
#if AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS
  if (sequence_length_mode == SequenceLengthMode::CUMULATIVE) {
    TORCH_INTERNAL_ASSERT(!is_paged);
    Q_->set_ragged_offset_multiplier(q.stride(-3));
    K_->set_ragged_offset_multiplier(k.stride(-3));
    V_->set_ragged_offset_multiplier(v.stride(-3));
  }
#endif
  auto [O_, Stats] =
      mha_graph->sdpa(Q_, K_, V_, scaled_dot_product_flash_attention_options);
  O_->set_output(true)
      .set_uid(O)
      .set_dim({b, h_q, s_q, d_v})
      .set_stride(thd_to_bhsd_strides(o));
  O_->set_ragged_offset(RAG_O_OFF_);
#if AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS
  if (sequence_length_mode == SequenceLengthMode::CUMULATIVE) {
    O_->set_ragged_offset_multiplier(o.stride(-3));
  }
#endif
  if (Stats) {
    Stats->set_output(true)
        .set_uid(LSE)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({b, h_q, s_q, 1})
        .set_stride(thd_to_bhsd_strides(softmaxstats));
    Stats->set_ragged_offset(
        ragged_offset_tensor(RAG_LSE_OFF, "cum_seq_stats"));
  }
  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));
  return mha_graph;
}

std::unique_ptr<fe::graph::Graph> build_graph_backward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset,
    cudnnHandle_t& handle) {
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }
  auto mha_graph = std::make_unique<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(SCALE)
                            .set_name("Attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));
  auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                   .set_name("CUDNN_SDPA_BACKWARD")
                                   .set_causal_mask(is_causal)
                                   .set_attn_scale(attn_scale);
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    auto SEQ_LEN_Q_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(SEQ_LEN_Q)
                              .set_name("Seq_q")
                              .set_dim({b, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto SEQ_LEN_KV_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(SEQ_LEN_KV)
                              .set_name("Seq_kv")
                              .set_dim({b, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    sdpa_backward_options.set_seq_len_q(SEQ_LEN_Q_)
        .set_seq_len_kv(SEQ_LEN_KV_)
        .set_padding_mask(true);
  }

  auto Q_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(Q).set_name("Q"));
  auto K_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(K).set_name("K"));
  auto V_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(V).set_name("V"));
  if (attn_bias.has_value()) {
    sdpa_backward_options.set_bias(mha_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_uid(BIAS)
            .set_name("bias")
            .set_dim(attn_bias.value().sizes().vec())
            .set_stride(attn_bias.value().strides().vec())
            .set_data_type(bias_data_type(attn_bias.value()))));
  }
  if (dropout_probability != 0.0f) {
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_uid(SEED)
                                      .set_name("Seed")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(
                                          dropoutseed.dtype() == kInt
                                              ? fe::DataType_t::INT32
                                              : fe::DataType_t::INT64));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_uid(OFFSET)
                                        .set_name("Offset")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(
                                            dropoutoffset.dtype() == kInt
                                                ? fe::DataType_t::INT32
                                                : fe::DataType_t::INT64));
    sdpa_backward_options.set_dropout(dropout_probability, seed, offset);
  }
  auto O_ = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(O).set_name("O"));
  auto Stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_uid(LSE)
                                     .set_name("Stats")
                                     .set_stride(softmaxstats.strides().vec())
                                     .set_data_type(fe::DataType_t::FLOAT));
  auto Do = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_uid(DO).set_name("DO"));
  auto [Dq, Dk, Dv] = mha_graph->sdpa_backward(
      Q_, K_, V_, O_, Do, Stats, sdpa_backward_options);
  Dq->set_uid(DQ).set_output(true);
  Dk->set_uid(DK).set_output(true);
  Dv->set_uid(DV).set_output(true);
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    auto RAG_Q_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_Q_OFF)
                              .set_name("cum_seq_q")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_K_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_K_OFF)
                              .set_name("cum_seq_k")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_V_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_V_OFF)
                              .set_name("cum_seq_v")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_O_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_O_OFF)
                              .set_name("cum_seq_o")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    auto RAG_STATS_OFF_ =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_LSE_OFF)
                              .set_name("cum_seq_stats")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    O_->set_ragged_offset(RAG_O_OFF_);
    Q_->set_ragged_offset(RAG_Q_OFF_);
    K_->set_ragged_offset(RAG_K_OFF_);
    V_->set_ragged_offset(RAG_V_OFF_);
    Dq->set_ragged_offset(RAG_Q_OFF_);
    Dk->set_ragged_offset(RAG_K_OFF_);
    Dv->set_ragged_offset(RAG_V_OFF_);
    Do->set_ragged_offset(RAG_O_OFF_);
    auto qsizevec = q.sizes().vec();
    auto ksizevec = k.sizes().vec();
    auto vsizevec = v.sizes().vec();
    auto osizevec = o.sizes().vec();
    qsizevec[2] = roundup_power2(qsizevec[2]);
    ksizevec[2] = roundup_power2(ksizevec[2]);
    vsizevec[2] = roundup_power2(vsizevec[2]);
    osizevec[2] = roundup_power2(osizevec[2]);
    // see corresponding section in the forward about the hardcoding
    // of strides here
    Q_->set_dim(qsizevec).set_stride(
        {INT_MAX, qsizevec[3], qsizevec[1] * qsizevec[3], 1});
    K_->set_dim(ksizevec).set_stride(
        {INT_MAX, ksizevec[3], ksizevec[1] * ksizevec[3], 1});
    V_->set_dim(vsizevec).set_stride(
        {INT_MAX, vsizevec[3], vsizevec[1] * vsizevec[3], 1});
    O_->set_dim(osizevec).set_stride(
        {INT_MAX, osizevec[3], osizevec[1] * osizevec[3], 1});
    // should be identical to their non-d counterparts
    Dq->set_dim(qsizevec).set_stride(
        {INT_MAX, qsizevec[3], qsizevec[1] * qsizevec[3], 1});
    Dk->set_dim(ksizevec).set_stride(
        {INT_MAX, ksizevec[3], ksizevec[1] * ksizevec[3], 1});
    Dv->set_dim(vsizevec).set_stride(
        {INT_MAX, vsizevec[3], vsizevec[1] * vsizevec[3], 1});
    Do->set_dim(osizevec).set_stride(
        {INT_MAX, osizevec[3], osizevec[1] * osizevec[3], 1});

    Stats->set_ragged_offset(RAG_STATS_OFF_);
    auto statssizevec = softmaxstats.sizes().vec();
    statssizevec[2] = roundup_power2(statssizevec[2]);
    Stats->set_dim(statssizevec);
  } else {
    O_->set_dim(o.sizes().vec()).set_stride(o.strides().vec());
    Q_->set_dim(q.sizes().vec()).set_stride(q.strides().vec());
    K_->set_dim(k.sizes().vec()).set_stride(k.strides().vec());
    V_->set_dim(v.sizes().vec()).set_stride(v.strides().vec());
    Dq->set_dim(dQ.sizes().vec()).set_stride(dQ.strides().vec());
    Dk->set_dim(dK.sizes().vec()).set_stride(dK.strides().vec());
    Dv->set_dim(dV.sizes().vec()).set_stride(dV.strides().vec());
    Do->set_dim(dO.sizes().vec()).set_stride(dO.strides().vec());
    Stats->set_dim(softmaxstats.sizes().vec());
  }

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));
  return mha_graph;
}

std::unique_ptr<fe::graph::Graph> build_graph_backward_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset,
    cudnnHandle_t& handle) {
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }
  auto mha_graph = std::make_unique<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(SCALE)
                            .set_name("Attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

  auto SEQ_LEN_Q_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(SEQ_LEN_Q)
                            .set_name("Seq_q")
                            .set_dim({b, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto SEQ_LEN_KV_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(SEQ_LEN_KV)
                            .set_name("Seq_kv")
                            .set_dim({b, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                   .set_name("CUDNN_SDPA_NESTEDTENSOR_BACKWARD")
                                   .set_causal_mask(is_causal)
                                   .set_attn_scale(attn_scale)
                                   .set_seq_len_q(SEQ_LEN_Q_)
                                   .set_seq_len_kv(SEQ_LEN_KV_)
                                   .set_padding_mask(true);
  if (dropout_probability != 0.0f) {
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_uid(SEED)
                                      .set_name("Seed")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(
                                          dropoutseed.dtype() == kInt
                                              ? fe::DataType_t::INT32
                                              : fe::DataType_t::INT64));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_uid(OFFSET)
                                        .set_name("Offset")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(
                                            dropoutoffset.dtype() == kInt
                                                ? fe::DataType_t::INT32
                                                : fe::DataType_t::INT64));
    sdpa_backward_options.set_dropout(dropout_probability, seed, offset);
  }
  auto Q_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(Q)
                                  .set_name("Q")
                                  .set_dim({b, h_q, s_q, d_qk})
                                  .set_stride(thd_to_bhsd_strides(q)));
  auto K_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(K)
                                  .set_name("K")
                                  .set_dim({b, h_k, s_kv, d_qk})
                                  .set_stride(thd_to_bhsd_strides(k)));
  auto V_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(V)
                                  .set_name("V")
                                  .set_dim({b, h_v, s_kv, d_v})
                                  .set_stride(thd_to_bhsd_strides(v)));
  auto O_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(O)
                                  .set_name("O")
                                  .set_dim({b, h_q, s_q, d_v})
                                  .set_stride(thd_to_bhsd_strides(o)));

  if (attn_bias.has_value()) {
    TORCH_CHECK(
        false,
        "attn_bias not yet supported with cuDNN Attention and NestedTensor");
    sdpa_backward_options.set_bias(
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec())));
  }
  auto RAG_Q_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_Q_OFF)
                            .set_name("cum_seq_q")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_K_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_K_OFF)
                            .set_name("cum_seq_k")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_V_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_V_OFF)
                            .set_name("cum_seq_v")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_DQ_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_DQ_OFF)
                            .set_name("cum_seq_q")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_DK_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_DK_OFF)
                            .set_name("cum_seq_k")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_DV_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_DV_OFF)
                            .set_name("cum_seq_v")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_O_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_O_OFF)
                            .set_name("cum_seq_o")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_DO_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_DO_OFF)
                            .set_name("cum_seq_do")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto RAG_STATS_OFF_ =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(RAG_LSE_OFF)
                            .set_name("cum_seq_stats")
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  O_->set_ragged_offset(RAG_O_OFF_);
  Q_->set_ragged_offset(RAG_Q_OFF_);
  K_->set_ragged_offset(RAG_K_OFF_);
  V_->set_ragged_offset(RAG_V_OFF_);
  auto STATS =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_uid(LSE)
                            .set_name("stats")
                            .set_dim({b, h_q, s_q, 1})
                            .set_stride(thd_to_bhsd_strides(softmaxstats))
                            .set_data_type(fe::DataType_t::FLOAT));
  STATS->set_ragged_offset(RAG_STATS_OFF_);
  auto DO_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                   .set_ragged_offset(RAG_DO_OFF_)
                                   .set_uid(DO)
                                   .set_name("DO")
                                   .set_dim({b, h_q, s_q, d_v})
                                   .set_stride(thd_to_bhsd_strides(dO)));
  auto [Dq, Dk, Dv] = mha_graph->sdpa_backward(
      Q_, K_, V_, O_, DO_, STATS, sdpa_backward_options);
  Dq->set_output(true)
      .set_uid(DQ)
      .set_ragged_offset(RAG_DQ_OFF_)
      .set_dim({b, h_q, s_q, d_qk})
      .set_stride(thd_to_bhsd_strides(dQ));
  Dk->set_output(true)
      .set_uid(DK)
      .set_ragged_offset(RAG_DK_OFF_)
      .set_dim({b, h_k, s_kv, d_qk})
      .set_stride(thd_to_bhsd_strides(dK));
  Dv->set_output(true)
      .set_uid(DV)
      .set_ragged_offset(RAG_DV_OFF_)
      .set_dim({b, h_v, s_kv, d_v})
      .set_stride(thd_to_bhsd_strides(dV));

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));
  return mha_graph;
}

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel()) {
    return;
  }
  check_cudnn_sdpa_decode(s_q);
  Tensor seqlen_q, seqlen_kv;
  Tensor rag_off_q, rag_off_k, rag_off_v, rag_off_o, rag_off_lse;

  if (!o.defined()) {
    // q is passed to us in BHSD dim order
    alloc_with_matching_layout(q, o, {b, h, s_q, d_v});
  }
  bool use_ragged = use_ragged_in_dense(q, k, v, o, attn_bias.has_value());
  if (return_softmaxstats && !softmaxstats.defined()) {
    // TODO(eqy): investigate why cuDNN doesn't like BSH layout softmaxstats
    if (!use_ragged) {
      softmaxstats = at::empty({b, h, s_q, 1}, q.options().dtype(kFloat));
    } else {
      softmaxstats =
          at::empty({b, s_q, h, 1}, q.options().dtype(kFloat)).transpose(1, 2);
    }
  }

  if (use_ragged) {
    seqlen_q = at::full({b, 1, 1, 1}, s_q, q.options().dtype(kInt));
    seqlen_kv = at::full({b, 1, 1, 1}, s_kv, q.options().dtype(kInt));
    auto cum_seqlen_q = at::full({b + 1, 1, 1, 1}, s_q, q.options().dtype(kInt))
                            .cumsum(0, kInt)
                            .add_(-s_q);
    auto cum_seqlen_kv =
        at::full({b + 1, 1, 1, 1}, s_kv, q.options().dtype(kInt))
            .cumsum(0, kInt)
            .add_(-s_kv);
    rag_off_q = cum_seqlen_q.mul(q.stride(-2));
    rag_off_k = cum_seqlen_kv.mul(k.stride(-2));
    rag_off_v = cum_seqlen_kv.mul(v.stride(-2));
    rag_off_o = cum_seqlen_q.mul(o.stride(-2));
    if (return_softmaxstats) {
      rag_off_lse = cum_seqlen_q.mul(softmaxstats.stride(-2));
    }
  }

  const auto dprops = at::cuda::getCurrentDeviceProperties();
  auto _dropoutseed = dropoutseed;
  auto _dropoutoffset = dropoutoffset;
  // cuDNN dropout bug requires these to be in int64
  if (dprops->major == 10 && dprops->minor == 0) {
    _dropoutseed = dropoutseed.to(kLong);
    _dropoutoffset = dropoutoffset.to(kLong);
  }

  cudnnHandle_t handle = getCudnnHandle();

  // NB: The key initialization will round up sequence length, stride data etc.
  // if use_ragged_in_dense is enabled (to allow multiple sequence lengths to
  // reuse the same cached value/graph)
  MHACacheKeyWrapper key(
      b,
      h,
      s_q,
      s_kv,
      d_qk,
      d_v,
      q,
      k,
      v,
      attn_bias,
      o,
      Tensor(),
      softmaxstats,
      dropout_probability,
      is_causal ? CausalMask::TOP_LEFT : CausalMask::NONE,
      return_softmaxstats,
      false);
  auto [cache_it, not_found] = getMHAGraphCache_().try_emplace(key, nullptr);
  if (not_found) {
    cache_it->second = build_graph(
        b,
        h,
        s_q,
        s_kv,
        d_qk,
        d_v,
        scaling_factor,
        return_softmaxstats,
        is_causal,
        dropout_probability,
        q,
        k,
        v,
        attn_bias,
        softmaxstats,
        o,
        _dropoutseed,
        _dropoutoffset,
        handle);
  }
  const fe::graph::Graph& mha_graph = *cache_it->second;
  std::unordered_map<int64_t, void*> variant_pack = {
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {SCALE, &scaling_factor},
      {O, o.mutable_data_ptr()}};
  if (return_softmaxstats) {
    variant_pack[LSE] = softmaxstats.mutable_data_ptr();
  }
  if (attn_bias.has_value()) {
    variant_pack[BIAS] = attn_bias.value().mutable_data_ptr();
  }
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = _dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = _dropoutoffset.mutable_data_ptr();
  }
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    variant_pack[SEQ_LEN_Q] = seqlen_q.mutable_data_ptr();
    variant_pack[SEQ_LEN_KV] = seqlen_kv.mutable_data_ptr();
    variant_pack[RAG_Q_OFF] = rag_off_q.mutable_data_ptr();
    variant_pack[RAG_K_OFF] = rag_off_k.mutable_data_ptr();
    variant_pack[RAG_V_OFF] = rag_off_v.mutable_data_ptr();
    variant_pack[RAG_O_OFF] = rag_off_o.mutable_data_ptr();
    if (return_softmaxstats) {
      variant_pack[RAG_LSE_OFF] = rag_off_lse.mutable_data_ptr();
    }
  }
  auto workspace_size = mha_graph.get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  auto err = mha_graph.execute(handle, variant_pack, workspace_ptr.get());
  check_cudnn_sdpa_execution(std::move(err));
}

void run_cudnn_SDP_fprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool return_softmaxstats,
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const std::optional<Tensor>& seqused_k,
    const std::optional<Tensor>& page_table,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  cudnnHandle_t handle = getCudnnHandle();
  // Return well-formed outputs for 0-element inputs instead of undefined
  // tensors; empty KV attends to nothing, so o is zero and the LSE is -inf.
  if (!q.numel() || !k.numel() || !v.numel()) {
    if (!o.defined()) {
      alloc_with_matching_layout(q, o, {q.size(0), h_q, d_v});
    }
    o.zero_();
    if (return_softmaxstats && !softmaxstats.defined()) {
      softmaxstats = at::full(
          {h_q, q.size(0)},
          -std::numeric_limits<float>::infinity(),
          q.options().dtype(kFloat));
    }
    return;
  }
  check_cudnn_sdpa_decode(s_q);
  const bool is_paged = page_table.has_value();
  TORCH_INTERNAL_ASSERT(
      !is_paged || seqused_k.has_value(),
      "paged cuDNN attention requires seqused_k");
  // seqused_k describes a KV cache whose queries are its newest tokens, so
  // causal masking aligns the diagonal to the bottom right of each sequence
  // (FlashAttention semantics) instead of the top left.
  const CausalMask causal_mask = !is_causal ? CausalMask::NONE
      : seqused_k.has_value()               ? CausalMask::BOTTOM_RIGHT
                                            : CausalMask::TOP_LEFT;
  if (causal_mask == CausalMask::BOTTOM_RIGHT) {
    TORCH_CHECK(
        at::detail::getCUDAHooks().versionRuntimeCuDNN() >= 92400,
        "cuDNN varlen causal attention with a KV cache requires cuDNN >= 9.24.");
  }
  if (seqused_k.has_value()) {
    checkInt32Alignment(seqused_k.value(), "seqused_k");
  }
  if (is_paged) {
    checkInt32Alignment(page_table.value(), "block_table");
  }

  if (!o.defined()) {
    alloc_with_matching_layout(q, o, {q.size(0), h_q, d_v});
  }
  const auto sequence_length_mode = AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS &&
          !seqused_k.has_value() && dropout_probability == 0.0 &&
          q.stride(-3) > 0 && k.stride(-3) > 0 && v.stride(-3) > 0 &&
          o.stride(-3) > 0
      ? SequenceLengthMode::CUMULATIVE
      : SequenceLengthMode::PER_SEQUENCE;

  if (return_softmaxstats && !softmaxstats.defined()) {
    // cuDNN wants T, H, 1, but torch/FA convention is H, T
    softmaxstats = at::empty({h_q, q.size(0)}, q.options().dtype(kFloat));
  }
  auto softmaxstats_ = softmaxstats;
  if (return_softmaxstats) {
    TORCH_INTERNAL_ASSERT(
        softmaxstats.dim() == 2, "cuDNN SDPA expected a 2D (H, T) softmax_lse");
    softmaxstats_ = softmaxstats.unsqueeze(-1).transpose(0, 1);
  }

  MHACacheKeyWrapper key(
      b,
      h_q,
      s_q, // max-seqlen-q
      s_kv, // max-seqlen-kv
      d_qk,
      d_v,
      q,
      k,
      v,
      attn_bias,
      o,
      Tensor(),
      softmaxstats_,
      dropout_probability,
      causal_mask,
      return_softmaxstats,
      true,
      page_table,
      sequence_length_mode);

  MHAGraphCache& cache = getMHAGraphCache_();
  auto cache_it = cache.find(key);
  if (cache_it == cache.end()) {
    auto graph = build_graph_nestedtensor(
        b,
        h_q,
        h_k,
        h_v,
        s_q,
        s_kv,
        d_qk,
        d_v,
        scaling_factor,
        return_softmaxstats,
        causal_mask,
        dropout_probability,
        cum_seqlen_q,
        cum_seqlen_kv,
        page_table,
        sequence_length_mode,
        q,
        k,
        v,
        attn_bias,
        softmaxstats_,
        o,
        dropoutseed,
        dropoutoffset,
        handle);
    cache_it = cache.try_emplace(key, std::move(graph)).first;
  }
  const fe::graph::Graph& mha_graph = *cache_it->second;

  check_ragged_offset_capacity(q, "query");
  check_ragged_offset_capacity(o, "out");
  if (!is_paged) {
    check_ragged_offset_capacity(k, "key");
    check_ragged_offset_capacity(v, "value");
  }
  std::unordered_map<int64_t, void*> variant_pack = {
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {SCALE, &scaling_factor},
      {O, o.mutable_data_ptr()}};
  Tensor seqlen_q, seqlen_kv, rag_q_off, rag_k_off, rag_v_off, rag_o_off;
#if AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS
  if (sequence_length_mode == SequenceLengthMode::CUMULATIVE) {
    variant_pack[CU_SEQ_LEN_Q] = cum_seqlen_q.mutable_data_ptr();
    variant_pack[CU_SEQ_LEN_KV] = cum_seqlen_kv.mutable_data_ptr();
    variant_pack[RAG_Q_OFF] = cum_seqlen_q.mutable_data_ptr();
    variant_pack[RAG_O_OFF] = cum_seqlen_q.mutable_data_ptr();
    variant_pack[RAG_K_OFF] = cum_seqlen_kv.mutable_data_ptr();
    variant_pack[RAG_V_OFF] = cum_seqlen_kv.mutable_data_ptr();
  }
#endif
  if (sequence_length_mode == SequenceLengthMode::PER_SEQUENCE) {
    const bool shared_cum_seqlen = cum_seqlen_q.is_same(cum_seqlen_kv);
    seqlen_q = at::diff(cum_seqlen_q, 1, 0);
    if (seqused_k.has_value()) {
      seqlen_kv = seqused_k.value();
    } else if (shared_cum_seqlen) {
      seqlen_kv = seqlen_q;
    } else {
      seqlen_kv = at::diff(cum_seqlen_kv, 1, 0);
    }
    rag_q_off = cum_seqlen_q.mul(q.stride(-3));
    rag_o_off =
        ragged_offset(cum_seqlen_q, o.stride(-3), rag_q_off, q.stride(-3));
    variant_pack[RAG_Q_OFF] = rag_q_off.mutable_data_ptr();
    variant_pack[RAG_O_OFF] = rag_o_off.mutable_data_ptr();
    variant_pack[SEQ_LEN_Q] = seqlen_q.mutable_data_ptr();
    variant_pack[SEQ_LEN_KV] = seqlen_kv.mutable_data_ptr();
    if (!is_paged) {
      rag_k_off = shared_cum_seqlen && k.stride(-3) == q.stride(-3)
          ? rag_q_off
          : cum_seqlen_kv.mul(k.stride(-3));
      rag_v_off =
          ragged_offset(cum_seqlen_kv, v.stride(-3), rag_k_off, k.stride(-3));
      variant_pack[RAG_K_OFF] = rag_k_off.mutable_data_ptr();
      variant_pack[RAG_V_OFF] = rag_v_off.mutable_data_ptr();
    }
  }
  if (is_paged) {
    variant_pack[PAGE_TABLE_K] = page_table.value().mutable_data_ptr();
    variant_pack[PAGE_TABLE_V] = page_table.value().mutable_data_ptr();
  }
  if (return_softmaxstats) {
    TORCH_INTERNAL_ASSERT(
        softmaxstats_.stride(-3) == 1,
        "cuDNN SDPA expected a contiguous (H, T) softmax_lse");
    variant_pack[LSE] = softmaxstats_.mutable_data_ptr();
    variant_pack[RAG_LSE_OFF] = cum_seqlen_q.mutable_data_ptr();
  }
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = dropoutoffset.mutable_data_ptr();
  }
  if (attn_bias.has_value()) {
    TORCH_CHECK(false, "bias not supported with nestedtensor");
  }
  auto workspace_size = mha_graph.get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  auto err = mha_graph.execute(handle, variant_pack, workspace_ptr.get());
  check_cudnn_sdpa_execution(std::move(err));
}

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel() || !o.numel() || !dO.numel() ||
      !softmaxstats.numel()) {
    return;
  }
  check_cudnn_sdpa_decode(s_q);
  Tensor seqlen_q, seqlen_kv;
  Tensor rag_off_q, rag_off_k, rag_off_v, rag_off_o, rag_off_lse;

  auto dprops = at::cuda::getCurrentDeviceProperties();
  auto _dropoutseed = dropoutseed;
  auto _dropoutoffset = dropoutoffset;
  // cuDNN dropout bug requires these to be in int64
  if (dprops->major == 10 && dprops->minor == 0) {
    _dropoutseed = dropoutseed.to(kLong);
    _dropoutoffset = dropoutoffset.to(kLong);
  }

  Tensor dO_ = dO;
// cuDNN < 9.5.1 assumes gradOutput has same strides as Output
#if defined(CUDNN_VERSION) && CUDNN_VERSION < 90501
  if (!same_strides(o, dO)) {
    TORCH_WARN_ONCE(
        "cuDNN SDPA backward got grad_output.strides() != output.strides(), "
        "attempting to materialize a grad_output with matching strides."
        "Consider upgrading cuDNN v9.5.1+ to avoid this warning.");
    permute_to_matching_layout(o, dO_);
  }
  TORCH_INTERNAL_ASSERT(
      same_strides(o, dO_),
      "cuDNN SDPA expected grad_output.strides() == output.strides(), "
      "the previous step probably failed to materialize a grad_output "
      "with matching strides...");
#else
  const auto innermost_dO_stride = dO.strides()[dO.strides().size() - 1];
  if (innermost_dO_stride != 1 ||
      use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    permute_to_matching_layout(o, dO_);
  }
#endif
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    seqlen_q = at::full({b, 1, 1, 1}, s_q, q.options().dtype(kInt));
    seqlen_kv = at::full({b, 1, 1, 1}, s_kv, q.options().dtype(kInt));
    auto cum_seqlen_q = at::full({b + 1, 1, 1, 1}, s_q, q.options().dtype(kInt))
                            .cumsum(0, kInt)
                            .add_(-s_q);
    auto cum_seqlen_kv =
        at::full({b + 1, 1, 1, 1}, s_kv, q.options().dtype(kInt))
            .cumsum(0, kInt)
            .add_(-s_kv);
    rag_off_q = cum_seqlen_q.mul(q.stride(-2));
    rag_off_k = cum_seqlen_kv.mul(k.stride(-2));
    rag_off_v = cum_seqlen_kv.mul(v.stride(-2));
    rag_off_o = cum_seqlen_q.mul(o.stride(-2));
    rag_off_lse = cum_seqlen_q.mul(softmaxstats.stride(-2));
  }

  cudnnHandle_t handle = getCudnnHandle();
  MHACacheKeyWrapper key(
      b,
      h,
      s_q,
      s_kv,
      d_qk,
      d_v,
      q,
      k,
      v,
      attn_bias,
      o,
      dO_,
      softmaxstats,
      dropout_probability,
      is_causal ? CausalMask::TOP_LEFT : CausalMask::NONE,
      true,
      false);
  auto [cache_it, not_found] =
      getMHAGraphBackwardCache_().try_emplace(key, nullptr);
  if (not_found) {
    cache_it->second = build_graph_backward(
        b,
        h,
        s_q,
        s_kv,
        d_qk,
        d_v,
        scaling_factor,
        is_causal,
        dropout_probability,
        q,
        k,
        v,
        attn_bias,
        o,
        dO_,
        softmaxstats,
        dQ,
        dK,
        dV,
        _dropoutseed,
        _dropoutoffset,
        handle);
  }
  const fe::graph::Graph& mha_graph = *cache_it->second;

  std::unordered_map<int64_t, void*> variant_pack = {
      // inputs
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {O, o.mutable_data_ptr()},
      {DO, dO_.mutable_data_ptr()},
      {LSE, softmaxstats.mutable_data_ptr()},
      // outputs
      {DQ, dQ.mutable_data_ptr()},
      {DK, dK.mutable_data_ptr()},
      {DV, dV.mutable_data_ptr()},
      {SCALE, &scaling_factor}};
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = _dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = _dropoutoffset.mutable_data_ptr();
  }
  if (attn_bias.has_value()) {
    variant_pack[BIAS] = attn_bias.value().mutable_data_ptr();
  }
  if (use_ragged_in_dense(q, k, v, o, attn_bias.has_value())) {
    variant_pack[SEQ_LEN_Q] = seqlen_q.mutable_data_ptr();
    variant_pack[SEQ_LEN_KV] = seqlen_kv.mutable_data_ptr();
    variant_pack[RAG_Q_OFF] = rag_off_q.mutable_data_ptr();
    variant_pack[RAG_K_OFF] = rag_off_k.mutable_data_ptr();
    variant_pack[RAG_V_OFF] = rag_off_v.mutable_data_ptr();
    variant_pack[RAG_O_OFF] = rag_off_o.mutable_data_ptr();
    variant_pack[RAG_LSE_OFF] = rag_off_lse.mutable_data_ptr();
  }

  auto workspace_size = mha_graph.get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  auto err = mha_graph.execute(handle, variant_pack, workspace_ptr.get());
  check_cudnn_sdpa_execution(std::move(err));
}

void run_cudnn_SDP_bprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset) {
  if (!q.numel() || !k.numel() || !v.numel() || !o.numel() || !dO.numel() ||
      !softmaxstats.numel()) {
    dQ.zero_();
    dK.zero_();
    dV.zero_();
    return;
  }
  check_cudnn_sdpa_decode(s_q);
  TORCH_CHECK(
      softmaxstats.dim() == 2, "cuDNN SDPA expected a 2D (H, T) softmax_lse");
  auto softmaxstats_ = softmaxstats.unsqueeze(-1).transpose(0, 1);

  // Alignment is not part of the cache key, and cuDNN requires 16-byte
  // pointer/stride alignment. Preserve dO strides when those hold.
  Tensor dO_ = dO;
  if (!has_aligned_varlen_layout(dO)) {
    dO_ = dO.clone(at::MemoryFormat::Contiguous);
  }
  TORCH_INTERNAL_ASSERT(
      has_aligned_varlen_layout(dO_),
      "cuDNN SDPA expected grad_output to have 16-byte-aligned storage and "
      "non-broadcast strides, with a contiguous last dimension");

  const bool shared_cum_seqlen = cum_seqlen_q.is_same(cum_seqlen_kv);
  auto seqlen_q = at::diff(cum_seqlen_q, 1, 0);
  auto seqlen_kv = shared_cum_seqlen ? seqlen_q : at::diff(cum_seqlen_kv, 1, 0);
  check_ragged_offset_capacity(q, "query");
  check_ragged_offset_capacity(k, "key");
  check_ragged_offset_capacity(v, "value");
  check_ragged_offset_capacity(o, "out");
  check_ragged_offset_capacity(dO_, "grad_out");
  check_ragged_offset_capacity(dQ, "grad_query");
  check_ragged_offset_capacity(dK, "grad_key");
  check_ragged_offset_capacity(dV, "grad_value");
  const int64_t q_token_stride = q.stride(-3);
  const int64_t kv_token_stride = k.stride(-3);
  auto rag_q_off = cum_seqlen_q.mul(q_token_stride);
  auto rag_k_off = shared_cum_seqlen && kv_token_stride == q_token_stride
      ? rag_q_off
      : cum_seqlen_kv.mul(kv_token_stride);
  auto rag_v_off =
      ragged_offset(cum_seqlen_kv, v.stride(-3), rag_k_off, kv_token_stride);
  auto rag_o_off =
      ragged_offset(cum_seqlen_q, o.stride(-3), rag_q_off, q_token_stride);
  auto rag_dq_off =
      ragged_offset(cum_seqlen_q, dQ.stride(-3), rag_q_off, q_token_stride);
  auto rag_dk_off =
      ragged_offset(cum_seqlen_kv, dK.stride(-3), rag_k_off, kv_token_stride);
  auto rag_dv_off =
      ragged_offset(cum_seqlen_kv, dV.stride(-3), rag_v_off, v.stride(-3));
  auto rag_do_off =
      ragged_offset(cum_seqlen_q, dO_.stride(-3), rag_o_off, o.stride(-3));
  TORCH_CHECK(
      softmaxstats_.stride(-3) == 1,
      "cuDNN SDPA expected a contiguous (H, T) softmax_lse");

  auto dprops = at::cuda::getCurrentDeviceProperties();
  auto _dropoutseed = dropoutseed;
  auto _dropoutoffset = dropoutoffset;
  // cuDNN dropout bug requires these to be in int64
  if (dprops->major == 10 && dprops->minor == 0) {
    _dropoutseed = dropoutseed.to(kLong);
    _dropoutoffset = dropoutoffset.to(kLong);
  }

  cudnnHandle_t handle = getCudnnHandle();

  MHACacheKeyWrapper key(
      b,
      h_q,
      s_q, // max-seqlen-q
      s_kv, // max-seqlen-kv
      d_qk,
      d_v,
      q,
      k,
      v,
      attn_bias,
      o,
      dO_,
      softmaxstats_,
      dropout_probability,
      is_causal ? CausalMask::TOP_LEFT : CausalMask::NONE,
      true,
      true);

  MHAGraphCache& cache = getMHAGraphBackwardCache_();
  auto cache_it = cache.find(key);
  if (cache_it == cache.end()) {
    auto graph = build_graph_backward_nestedtensor(
        b,
        h_q,
        h_k,
        h_v,
        s_q,
        s_kv,
        d_qk,
        d_v,
        scaling_factor,
        is_causal,
        dropout_probability,
        cum_seqlen_q,
        cum_seqlen_kv,
        q,
        k,
        v,
        attn_bias,
        o,
        dO_,
        softmaxstats_,
        dQ,
        dK,
        dV,
        dropoutseed,
        dropoutoffset,
        handle);
    cache_it = cache.try_emplace(key, std::move(graph)).first;
  }
  const fe::graph::Graph& mha_graph = *cache_it->second;
  std::unordered_map<int64_t, void*> variant_pack = {
      // inputs
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {O, o.mutable_data_ptr()},
      {DO, dO_.mutable_data_ptr()},
      {LSE, softmaxstats_.mutable_data_ptr()},
      // outputs
      {DQ, dQ.mutable_data_ptr()},
      {DK, dK.mutable_data_ptr()},
      {DV, dV.mutable_data_ptr()},
      {SCALE, &scaling_factor},
      {RAG_Q_OFF, rag_q_off.mutable_data_ptr()},
      {RAG_O_OFF, rag_o_off.mutable_data_ptr()},
      {RAG_K_OFF, rag_k_off.mutable_data_ptr()},
      {RAG_V_OFF, rag_v_off.mutable_data_ptr()},
      {RAG_DQ_OFF, rag_dq_off.mutable_data_ptr()},
      {RAG_DK_OFF, rag_dk_off.mutable_data_ptr()},
      {RAG_DV_OFF, rag_dv_off.mutable_data_ptr()},
      {RAG_DO_OFF, rag_do_off.mutable_data_ptr()},
      {RAG_LSE_OFF, cum_seqlen_q.mutable_data_ptr()},
      {SEQ_LEN_Q, seqlen_q.mutable_data_ptr()},
      {SEQ_LEN_KV, seqlen_kv.mutable_data_ptr()}};
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = _dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = _dropoutoffset.mutable_data_ptr();
  }
  TORCH_CHECK(
      !attn_bias.has_value(),
      "attn_bias not yet supported with cuDNN Attention and NestedTensor");

  auto workspace_size = mha_graph.get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  auto err = mha_graph.execute(handle, variant_pack, workspace_ptr.get());
  check_cudnn_sdpa_execution(std::move(err));
}

} // namespace at::native

#endif
