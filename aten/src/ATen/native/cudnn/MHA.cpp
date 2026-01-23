#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()
#include <cudnn_frontend.h>
#endif

#if defined(USE_ROCM) || !AT_CUDNN_ENABLED() ||         \
    (defined(CUDNN_VERSION) && CUDNN_VERSION < 8900) || \
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

#else // AT_CUDNN_ENABLED && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8900
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/native/cudnn/MHA.h>
#include <ATen/native/transformers/sdp_utils.h>

#include <ATen/cuda/Exceptions.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cudnn.h>

#include <iostream>

namespace at::native {

namespace fe = cudnn_frontend;

constexpr uint8_t MAX_MHA_DIM = 4;

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

struct MHAParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  std::array<int, MAX_MHA_DIM> q_dim;
  std::array<int, MAX_MHA_DIM> k_dim;
  std::array<int, MAX_MHA_DIM> v_dim;
  std::array<int, MAX_MHA_DIM> q_stride;
  std::array<int, MAX_MHA_DIM> k_stride;
  std::array<int, MAX_MHA_DIM> v_stride;
  std::array<int, MAX_MHA_DIM> bias_dim;
  std::array<int, MAX_MHA_DIM> bias_stride;
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d_qk;
  int64_t d_v;
  double dropout_probability;
  bool is_causal;
  bool return_softmaxstats;
  // might be redundant if we take 0 dim/stride
  // as signaling no-bias
  bool has_attn_bias;
  bool use_ragged;
};

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
    double dropout_probability,
    bool is_causal,
    bool return_softmaxstats,
    bool is_nested) {
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
  params.is_causal = is_causal;
  params.return_softmaxstats = return_softmaxstats;
  params.has_attn_bias = attn_bias.has_value();
  // Expect 4D dense tensor, 3D nested case (THD)
  TORCH_INTERNAL_ASSERT(
      q.sizes().size() == (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested),
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      q.strides().size() == (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested),
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.sizes().size() == (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested),
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.strides().size() == (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested),
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.sizes().size() == (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested),
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.strides().size() == (uint8_t)(MAX_MHA_DIM - (uint8_t)is_nested),
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim.begin());
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride.begin());
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim.begin());
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride.begin());
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim.begin());
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride.begin());
  bool use_ragged = use_ragged_in_dense(q, k, v, q, params.has_attn_bias);
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
  // uninit is OK as the struct is memset 0'd
  if (params.has_attn_bias) {
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
      double dropout_probability,
      bool is_causal,
      bool return_softmaxstats,
      bool is_nested) {
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
        dropout_probability,
        is_causal,
        return_softmaxstats,
        is_nested);
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
  RAG_Q_OFF,
  RAG_K_OFF,
  RAG_V_OFF,
  RAG_O_OFF,
  RAG_LSE_OFF
};

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
    scaled_dot_product_flash_attention_options.set_bias(
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec())));
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
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
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

  auto scaled_dot_product_flash_attention_options =
      fe::graph::SDPA_attributes()
          .set_name("CUDNN_SDPA_NESTEDTENSOR")
#if CUDNN_FRONTEND_VERSION <= 11200
          .set_is_inference(!return_softmaxstats)
#else
          .set_generate_stats(return_softmaxstats)
#endif
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
    scaled_dot_product_flash_attention_options.set_dropout(
        dropout_probability, seed, offset);
  }
  // We hardcode BSHD to cuDNN even though the underlying layout is THD
  auto q_strides = q.strides();
  auto k_strides = k.strides();
  auto v_strides = v.strides();
  // NB: cuDNN API shape is transposed: we pass it nominally as HTD
  constexpr int strideidx0 = 1;
  constexpr int strideidx1 = 0;
  constexpr int strideidx2 = 2;
  auto Q_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(Q)
                                  .set_name("Q")
                                  .set_dim({b, h_q, s_q, d_qk})
                                  .set_stride(
                                      {INT_MAX,
                                       q_strides[strideidx0],
                                       q_strides[strideidx1],
                                       q_strides[strideidx2]}));
  auto K_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(K)
                                  .set_name("K")
                                  .set_dim({b, h_k, s_kv, d_qk})
                                  .set_stride(
                                      {INT_MAX,
                                       k_strides[strideidx0],
                                       k_strides[strideidx1],
                                       k_strides[strideidx2]}));
  auto V_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(V)
                                  .set_name("V")
                                  .set_dim({b, h_v, s_kv, d_v})
                                  .set_stride(
                                      {INT_MAX,
                                       v_strides[strideidx0],
                                       v_strides[strideidx1],
                                       v_strides[strideidx2]}));
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
  Q_->set_ragged_offset(RAG_Q_OFF_);
  K_->set_ragged_offset(RAG_K_OFF_);
  V_->set_ragged_offset(RAG_V_OFF_);
  auto [O_, Stats] =
      mha_graph->sdpa(Q_, K_, V_, scaled_dot_product_flash_attention_options);
  auto o_strides = o.strides();
  O_->set_output(true)
      .set_uid(O)
      .set_dim({b, h_q, s_q, d_v})
      .set_stride(
          {INT_MAX,
           o_strides[strideidx0],
           o_strides[strideidx1],
           o_strides[strideidx2]});

  O_->set_ragged_offset(RAG_O_OFF_);
  if (Stats) {
    auto RAG_STATS_OFF =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(RAG_LSE_OFF)
                              .set_name("cum_seq_stats")
                              .set_dim({b + 1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::INT32));
    Stats->set_output(true)
        .set_uid(LSE)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({b, h_q, s_q, 1})
        .set_stride({h_q * s_q, 1, h_q, 1});
    Stats->set_ragged_offset(RAG_STATS_OFF);
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
    sdpa_backward_options.set_bias(
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec())));
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
  auto q_strides = q.strides();
  auto k_strides = k.strides();
  auto v_strides = v.strides();
  // NB: cuDNN API shape is transposed
  constexpr int strideidx0 = 1;
  constexpr int strideidx1 = 0;
  constexpr int strideidx2 = 2;
  auto Q_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(Q)
                                  .set_name("Q")
                                  .set_dim({b, h_q, s_q, d_qk})
                                  .set_stride(
                                      {INT_MAX,
                                       q_strides[strideidx0],
                                       q_strides[strideidx1],
                                       q_strides[strideidx2]}));
  auto K_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(K)
                                  .set_name("K")
                                  .set_dim({b, h_k, s_kv, d_qk})
                                  .set_stride(
                                      {INT_MAX,
                                       k_strides[strideidx0],
                                       k_strides[strideidx1],
                                       k_strides[strideidx2]}));
  auto V_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(V)
                                  .set_name("V")
                                  .set_dim({b, h_v, s_kv, d_v})
                                  .set_stride(
                                      {INT_MAX,
                                       v_strides[strideidx0],
                                       v_strides[strideidx1],
                                       v_strides[strideidx2]}));
  auto o_strides = o.strides();
  auto O_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(O)
                                  .set_name("O")
                                  .set_dim({b, h_q, s_q, d_v})
                                  .set_stride(
                                      {INT_MAX,
                                       o_strides[strideidx0],
                                       o_strides[strideidx1],
                                       o_strides[strideidx2]}));

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
  auto STATS = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_uid(LSE)
                                     .set_name("stats")
                                     .set_dim({b, h_q, s_q, 1})
                                     .set_stride({s_q * h_q, 1, h_q, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
  STATS->set_ragged_offset(RAG_STATS_OFF_);
  auto do_strides = dO.strides();
  auto DO_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                   .set_ragged_offset(RAG_O_OFF_)
                                   .set_uid(DO)
                                   .set_name("DO")
                                   .set_dim({b, h_q, s_q, d_v})
                                   .set_stride(
                                       {INT_MAX,
                                        do_strides[strideidx0],
                                        do_strides[strideidx1],
                                        do_strides[strideidx2]}));
  auto [Dq, Dk, Dv] = mha_graph->sdpa_backward(
      Q_, K_, V_, O_, DO_, STATS, sdpa_backward_options);
  Dq->set_output(true)
      .set_uid(DQ)
      .set_ragged_offset(RAG_Q_OFF_)
      .set_dim({b, h_q, s_q, d_qk})
      .set_stride(
          {INT_MAX,
           q_strides[strideidx0],
           q_strides[strideidx1],
           q_strides[strideidx2]});
  Dk->set_output(true)
      .set_uid(DK)
      .set_ragged_offset(RAG_K_OFF_)
      .set_dim({b, h_k, s_kv, d_qk})
      .set_stride(
          {INT_MAX,
           k_strides[strideidx0],
           k_strides[strideidx1],
           k_strides[strideidx2]});
  Dv->set_output(true)
      .set_uid(DV)
      .set_ragged_offset(RAG_V_OFF_)
      .set_dim({b, h_v, s_kv, d_v})
      .set_stride(
          {INT_MAX,
           v_strides[strideidx0],
           v_strides[strideidx1],
           v_strides[strideidx2]});

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
      dropout_probability,
      is_causal,
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
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
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
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset) {
  cudnnHandle_t handle = getCudnnHandle();
  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel()) {
    return;
  }

  if (!o.defined()) {
    o = at::empty({q.size(0), h_q, d_v}, q.options());
  }

  if (return_softmaxstats && !softmaxstats.defined()) {
    softmaxstats = at::empty({q.size(0), h_q, 1}, q.options().dtype(kFloat));
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
      dropout_probability,
      is_causal,
      return_softmaxstats,
      true);

  MHAGraphCache& cache = getMHAGraphCache_();
  auto cache_it = cache.find(key);
  std::unique_ptr<fe::graph::Graph> mha_graph_storage;
  if (cache_it == cache.end()) {
    mha_graph_storage = build_graph_nestedtensor(
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
        is_causal,
        dropout_probability,
        cum_seqlen_q,
        cum_seqlen_kv,
        q,
        k,
        v,
        attn_bias,
        softmaxstats,
        o,
        dropoutseed,
        dropoutoffset,
        handle);
  }
  const fe::graph::Graph& mha_graph =
      mha_graph_storage ? *mha_graph_storage : *cache_it->second;

  auto seqlen_q = at::diff(cum_seqlen_q, 1, 0);
  auto seqlen_kv = at::diff(cum_seqlen_kv, 1, 0);
  auto rag_q_off = cum_seqlen_q.mul(q.stride(-3));
  auto rag_k_off = cum_seqlen_kv.mul(k.stride(-3));
  auto rag_v_off = cum_seqlen_kv.mul(v.stride(-3));
  auto rag_o_off = cum_seqlen_q.mul(o.stride(-3));
  auto rag_stats_off = cum_seqlen_q.mul(h_q);
  std::unordered_map<int64_t, void*> variant_pack = {
      {Q, q.mutable_data_ptr()},
      {K, k.mutable_data_ptr()},
      {V, v.mutable_data_ptr()},
      {SCALE, &scaling_factor},
      {O, o.mutable_data_ptr()},
      {RAG_Q_OFF, rag_q_off.mutable_data_ptr()},
      {RAG_O_OFF, rag_o_off.mutable_data_ptr()},
      {RAG_K_OFF, rag_k_off.mutable_data_ptr()},
      {RAG_V_OFF, rag_v_off.mutable_data_ptr()},
      {SEQ_LEN_Q, seqlen_q.mutable_data_ptr()},
      {SEQ_LEN_KV, seqlen_kv.mutable_data_ptr()}};
  if (return_softmaxstats) {
    variant_pack[LSE] = softmaxstats.mutable_data_ptr();
    variant_pack[RAG_LSE_OFF] = rag_stats_off.mutable_data_ptr();
  }
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = dropoutseed.mutable_data_ptr();
    variant_pack[OFFSET] = dropoutoffset.mutable_data_ptr();
  }
  if (attn_bias.has_value()) {
    TORCH_CHECK("bias not supported with nestedtensor");
  }
  auto workspace_size = mha_graph.get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
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
      dropout_probability,
      is_causal,
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
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
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
  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel() || !o.numel() || !dO.numel() ||
      !softmaxstats.numel()) {
    return;
  }

  Tensor dO_ = dO;
  const auto innermost_dO_stride = dO.strides()[dO.strides().size() - 1];
  if (innermost_dO_stride != 1) {
    permute_to_matching_layout(o, dO_);
  }

  auto seqlen_q = at::diff(cum_seqlen_q, 1, 0);
  auto seqlen_kv = at::diff(cum_seqlen_kv, 1, 0);
  auto rag_q_off = cum_seqlen_q.mul(q.stride(-3));
  auto rag_k_off = cum_seqlen_kv.mul(k.stride(-3));
  auto rag_v_off = cum_seqlen_kv.mul(v.stride(-3));
  auto rag_o_off = cum_seqlen_q.mul(o.stride(-3));
  auto rag_stats_off = cum_seqlen_q.mul(h_q);

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
      dropout_probability,
      is_causal,
      true,
      true);

  MHAGraphCache& cache = getMHAGraphCache_();
  auto cache_it = cache.find(key);
  std::unique_ptr<fe::graph::Graph> mha_graph_storage;
  if (cache_it == cache.end()) {
    mha_graph_storage = build_graph_backward_nestedtensor(
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
        softmaxstats,
        dQ,
        dK,
        dV,
        dropoutseed,
        dropoutoffset,
        handle);
  }
  const fe::graph::Graph& mha_graph =
      mha_graph_storage ? *mha_graph_storage : *cache_it->second;

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
      {SCALE, &scaling_factor},
      {RAG_Q_OFF, rag_q_off.mutable_data_ptr()},
      {RAG_O_OFF, rag_o_off.mutable_data_ptr()},
      {RAG_K_OFF, rag_k_off.mutable_data_ptr()},
      {RAG_V_OFF, rag_v_off.mutable_data_ptr()},
      {RAG_LSE_OFF, rag_stats_off.mutable_data_ptr()},
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
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
}

} // namespace at::native

#endif
