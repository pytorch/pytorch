#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if defined(USE_ROCM) || !AT_CUDNN_ENABLED() || \
    (defined(CUDNN_VERSION) && CUDNN_VERSION < 8900)

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

#include <ATen/cuda/Exceptions.h>
#include <cudnn_frontend.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cudnn.h>

#include <iostream>

namespace at {
namespace native {

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

#define MAX_MHA_DIM 4

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
    bool return_softmaxstats) {
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
  TORCH_INTERNAL_ASSERT(
      q.sizes().size() == MAX_MHA_DIM,
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      q.strides().size() == MAX_MHA_DIM,
      "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.sizes().size() == MAX_MHA_DIM,
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      k.strides().size() == MAX_MHA_DIM,
      "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.sizes().size() == MAX_MHA_DIM,
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(
      v.strides().size() == MAX_MHA_DIM,
      "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim.begin());
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride.begin());
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim.begin());
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride.begin());
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim.begin());
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride.begin());
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
      bool return_softmaxstats) {
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
        return_softmaxstats);
  }
};

template <typename T, typename KeyType>
struct MHAGraphCache {
  std::unordered_map<KeyType, T, ParamsWrapperHash<KeyType>> engine_cache;

  // no mutexes here as caches are now thread local for v8, can also return a
  // pointer to the Execution Plan if we know it will not be invalidated by
  // another thread
  T* find(const KeyType& key) {
    auto it = engine_cache.find(key);
    if (it == engine_cache.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  void update(const KeyType& key, T& results) {
    engine_cache.erase(key);
    engine_cache.emplace(key, std::move(results));
  }
};

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to
// be thread safe across all engines see Limitations in
// https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html
thread_local MHAGraphCache<
    std::shared_ptr<fe::graph::Graph>,
    MHACacheKeyWrapper>
    mhagraphcache;
thread_local MHAGraphCache<
    std::shared_ptr<fe::graph::Graph>,
    MHACacheKeyWrapper>
    mhagraphbackwardcache;

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

void alloc_with_matching_layout(
    const Tensor& q,
    Tensor& output,
    const std::vector<int64_t>& shape) {
  TORCH_INTERNAL_ASSERT(
      shape.size() == q.sizes().size(),
      "cuDNN SDPA alloc_with_matching_layout got requested shape ndim != q ndim");

  if (std::equal(q.sizes().begin(), q.sizes().end(), shape.begin())) {
    output = at::empty_like(q);
    return;
  }

  // get the "fill order," which is just an argsort on the strides
  std::vector<int> fill_order(shape.size());
  std::iota(fill_order.begin(), fill_order.end(), 0);
  const auto q_strides = q.strides();
  std::stable_sort(
      fill_order.begin(), fill_order.end(), [&q_strides](int idx1, int idx2) {
	 return q_strides[idx1] ? q_strides[idx1] : 1 <  q_strides[idx2] ? q_strides[idx2] : 1;
      });
  std::vector<int64_t> ordered_strides(shape.size());
  int64_t current_stride = 1;
  for (const int dim_idx : fill_order) {
    ordered_strides[dim_idx] = current_stride;
    current_stride *= shape[dim_idx];
  }
  output = at::empty(at::IntArrayRef(shape), q.options())
               .as_strided(
                   at::IntArrayRef(shape), at::IntArrayRef(ordered_strides), 0);
}

void permute_to_matching_layout(const Tensor& output, Tensor& grad_output) {
  const int dims = output.sizes().size();
  std::vector<int64_t> outer_to_inner(dims);
  std::iota(outer_to_inner.begin(), outer_to_inner.end(), 0);
  const auto o_strides = output.strides();
  std::stable_sort(
      outer_to_inner.begin(),
      outer_to_inner.end(),
      [&o_strides](int idx1, int idx2) {
        return o_strides[idx1] > o_strides[idx2];
      });
  std::vector<int64_t> inverse(dims);
  for (int d = 0; d < dims; d++) {
    inverse[d] = std::find(outer_to_inner.begin(), outer_to_inner.end(), d) -
        outer_to_inner.begin();
  }
  grad_output = grad_output.permute(at::IntArrayRef(outer_to_inner))
                    .contiguous()
                    .permute(at::IntArrayRef(inverse));
}

bool same_strides(const Tensor& t1, const Tensor& t2) {
  std::vector<int> t1_strides_no_ones;
  std::vector<int> t2_strides_no_ones;
  const auto t1strides = t1.strides();
  const auto t2strides = t2.strides();
  const int dim = t1strides.size();
  if (dim != (int)t2strides.size()) {
    return false;
  }
  const auto t1sizes = t1.sizes();
  const auto t2sizes = t2.sizes();

  // we are going through strides backward here, but if both are backward it's
  // comparable
  for (int i = 0; i < dim; i++) {
    if (t1sizes[i] > 1) {
      t1_strides_no_ones.push_back(t1strides[i]);
    }
    if (t2sizes[i] > 1) {
      t2_strides_no_ones.push_back(t2strides[i]);
    }
  }
  return std::equal(
      t1_strides_no_ones.begin(),
      t1_strides_no_ones.end(),
      t2_strides_no_ones.begin(),
      t2_strides_no_ones.end());
}
} // namespace

auto build_graph(
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
  auto mha_graph = std::make_shared<fe::graph::Graph>();
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
          .set_is_inference(return_softmaxstats == false)
          .set_causal_mask(is_causal)
          .set_attn_scale(attn_scale);
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
      fe::graph::Tensor_attributes()
          .set_uid(Q)
          .set_name("Q")
          .set_dim(q.sizes().vec())
          .set_stride(fixSizeOneDimStrideSDPA(q.sizes(), q.strides().vec())));
  auto K_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(K)
          .set_name("K")
          .set_dim(k.sizes().vec())
          .set_stride(fixSizeOneDimStrideSDPA(k.sizes(), k.strides().vec())));
  auto V_ = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_uid(V)
          .set_name("V")
          .set_dim(v.sizes().vec())
          .set_stride(fixSizeOneDimStrideSDPA(v.sizes(), v.strides().vec())));
  std::optional<std::shared_ptr<fe::graph::Tensor_attributes>> bias;
  if (attn_bias.has_value()) {
    bias =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec()));
    scaled_dot_product_flash_attention_options.set_bias(bias.value());
  }

  auto [O_, Stats] =
      mha_graph->sdpa(Q_, K_, V_, scaled_dot_product_flash_attention_options);
  O_->set_uid(O);
  O_->set_output(true).set_dim(o.sizes().vec()).set_stride(o.strides().vec());

  if (Stats) {
    Stats->set_uid(LSE);
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
  }

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));

  return mha_graph;
}

auto build_graph_nestedtensor(
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
  auto mha_graph = std::make_shared<fe::graph::Graph>();
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
          .set_is_inference(return_softmaxstats == false)
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
  std::optional<std::shared_ptr<fe::graph::Tensor_attributes>> bias;
  if (attn_bias.has_value()) {
    TORCH_CHECK(
        false,
        "attn_bias not yet supportd with cuDNN Attention and NestedTensor");
    bias =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec()));
    scaled_dot_product_flash_attention_options.set_bias(bias.value());
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

auto build_graph_backward(
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
  auto mha_graph = std::make_shared<fe::graph::Graph>();
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
  auto Q_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(Q)
                                  .set_name("Q")
                                  .set_dim(q.sizes().vec())
                                  .set_stride(q.strides().vec()));
  auto K_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(K)
                                  .set_name("K")
                                  .set_dim(k.sizes().vec())
                                  .set_stride(k.strides().vec()));
  auto V_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(V)
                                  .set_name("V")
                                  .set_dim(v.sizes().vec())
                                  .set_stride(v.strides().vec()));
  std::optional<std::shared_ptr<fe::graph::Tensor_attributes>> bias;
  if (attn_bias.has_value()) {
    bias =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec()));
    sdpa_backward_options.set_bias(bias.value());
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

  auto O_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(O)
                                  .set_name("O")
                                  .set_dim(o.sizes().vec())
                                  .set_stride(o.strides().vec()));
  auto Stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_uid(LSE)
                                     .set_name("Stats")
                                     .set_dim(softmaxstats.sizes().vec())
                                     .set_stride(softmaxstats.strides().vec())
                                     .set_data_type(fe::DataType_t::FLOAT));
  auto Do = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_uid(DO)
                                  .set_name("DO")
                                  .set_dim(dO.sizes().vec())
                                  .set_stride(dO.strides().vec()));
  auto [Dq, Dk, Dv] = mha_graph->sdpa_backward(
      Q_, K_, V_, O_, Do, Stats, sdpa_backward_options);
  Dq->set_uid(DQ);
  Dq->set_output(true).set_dim(dQ.sizes().vec()).set_stride(dQ.strides().vec());
  Dk->set_uid(DK);
  Dk->set_output(true).set_dim(dK.sizes().vec()).set_stride(dK.strides().vec());
  Dv->set_uid(DV);
  Dv->set_output(true).set_dim(dV.sizes().vec()).set_stride(dV.strides().vec());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));
  return mha_graph;
}

auto build_graph_backward_nestedtensor(
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
  auto mha_graph = std::make_shared<fe::graph::Graph>();
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

  std::optional<std::shared_ptr<fe::graph::Tensor_attributes>> bias;
  if (attn_bias.has_value()) {
    TORCH_CHECK(
        false,
        "attn_bias not yet supportd with cuDNN Attention and NestedTensor");
    bias =
        mha_graph->tensor(fe::graph::Tensor_attributes()
                              .set_uid(BIAS)
                              .set_name("bias")
                              .set_dim(attn_bias.value().sizes().vec())
                              .set_stride(attn_bias.value().strides().vec()));
    sdpa_backward_options.set_bias(bias.value());
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
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  auto _dropoutseed = dropoutseed;
  auto _dropoutoffset = dropoutoffset;
  // cuDNN dropout bug requires these to be in int64
  if (dprops->major == 10 && dprops->minor == 0) {
    _dropoutseed = dropoutseed.to(kLong);
    _dropoutoffset = dropoutoffset.to(kLong);
  }

  cudnnHandle_t handle = getCudnnHandle();
  if (!o.defined()) {
    // q is passed to us in BHSD dim order
    alloc_with_matching_layout(q, o, {b, h, s_q, d_v});
  }

  if (return_softmaxstats && !softmaxstats.defined()) {
    // TODO(eqy): verify that this is correct
    softmaxstats = at::empty({b, h, s_q}, q.options().dtype(kFloat));
  }

  // do nothing if we got 0-element tensors
  if (!q.numel() || !k.numel() || !v.numel()) {
    return;
  }

  auto key = MHACacheKeyWrapper(
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
      return_softmaxstats);
  auto graph_ptr = mhagraphcache.find(key);
  std::shared_ptr<fe::graph::Graph> mha_graph;
  if (graph_ptr) {
    mha_graph = *graph_ptr;
  } else {
    mha_graph = build_graph(
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
  std::unordered_map<int64_t, void*> variant_pack = {
      {Q, q.data_ptr()},
      {K, k.data_ptr()},
      {V, v.data_ptr()},
      {SCALE, &scaling_factor},
      {O, o.data_ptr()}};
  if (return_softmaxstats) {
    variant_pack[LSE] = softmaxstats.data_ptr();
  }
  if (attn_bias.has_value()) {
    variant_pack[BIAS] = attn_bias.value().data_ptr();
  }
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = _dropoutseed.data_ptr();
    variant_pack[OFFSET] = _dropoutoffset.data_ptr();
  }
  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(
      mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
  mhagraphcache.update(key, mha_graph);
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
  auto mha_graph = build_graph_nestedtensor(
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
  auto seqlen_q = at::diff(cum_seqlen_q, 1, 0);
  auto seqlen_kv = at::diff(cum_seqlen_kv, 1, 0);
  auto rag_q_off = cum_seqlen_q.mul(h_q * d_qk);
  auto rag_k_off = cum_seqlen_kv.mul(h_k * d_v);
  auto rag_v_off = cum_seqlen_kv.mul(h_v * d_v);
  auto rag_stats_off = cum_seqlen_q.mul(h_q);
  std::unordered_map<int64_t, void*> variant_pack = {
      {Q, q.data_ptr()},
      {K, k.data_ptr()},
      {V, v.data_ptr()},
      {SCALE, &scaling_factor},
      {O, o.data_ptr()},
      {RAG_Q_OFF, rag_q_off.data_ptr()},
      {RAG_O_OFF, rag_q_off.data_ptr()},
      {RAG_K_OFF, rag_k_off.data_ptr()},
      {RAG_V_OFF, rag_v_off.data_ptr()},
      {SEQ_LEN_Q, seqlen_q.data_ptr()},
      {SEQ_LEN_KV, seqlen_kv.data_ptr()}};
  if (return_softmaxstats) {
    variant_pack[LSE] = softmaxstats.data_ptr();
    variant_pack[RAG_LSE_OFF] = rag_stats_off.data_ptr();
  }
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = dropoutseed.data_ptr();
    variant_pack[OFFSET] = dropoutoffset.data_ptr();
  }
  if (attn_bias.has_value()) {
    TORCH_CHECK("bias not supported with nestedtensor");
  }
  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(
      mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
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
  if (innermost_dO_stride != 1) {
    permute_to_matching_layout(o, dO_);
  }
#endif
  cudnnHandle_t handle = getCudnnHandle();
  auto key = MHACacheKeyWrapper(
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
      true);
  auto graph_backward_ptr = mhagraphbackwardcache.find(key);
  std::shared_ptr<fe::graph::Graph> mha_graph;
  if (graph_backward_ptr) {
    mha_graph = *graph_backward_ptr;
  } else {
    mha_graph = build_graph_backward(
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
  std::unordered_map<int64_t, void*> variant_pack = {
      // inputs
      {Q, q.data_ptr()},
      {K, k.data_ptr()},
      {V, v.data_ptr()},
      {O, o.data_ptr()},
      {DO, dO_.data_ptr()},
      {LSE, softmaxstats.data_ptr()},
      // outputs
      {DQ, dQ.data_ptr()},
      {DK, dK.data_ptr()},
      {DV, dV.data_ptr()},
      {SCALE, &scaling_factor}};
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = _dropoutseed.data_ptr();
    variant_pack[OFFSET] = _dropoutoffset.data_ptr();
  }
  if (attn_bias.has_value()) {
    variant_pack[BIAS] = attn_bias.value().data_ptr();
  }
  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  TORCH_CHECK(
      mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
  mhagraphbackwardcache.update(key, mha_graph);
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
  auto rag_q_off = cum_seqlen_q.mul(h_q * d_qk);
  auto rag_k_off = cum_seqlen_kv.mul(h_k * d_v);
  auto rag_v_off = cum_seqlen_kv.mul(h_v * d_v);
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

  auto mha_graph = build_graph_backward_nestedtensor(
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

  std::unordered_map<int64_t, void*> variant_pack = {
      // inputs
      {Q, q.data_ptr()},
      {K, k.data_ptr()},
      {V, v.data_ptr()},
      {O, o.data_ptr()},
      {DO, dO_.data_ptr()},
      {LSE, softmaxstats.data_ptr()},
      // outputs
      {DQ, dQ.data_ptr()},
      {DK, dK.data_ptr()},
      {DV, dV.data_ptr()},
      {SCALE, &scaling_factor},
      {RAG_Q_OFF, rag_q_off.data_ptr()},
      {RAG_O_OFF, rag_q_off.data_ptr()},
      {RAG_K_OFF, rag_k_off.data_ptr()},
      {RAG_V_OFF, rag_v_off.data_ptr()},
      {RAG_LSE_OFF, rag_stats_off.data_ptr()},
      {SEQ_LEN_Q, seqlen_q.data_ptr()},
      {SEQ_LEN_KV, seqlen_kv.data_ptr()}};
  if (dropout_probability != 0.0f) {
    variant_pack[SEED] = _dropoutseed.data_ptr();
    variant_pack[OFFSET] = _dropoutoffset.data_ptr();
  }
  TORCH_CHECK(
      !attn_bias.has_value(),
      "attn_bias not yet supportd with cuDNN Attention and NestedTensor");

  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  TORCH_CHECK(
      mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
}

} // namespace native
} // namespace at
#endif
