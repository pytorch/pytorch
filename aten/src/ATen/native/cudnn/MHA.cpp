#include <ATen/ATen.h>
//#ifndef AT_PER_OPERATOR_HEADERS
//#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

}} // namespace at::native

#else // AT_CUDNN_ENABLED
#include <ATen/native/cudnn/MHA.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <ATen/cuda/Exceptions.h>
#include <cudnn_frontend.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cudnn.h>

#include <iostream>

namespace at { namespace native {


#if (CUDNN_VERSION >= 8900)
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;
using graph_and_tensors = std::tuple<
                                    std::shared_ptr<fe::graph::Graph>,                    
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // Q,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // K,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // V,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // Attn_scale,
                                    //std::shared_ptr<fe::graph::Tensor_attributes>, // Bias,
                                    //std::shared_ptr<fe::graph::Tensor_attributes>, // SEQ_LEN_Q,
                                    //std::shared_ptr<fe::graph::Tensor_attributes>, // SEQ_LEN_KV,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // Seed,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // Offset,
                                    //std::shared_ptr<fe::graph::Tensor_attributes>, // Dropout_mask,
                                    //std::shared_ptr<fe::graph::Tensor_attributes>, // Dropout_scale
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // O
                                    std::shared_ptr<fe::graph::Tensor_attributes> // Stats
                        >;

#define MAX_MHA_DIM 4

struct MHAParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  int q_dim[MAX_MHA_DIM];
  int k_dim[MAX_MHA_DIM];
  int v_dim[MAX_MHA_DIM];
  int q_stride[MAX_MHA_DIM];
  int k_stride[MAX_MHA_DIM];
  int v_stride[MAX_MHA_DIM];
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d;
  double dropout_probability;
  bool is_causal;
  bool return_softmaxstats;
};

void setMHAParams(MHAParams& params, int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, const Tensor& q, const Tensor& k, const Tensor & v, double dropout_probability, bool is_causal, bool return_softmaxstats) {
  memset(&params, 0, sizeof(MHAParams));
  params.device_id = at::cuda::current_device();
  params.dataType = fe::DataType_t::HALF;
  if (q.scalar_type() == kBFloat16) {
    params.dataType = fe::DataType_t::BFLOAT16;
  }
  params.b = b;
  params.h = h;
  params.d = d;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.dropout_probability = dropout_probability;
  params.is_causal = is_causal;
  params.return_softmaxstats = return_softmaxstats;
  TORCH_INTERNAL_ASSERT(q.sizes().size() == MAX_MHA_DIM, "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(q.strides().size() == MAX_MHA_DIM, "Q tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(k.sizes().size() == MAX_MHA_DIM, "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(k.strides().size() == MAX_MHA_DIM, "K tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(v.sizes().size() == MAX_MHA_DIM, "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(v.strides().size() == MAX_MHA_DIM, "V tensor has unexpected number of dims, please report a bug to PyTorch.");
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim);
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride);
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim);
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride);
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim);
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride);
}

struct MHACacheKeyWrapper : ParamsWrapper<MHAParams> {
  MHACacheKeyWrapper(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, const Tensor& q, const Tensor& k, const Tensor & v, double dropout_probability, bool is_causal, bool return_softmaxstats) {
    setMHAParams(this->pod, b, h, s_q, s_kv, d, q, k, v, dropout_probability, is_causal, return_softmaxstats);
  }
};

template <typename T, typename KeyType>
struct MHAGraphCache {
std::unordered_map<KeyType, graph_and_tensors, ParamsWrapperHash<KeyType>> engine_cache;

// no mutexes here as caches are now thread local for v8, can also return a pointer
// to the Execution Plan if we know it will not be invalidated by another thread
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

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to be thread safe across all engines
// see Limitations in https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
thread_local MHAGraphCache<graph_and_tensors, MHACacheKeyWrapper> mhagraphcache;

auto build_graph_and_tensors(int64_t b,
                                int64_t h,
                                int64_t s_q,
                                int64_t s_kv,
                                int64_t d,
                                float scaling_factor,
                                bool return_softmaxstats,
		                        bool is_causal,
                                double dropout_probability,
                                const Tensor& q,
                                const Tensor& k,
                                const Tensor& v,
                                Tensor& softmaxstats,
                                Tensor& o,
                                Tensor& dropoutseed,
                                Tensor& dropoutoffset,
                                cudnnHandle_t& handle,
                                MHAParams& params) {
    auto dtype = fe::DataType_t::HALF;
    if (q.scalar_type() == kBFloat16) {
      dtype = fe::DataType_t::BFLOAT16;
    }
    auto mha_graph = std::make_shared<fe::graph::Graph>();
    mha_graph->set_io_data_type(dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    //std::vector<int64_t> q_dim;
    //std::vector<int64_t> q_stride;
    //std::vector<int64_t> k_dim;
    //std::vector<int64_t> k_stride;
    //std::vector<int64_t> v_dim;
    //std::vector<int64_t> v_stride;
    //q_dim.assign(q.sizes().data(), q.sizes().data() + q.sizes().size());
    //q_stride.assign(q.strides().data(), q.strides().data() + q.strides().size());
    //k_dim.assign(k.sizes().data(), k.sizes().data() + k.sizes().size());
    //k_stride.assign(k.strides().data(), k.strides().data() + k.strides().size());
    //v_dim.assign(v.sizes().data(), v.sizes().data() + v.sizes().size());
    //v_stride.assign(v.strides().data(), v.strides().data() + v.strides().size());
    //std::cout << q.sizes() << q.strides() << k.sizes() << k.strides() << v.sizes() << v.strides() << std::endl;
    auto Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(std::vector<int64_t>(params.q_dim, params.q_dim+MAX_MHA_DIM))
                                  .set_stride(std::vector<int64_t>(params.q_stride, params.q_stride+MAX_MHA_DIM)));
    //std::cout << "q stride: " << q.strides() << std::endl;
    //for (auto it = q_stride.begin(); it != q_stride.end(); it++) std::cout << *it << std::endl;
    //std::cout << "k stride: " << k.strides() << std::endl;
    //for (auto it = k_stride.begin(); it != k_stride.end(); it++) std::cout << *it << std::endl;
    //std::cout << "v stride: " << v.strides() << std::endl;
    //for (auto it = v_stride.begin(); it != v_stride.end(); it++) std::cout << *it << std::endl;

    auto K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(std::vector<int64_t>(params.k_dim, params.k_dim+MAX_MHA_DIM))
                                  .set_stride(std::vector<int64_t>(params.k_stride, params.k_stride+MAX_MHA_DIM)));
    auto V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(std::vector<int64_t>(params.v_dim, params.v_dim+MAX_MHA_DIM))
                                  .set_stride(std::vector<int64_t>(params.v_stride, params.v_stride+MAX_MHA_DIM)));
    auto attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("attn_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_is_pass_by_value(true)
                                       .set_data_type(fe::DataType_t::FLOAT));
    //auto bias = mha_graph->tensor(fe::graph::Tensor_attributes()
    //                         .set_name("bias")
    //                         .set_dim({b, 1, s_q, s_kv})
    //                         .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seed")
                                     .set_dim({1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
    auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("Offset")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
    auto scaled_dot_product_flash_attention_options = fe::graph::Scaled_dot_product_flash_attention_attributes()
                                                          .set_name("flash_attention")
                                                          .set_is_inference(return_softmaxstats == false)
                                                          .set_causal_mask(is_causal)
                                                          .set_attn_scale(attn_scale)
                                                          .set_dropout(dropout_probability, seed, offset);
    // Optional bias in flash attention is only supported 8.9.3 onwards
    if (cudnnGetVersion() >= 8904) {
        //scaled_dot_product_flash_attention_options.set_alibi_mask(true);
    }

    auto seq_q  = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("seq_q")
                                    .set_dim({b, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("seq_kv")
                                    .set_dim({b, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    //if (cudnnGetVersion() >= 8903) {
    //    scaled_dot_product_flash_attention_options.set_bias(bias)
    //        .set_padding_mask(true)
    //        .set_seq_len_q(seq_q)
    //        .set_seq_len_kv(seq_kv);
    //}


    auto [O, Stats] = mha_graph->scaled_dot_product_flash_attention(Q, K, V, scaled_dot_product_flash_attention_options);

    //O->set_output(true).set_stride({h * d, d, b * h * d, 1});
    // std::vector<int64_t> o_stride;
    // o_stride.assign(o.strides().data(), o.strides().data() + o.strides().size());
    // std::cout << "out stride set: " << h*d << " " << d << " " << b * h * d << " " << 1 << std::endl;
    //std::cout << "tensor stride: " << o.strides() << std::endl;
    O->set_output(true).set_stride(std::vector<int64_t>(o.strides().data(), o.strides().data() + o.strides().size()));

    // Check that Stats tensor is real, which is only when its training step
    if (Stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    TORCH_INTERNAL_ASSERT(mha_graph->validate().is_good());

    TORCH_INTERNAL_ASSERT(mha_graph->build_operation_graph(handle).is_good());

    auto plans = mha_graph->get_execution_plan_list({fe::HeurMode_t::A});


    TORCH_INTERNAL_ASSERT(plans.check_support(handle).is_good());

    TORCH_INTERNAL_ASSERT(mha_graph->set_execution_plans(plans).is_good());
    return std::make_tuple(mha_graph, Q, K, V, attn_scale, seed, offset, O, Stats);
}

void
run_cudnn_LLM_fprop(int64_t b,
                    int64_t h,
                    int64_t s_q,
                    int64_t s_kv,
                    int64_t d,
                    float scaling_factor,
                    bool return_softmaxstats,
		            bool is_causal,
                    double dropout_probability,
                    const Tensor& q,
                    const Tensor& k,
                    const Tensor& v,
                    Tensor& softmaxstats,
                    Tensor& o,
                    Tensor& dropoutseed,
                    Tensor& dropoutoffset) {
    std::cout << "running cuDNN" << std::endl;
    cudnnHandle_t handle = getCudnnHandle();
    o = at::empty_strided({b, h, s_q, d}, {s_q * h * d, d, h * d, 1}, q.options());
    if (return_softmaxstats) {
      // TODO(eqy): fix strides
      softmaxstats = at::empty({b, h, s_q}, q.options());
    }
    //auto dtype = fe::DataType_t::HALF;
    //fe::graph::Graph mha_graph;
    //mha_graph.set_io_data_type(dtype)
    //    .set_intermediate_data_type(fe::DataType_t::FLOAT)
    //    .set_compute_data_type(fe::DataType_t::FLOAT);

    //std::vector<int64_t> q_dim;
    //std::vector<int64_t> q_stride;
    //std::vector<int64_t> k_dim;
    //std::vector<int64_t> k_stride;
    //std::vector<int64_t> v_dim;
    //std::vector<int64_t> v_stride;
    //q_dim.assign(q.sizes().data(), q.sizes().data() + q.sizes().size());
    //q_stride.assign(q.strides().data(), q.strides().data() + q.strides().size());
    //k_dim.assign(k.sizes().data(), k.sizes().data() + k.sizes().size());
    //k_stride.assign(k.strides().data(), k.strides().data() + k.strides().size());
    //v_dim.assign(v.sizes().data(), v.sizes().data() + v.sizes().size());
    //v_stride.assign(v.strides().data(), v.strides().data() + v.strides().size());
    //std::cout << q.sizes() << q.strides() << k.sizes() << k.strides() << v.sizes() << v.strides() << std::endl;
    //auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                              .set_name("Q")
    //                              .set_dim(q_dim)
    //                              .set_stride(q_stride));
    //std::cout << "q stride: " << q.strides() << std::endl;
    //for (auto it = q_stride.begin(); it != q_stride.end(); it++) std::cout << *it << std::endl;
    //std::cout << "k stride: " << k.strides() << std::endl;
    //for (auto it = k_stride.begin(); it != k_stride.end(); it++) std::cout << *it << std::endl;
    //std::cout << "v stride: " << v.strides() << std::endl;
    //for (auto it = v_stride.begin(); it != v_stride.end(); it++) std::cout << *it << std::endl;

    //auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                              .set_name("K")
    //                              .set_dim(k_dim)
    //                              .set_stride(k_stride));
    //auto V = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                              .set_name("V")
    //                              .set_dim(v_dim)
    //                              .set_stride(v_stride));
    //auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                                   .set_name("attn_scale")
    //                                   .set_dim({1, 1, 1, 1})
    //                                   .set_stride({1, 1, 1, 1})
    //                                   .set_is_pass_by_value(true)
    //                                   .set_data_type(fe::DataType_t::FLOAT));
    ////auto bias = mha_graph.tensor(fe::graph::Tensor_attributes()
    ////                         .set_name("bias")
    ////                         .set_dim({b, 1, s_q, s_kv})
    ////                         .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    //auto seed = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                                 .set_name("Seed")
    //                                 .set_dim({1, 1, 1, 1})
    //                                 .set_stride({1, 1, 1, 1})
    //                                 .set_data_type(fe::DataType_t::INT32));
    //auto offset = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                                   .set_name("Offset")
    //                                   .set_dim({1, 1, 1, 1})
    //                                   .set_stride({1, 1, 1, 1})
    //                                   .set_data_type(fe::DataType_t::INT32));
    //auto scaled_dot_product_flash_attention_options = fe::graph::Scaled_dot_product_flash_attention_attributes()
    //                                                      .set_name("flash_attention")
    //                                                      .set_is_inference(return_softmaxstats == false)
    //                                                      .set_causal_mask(is_causal)
    //                                                      .set_attn_scale(attn_scale)
    //                                                      .set_dropout(dropout_probability, seed, offset);
    //// Optional bias in flash attention is only supported 8.9.3 onwards
    //if (cudnnGetVersion() >= 8904) {
    //    //scaled_dot_product_flash_attention_options.set_alibi_mask(true);
    //}

    //auto seq_q  = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                                .set_name("seq_q")
    //                                .set_dim({b, 1, 1, 1})
    //                                .set_stride({1, 1, 1, 1})
    //                                .set_data_type(fe::DataType_t::INT32));
    //auto seq_kv = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                                .set_name("seq_kv")
    //                                .set_dim({b, 1, 1, 1})
    //                                .set_stride({1, 1, 1, 1})
    //                                .set_data_type(fe::DataType_t::INT32));
    ////if (cudnnGetVersion() >= 8903) {
    ////    scaled_dot_product_flash_attention_options.set_bias(bias)
    ////        .set_padding_mask(true)
    ////        .set_seq_len_q(seq_q)
    ////        .set_seq_len_kv(seq_kv);
    ////}


    //auto [O, Stats] = mha_graph.scaled_dot_product_flash_attention(Q, K, V, scaled_dot_product_flash_attention_options);

    ////O->set_output(true).set_stride({h * d, d, b * h * d, 1});
    //std::vector<int64_t> o_stride;
    //o_stride.assign(o.strides().data(), o.strides().data() + o.strides().size());
    //std::cout << "out stride set: " << h*d << " " << d << " " << b * h * d << " " << 1 << std::endl;
    //std::cout << "tensor stride: " << o.strides() << std::endl;
    //O->set_output(true).set_stride(o_stride);

    //// Check that Stats tensor is real, which is only when its training step
    //if (Stats) {
    //    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    //}

    //TORCH_INTERNAL_ASSERT(mha_graph.validate().is_good());

    //TORCH_INTERNAL_ASSERT(mha_graph.build_operation_graph(handle).is_good());

    //auto plans = mha_graph.get_execution_plan_list({fe::HeurMode_t::A});


    //TORCH_INTERNAL_ASSERT(plans.check_support(handle).is_good());

    //TORCH_INTERNAL_ASSERT(mha_graph.set_execution_plans(plans).is_good());
    
    auto key = MHACacheKeyWrapper(b, h, s_q, s_kv, d, q, k, v, dropout_probability, is_causal, return_softmaxstats);
    auto graph_and_tensors_ptr = mhagraphcache.find(key);
    graph_and_tensors graph_and_tensors_values;
    if (graph_and_tensors_ptr) {
        std::cout << "cache hit" << std::endl;
        graph_and_tensors_values = *graph_and_tensors_ptr;
    } else {
        std::cout << "cache miss" << std::endl;
        graph_and_tensors_values = build_graph_and_tensors(b,
                                 h,
                                 s_q,
                                 s_kv,
                                 d,
                                 scaling_factor,
                                 return_softmaxstats,
                                 is_causal,
                                 dropout_probability,
                                 q,
                                 k,
                                 v,
                                 softmaxstats,
                                 o,
                                 dropoutseed,
                                 dropoutoffset,
                                 handle,
                                 key.pod);
    }
    auto [mha_graph, Q, K, V, attn_scale, seed, offset, O, Stats] = graph_and_tensors_values;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = { {Q, q.data_ptr()},
         {K, k.data_ptr()},
         {V, v.data_ptr()},
         {attn_scale, &scaling_factor},
         //{bias, bias.data_ptr()},
         {seed, dropoutseed.data_ptr()},
         {offset, dropoutoffset.data_ptr()},
         {O, o.data_ptr()}};
    if (return_softmaxstats) {
        variant_pack[Stats] = softmaxstats.data_ptr();
    }
    auto workspace_size = mha_graph->get_workspace_size();
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    TORCH_INTERNAL_ASSERT(mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());
    mhagraphcache.update(key, graph_and_tensors_values);
}


}} // namespace at::native

#endif
#endif
