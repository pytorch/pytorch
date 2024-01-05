#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>

#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/cudnn/LayerNorm_v8.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

#include <c10/util/env.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <unordered_map>
#include <list>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

#include <cudnn_frontend.h>

namespace at { namespace native {

 

auto get_fe_dtype(const Tensor& t) {
  namespace fe = cudnn_frontend;
  auto dtype = t.scalar_type();
  auto fe_dtype = fe::DataType_t::FLOAT;
  if (dtype == at::ScalarType::Half) {
    fe_dtype = fe::DataType_t::HALF;
  } else if (dtype == at::ScalarType::BFloat16) {
    fe_dtype = fe::DataType_t::BFLOAT16;
  } else {
    TORCH_INTERNAL_ASSERT("cuDNN layernorm got unsupported dtype", dtype);
  }
  return fe_dtype;
}

 namespace { 
 namespace fe = cudnn_frontend;
using graph_and_tensors = std::tuple<
                                    std::shared_ptr<fe::graph::Graph>,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // X,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // mean,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // inv_var,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // scale,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // bias,
                                    std::shared_ptr<fe::graph::Tensor_attributes>, // epsilon,
                                    std::shared_ptr<fe::graph::Tensor_attributes> // Y
                        >;

struct LayerNormParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  int64_t M;
  int64_t N; 
};

void setLayerNormParams(LayerNormParams& params, const Tensor& X, int64_t M, int64_t N) {
  memset(&params, 0, sizeof(params));
  params.device_id = at::cuda::current_device();
  params.dataType = get_fe_dtype(X);
}

struct LayerNormCacheKeyWrapper : ParamsWrapper<LayerNormParams> {
  LayerNormCacheKeyWrapper(const Tensor& X, int64_t M, int64_t N) {
    setLayerNormParams(this->pod, X, M, N);
  }
};

 template <typename T, typename KeyType>
struct LayerNormGraphCache {
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

 }

void raw_cudnn_layernorm_forward_out(const Tensor& X, const Tensor& scale, const Tensor& bias, float epsilon, Tensor* mean, Tensor* rstd, Tensor* Y, int64_t M, int64_t N) {
  TORCH_WARN("called");
  namespace fe = cudnn_frontend;
  fe::graph::Graph graph;
  graph.set_io_data_type(get_fe_dtype(X))
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  TORCH_WARN(X.sizes(), scale.sizes(), bias.sizes(), mean->sizes(), rstd->sizes());
  // cuDNN only seems to care about non-normalized and normalized dimensions, and we can only have one non-normalized-dimension, so...
  // we reshape to
  // N, M, 1, 1 because cuDNN also has the restruction that everything must be in 4-D
  auto X_reshaped = X.reshape({M, N, 1, 1});
  auto X_fe = graph.tensor(fe::graph::Tensor_attributes()
                           .set_name("X")
                           .set_dim(std::vector<int64_t>(X_reshaped.sizes().begin(), X_reshaped.sizes().end()))
                           .set_stride(std::vector<int64_t>(X_reshaped.strides().begin(), X_reshaped.strides().end())));
  //std::vector<int64_t> scale_bias_sizes(X.sizes().size(), 1); // init to 1
  //std::vector<int64_t> scale_bias_strides(X.strides().size(), scale.sizes()[0]); // init to M
  //for (int i = scale_bias_sizes.size() - 1; i >= 0; i--) {
  //  if (scale.sizes()[0] == X.sizes()[i]) {
  //      scale_bias_sizes[i] = scale.sizes()[0];
  //      scale_bias_strides[i] = 1;
  //      break;
  //  }
  //}
  auto scale_fe = graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("scale")
                                .set_dim({1, N, 1, 1})
                                .set_stride({N, 1, N, N})
                                .set_data_type(get_fe_dtype(scale)));
  auto bias_fe  = bias.numel() ? graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("bias")
                               .set_dim({1, N, 1, 1})
                               .set_stride({N, 1, N, N})
                               .set_data_type(get_fe_dtype(bias))) : nullptr;
  auto epsilon_fe = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("epsilon")
                                  .set_dim({1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
  auto layernorm_options =
      fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon_fe);
  auto [Y_fe, mean_fe, inv_variance_fe] = graph.layernorm(X_fe, scale_fe, bias_fe, layernorm_options);
  mean_fe->set_output(true).set_data_type(get_fe_dtype(*mean));
  inv_variance_fe->set_output(true).set_data_type(get_fe_dtype(*rstd));

  Y_fe->set_output(true);


  cudnnHandle_t handle = getCudnnHandle();
  TORCH_INTERNAL_ASSERT(graph.validate().is_good());

  TORCH_INTERNAL_ASSERT(graph.build_operation_graph(handle).is_good());

  TORCH_INTERNAL_ASSERT(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

  TORCH_INTERNAL_ASSERT(graph.check_support(handle).is_good());

  TORCH_INTERNAL_ASSERT(graph.build_plans(handle).is_good());

  auto workspace_size = graph.get_workspace_size();
  auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_INTERNAL_ASSERT(!workspace_size || workspace_ptr);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {X_fe, X_reshaped.data_ptr()},
      {mean_fe, mean->data_ptr()},
      {inv_variance_fe, rstd->data_ptr()},
      {scale_fe, scale.data_ptr()},
      {bias_fe, bias.data_ptr()},
      {epsilon_fe, &epsilon},
      {Y_fe, Y->data_ptr()}};
  TORCH_INTERNAL_ASSERT(graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
}


}} // at::native

#endif  // AT_CUDNN_ENABLED
