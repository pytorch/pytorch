#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>

#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/cudnn/RMSNorm_v8.h>
#include <ATen/native/utils/ParamsHash.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/env.h>

#include <list>
#include <unordered_map>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

#include <cudnn_frontend.h>

namespace at {
namespace native {

auto get_fe_dtype_rmsnorm(const Tensor& t) {
  namespace fe = cudnn_frontend;
  auto dtype = t.scalar_type();
  auto fe_dtype = fe::DataType_t::FLOAT;
  if (dtype == at::ScalarType::Half) {
    fe_dtype = fe::DataType_t::HALF;
  } else if (dtype == at::ScalarType::BFloat16) {
    fe_dtype = fe::DataType_t::BFLOAT16;
  } else {
    TORCH_INTERNAL_ASSERT("cuDNN rmsnorm got unsupported dtype", dtype);
  }
  return fe_dtype;
}

namespace {
namespace fe = cudnn_frontend;
using graph_and_tensors_forward = std::tuple<
    std::shared_ptr<fe::graph::Graph>,
    std::shared_ptr<fe::graph::Tensor_attributes>, // X,
    std::shared_ptr<fe::graph::Tensor_attributes>, // inv_var,
    std::shared_ptr<fe::graph::Tensor_attributes>, // scale,
    std::shared_ptr<fe::graph::Tensor_attributes> // Y
    >;
using graph_and_tensors_backward = std::tuple<
    std::shared_ptr<fe::graph::Graph>,
    std::shared_ptr<fe::graph::Tensor_attributes>, // X,
    std::shared_ptr<fe::graph::Tensor_attributes>, // DY,
    std::shared_ptr<fe::graph::Tensor_attributes>, // inv_variance,
    std::shared_ptr<fe::graph::Tensor_attributes>, // scale,
    std::shared_ptr<fe::graph::Tensor_attributes>, // dscale,
    std::shared_ptr<fe::graph::Tensor_attributes> // DX
    >;

struct RMSNormParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  int64_t M;
  int64_t N;
};

void setRMSNormParams(
    RMSNormParams& params,
    const Tensor& X,
    int64_t M,
    int64_t N) {
  std::memset(&params, 0, sizeof(params));
  params.device_id = at::cuda::current_device();
  params.dataType = get_fe_dtype_rmsnorm(X);
  params.M = M;
  params.N = N;
}

struct RMSNormCacheKeyWrapper : ParamsWrapper<RMSNormParams> {
  RMSNormCacheKeyWrapper(const Tensor& X, int64_t M, int64_t N) {
    setRMSNormParams(this->pod, X, M, N);
  }
};

template <typename T, typename KeyType>
struct RMSNormGraphCache {
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

  template <typename U>
  void update(const KeyType& key, U&& results) {
    engine_cache.erase(key);
    engine_cache.emplace(key, std::forward<U>(results));
  }
};

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to
// be thread safe across all engines see Limitations in
// https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
thread_local RMSNormGraphCache<
    graph_and_tensors_forward,
    RMSNormCacheKeyWrapper>
    rmsnorm_forward_graph_cache;
thread_local RMSNormGraphCache<
    graph_and_tensors_backward,
    RMSNormCacheKeyWrapper>
    rmsnorm_backward_graph_cache;

} // namespace

void raw_cudnn_rmsnorm_forward_out(
    const Tensor& X,
    const Tensor& scale,
    float epsilon,
    Tensor* rstd,
    Tensor* Y,
    int64_t M,
    int64_t N) {
  namespace fe = cudnn_frontend;
  auto key = RMSNormCacheKeyWrapper(X, M, N);
  auto graph_and_tensors_forward_ptr = rmsnorm_forward_graph_cache.find(key);
  auto rmsnorm_graph = std::make_shared<fe::graph::Graph>();
  graph_and_tensors_forward graph_and_tensors_forward_values;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack;
  if (graph_and_tensors_forward_ptr) {
    auto [graph, X_fe, inv_variance_fe, scale_fe, Y_fe] =
        *graph_and_tensors_forward_ptr;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {inv_variance_fe, rstd->data_ptr()},
            {scale_fe, scale.data_ptr()},
            {Y_fe, Y->data_ptr()}};
    variant_pack = std::move(variant_pack_);
    rmsnorm_graph = std::move(graph);
  } else {
    rmsnorm_graph->set_io_data_type(get_fe_dtype_rmsnorm(X))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    // cuDNN only seems to care about non-normalized and normalized dimensions,
    // and we can only have one non-normalized-dimension, so... we reshape to M,
    // N, 1, 1 because cuDNN also has the restriction that everything must be in
    // 4-D
    auto X_reshaped = X.reshape({M, N, 1, 1});
    auto X_fe = rmsnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("X")
            .set_dim(std::vector<int64_t>(
                X_reshaped.sizes().begin(), X_reshaped.sizes().end()))
            .set_stride(std::vector<int64_t>(
                X_reshaped.strides().begin(), X_reshaped.strides().end())));
    auto scale_fe =
        rmsnorm_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, N, 1, 1})
                                  .set_stride({N, 1, N, N})
                                  .set_data_type(get_fe_dtype_rmsnorm(scale)));
    auto epsilon_fe = rmsnorm_graph->tensor(epsilon);
    auto rmsnorm_options = fe::graph::Rmsnorm_attributes()
                               .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                               .set_epsilon(epsilon_fe);
    auto [Y_fe, inv_variance_fe] =
        rmsnorm_graph->rmsnorm(X_fe, scale_fe, rmsnorm_options);
    inv_variance_fe->set_output(true).set_data_type(
        get_fe_dtype_rmsnorm(*rstd));
    Y_fe->set_output(true);

    cudnnHandle_t handle = getCudnnHandle();
    TORCH_INTERNAL_ASSERT(rmsnorm_graph->validate().is_good());
    TORCH_INTERNAL_ASSERT(
        rmsnorm_graph->build_operation_graph(handle).is_good());
    TORCH_INTERNAL_ASSERT(rmsnorm_graph
                              ->create_execution_plans(
                                  {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
                              .is_good());
    TORCH_INTERNAL_ASSERT(
        rmsnorm_graph->check_support(handle).is_good(),
        rmsnorm_graph->check_support(handle).get_message());
    TORCH_INTERNAL_ASSERT(rmsnorm_graph->build_plans(handle).is_good());
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {inv_variance_fe, rstd->data_ptr()},
            {scale_fe, scale.data_ptr()},
            {Y_fe, Y->data_ptr()}};
    variant_pack = std::move(variant_pack_);
    auto result =
        std::make_tuple(rmsnorm_graph, X_fe, inv_variance_fe, scale_fe, Y_fe);
    rmsnorm_forward_graph_cache.update(key, std::move(result));
  }
  cudnnHandle_t handle = getCudnnHandle();
  size_t workspace_size = rmsnorm_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_INTERNAL_ASSERT(!workspace_size || workspace_ptr);
  TORCH_INTERNAL_ASSERT(
      rmsnorm_graph->execute(handle, variant_pack, workspace_ptr.get())
          .is_good());
}

void raw_cudnn_rmsnorm_backward_out(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma) {
  namespace fe = cudnn_frontend;
  auto key = RMSNormCacheKeyWrapper(X, M, N);
  auto graph_and_tensors_backward_ptr = rmsnorm_backward_graph_cache.find(key);
  auto rmsnorm_graph = std::make_shared<fe::graph::Graph>();
  graph_and_tensors_backward graph_and_tensors_backward_values;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack;
  if (graph_and_tensors_backward_ptr) {
    auto [graph, X_fe, DY_fe, inv_variance_fe, scale_fe, dscale_fe, DX_fe] =
        *graph_and_tensors_backward_ptr;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {DY_fe, dY.data_ptr()},
            {inv_variance_fe, rstd.data_ptr()},
            {scale_fe, gamma.data_ptr()},
            {dscale_fe, dgamma->data_ptr()},
            {DX_fe, dX->data_ptr()}};
    variant_pack = std::move(variant_pack_);
    rmsnorm_graph = std::move(graph);
  } else {
    rmsnorm_graph->set_io_data_type(get_fe_dtype_rmsnorm(X))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    // cuDNN only seems to care about non-normalized and normalized dimensions,
    // and we can only have one non-normalized-dimension, so... we reshape to M,
    // N, 1, 1 because cuDNN also has the restriction that everything must be in
    // 4-D
    auto X_reshaped = X.reshape({M, N, 1, 1});
    auto DY_reshaped = dY.reshape({M, N, 1, 1});
    auto X_fe = rmsnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("X")
            .set_dim(std::vector<int64_t>(
                X_reshaped.sizes().begin(), X_reshaped.sizes().end()))
            .set_stride({N, 1, N, N}));
    auto DY_fe = rmsnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("DY")
            .set_dim(std::vector<int64_t>(
                DY_reshaped.sizes().begin(), DY_reshaped.sizes().end()))
            .set_stride({N, 1, N, N}));
    auto scale_fe =
        rmsnorm_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, N, 1, 1})
                                  .set_stride({N, 1, N, N})
                                  .set_data_type(get_fe_dtype_rmsnorm(gamma)));
    auto inv_variance_fe =
        rmsnorm_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("inv_variance")
                                  .set_dim({M, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(get_fe_dtype_rmsnorm(rstd)));
    auto rmsnorm_options =
        fe::graph::Rmsnorm_backward_attributes().has_dbias(false);
    auto [DX_fe, dscale_fe, dbias_fe] = rmsnorm_graph->rmsnorm_backward(
        DY_fe, X_fe, scale_fe, inv_variance_fe, rmsnorm_options);
    DX_fe->set_output(true);
    dscale_fe->set_output(true).set_data_type(get_fe_dtype_rmsnorm(*dgamma));
    // dbias_fe should be nullptr for RMSNorm
    TORCH_INTERNAL_ASSERT(
        dbias_fe == nullptr, "RMSNorm backward should not have dbias");

    cudnnHandle_t handle = getCudnnHandle();
    TORCH_INTERNAL_ASSERT(rmsnorm_graph->validate().is_good());
    TORCH_INTERNAL_ASSERT(
        rmsnorm_graph->build_operation_graph(handle).is_good());
    TORCH_INTERNAL_ASSERT(rmsnorm_graph
                              ->create_execution_plans(
                                  {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
                              .is_good());
    TORCH_INTERNAL_ASSERT(
        rmsnorm_graph->check_support(handle).is_good(),
        rmsnorm_graph->check_support(handle).get_message());
    TORCH_INTERNAL_ASSERT(rmsnorm_graph->build_plans(handle).is_good());
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {DY_fe, dY.data_ptr()},
            {inv_variance_fe, rstd.data_ptr()},
            {scale_fe, gamma.data_ptr()},
            {dscale_fe, dgamma->data_ptr()},
            {DX_fe, dX->data_ptr()}};
    variant_pack = std::move(variant_pack_);
    auto result = std::make_tuple(
        rmsnorm_graph,
        X_fe,
        DY_fe,
        inv_variance_fe,
        scale_fe,
        dscale_fe,
        DX_fe);
    rmsnorm_backward_graph_cache.update(key, std::move(result));
  }
  cudnnHandle_t handle = getCudnnHandle();
  size_t workspace_size = rmsnorm_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_INTERNAL_ASSERT(!workspace_size || workspace_ptr);
  TORCH_INTERNAL_ASSERT(
      rmsnorm_graph->execute(handle, variant_pack, workspace_ptr.get())
          .is_good());
}

} // namespace native
} // namespace at

#endif // AT_CUDNN_ENABLED
