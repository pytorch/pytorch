#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>

#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/BatchNorm.h>
#else
#include <ATen/native/cudnn/BatchNorm_v8.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double exponential_average_factor,
    double epsilon) {
  TORCH_CHECK(false, "cudnn_batch_norm: ATen not compiled with cuDNN support");
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> cudnn_batch_norm_out(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_var,
    Tensor& reserve) {
  AT_ERROR("cudnn_batch_norm_out: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt,
    const std::optional<Tensor>& save_var_opt,
    double epsilon,
    const Tensor& reservedSpace) {
  TORCH_CHECK(
      false, "cudnn_batch_norm_backward: ATen not compiled with cuDNN support");
}

size_t _get_cudnn_batch_norm_reserve_space_size(
    const Tensor& input_t,
    bool training) {
  TORCH_CHECK(
      false,
      "_get_cudnn_batch_norm_reserve_space_size: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/cudnn-wrapper.h>
#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
#include <cudnn_frontend.h>
C10_DIAGNOSTIC_POP()

#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/env.h>

#include <list>
#include <unordered_map>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_batch_norm_backward_native.h>
#include <ATen/ops/cudnn_batch_norm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at {
namespace native {

namespace {
namespace fe = cudnn_frontend;

auto get_fe_dtype(const Tensor& t) {
  auto dtype = t.scalar_type();
  auto fe_dtype = fe::DataType_t::FLOAT;
  if (dtype == at::ScalarType::Half) {
    fe_dtype = fe::DataType_t::HALF;
  } else if (dtype == at::ScalarType::BFloat16) {
    fe_dtype = fe::DataType_t::BFLOAT16;
  } else if (dtype == at::ScalarType::Float) {
    fe_dtype = fe::DataType_t::FLOAT;
  } else {
    TORCH_INTERNAL_ASSERT(false, "cuDNN batch norm got unsupported dtype: ", dtype);
  }
  return fe_dtype;
}

Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{1, t.numel()};
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

cudnnBatchNormMode_t getCudnnBatchNormMode(
    bool training,
    at::MemoryFormat memory_format,
    int64_t dim) {
  if (dim == 2) {
    return CUDNN_BATCHNORM_PER_ACTIVATION;
  } else if (training && memory_format == at::MemoryFormat::ChannelsLast) {
    return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  } else if (training && memory_format == at::MemoryFormat::ChannelsLast3d) {
    return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  } else {
    return CUDNN_BATCHNORM_SPATIAL;
  }
}

using graph_and_tensors_forward = std::tuple<
    std::shared_ptr<fe::graph::Graph>,
    std::shared_ptr<fe::graph::Tensor_attributes>, // X
    std::shared_ptr<fe::graph::Tensor_attributes>, // scale
    std::shared_ptr<fe::graph::Tensor_attributes>, // bias
    std::shared_ptr<fe::graph::Tensor_attributes>, // running_mean (optional)
    std::shared_ptr<fe::graph::Tensor_attributes>, // running_var (optional)
    std::shared_ptr<fe::graph::Tensor_attributes>, // Y
    std::shared_ptr<fe::graph::Tensor_attributes>, // saved_mean
    std::shared_ptr<fe::graph::Tensor_attributes>, // saved_inv_var
    std::shared_ptr<fe::graph::Tensor_attributes>, // next_running_mean (optional)
    std::shared_ptr<fe::graph::Tensor_attributes>  // next_running_var (optional)
    >;

using graph_and_tensors_backward = std::tuple<
    std::shared_ptr<fe::graph::Graph>,
    std::shared_ptr<fe::graph::Tensor_attributes>, // X
    std::shared_ptr<fe::graph::Tensor_attributes>, // DY
    std::shared_ptr<fe::graph::Tensor_attributes>, // scale
    std::shared_ptr<fe::graph::Tensor_attributes>, // mean
    std::shared_ptr<fe::graph::Tensor_attributes>, // inv_variance
    std::shared_ptr<fe::graph::Tensor_attributes>, // DX
    std::shared_ptr<fe::graph::Tensor_attributes>, // dscale
    std::shared_ptr<fe::graph::Tensor_attributes>  // dbias
    >;

struct BatchNormParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  int64_t N;  // batch size
  int64_t C;  // channels
  int64_t H;  // height
  int64_t W;  // width
  int64_t D;  // depth (for 3D)
  bool training;
  bool channels_last;
  at::MemoryFormat memory_format;
};

void setBatchNormParams(
    BatchNormParams& params,
    const Tensor& X,
    bool training,
    at::MemoryFormat memory_format) {
  std::memset(&params, 0, sizeof(params));
  params.device_id = at::cuda::current_device();
  params.dataType = get_fe_dtype(X);
  params.N = X.size(0);
  params.C = X.size(1);
  if (X.dim() >= 3) params.H = X.size(2);
  if (X.dim() >= 4) params.W = X.size(3);
  if (X.dim() >= 5) params.D = X.size(4);
  params.training = training;
  params.memory_format = memory_format;
  params.channels_last = (memory_format == at::MemoryFormat::ChannelsLast ||
                         memory_format == at::MemoryFormat::ChannelsLast3d);
}

struct BatchNormCacheKeyWrapper : ParamsWrapper<BatchNormParams> {
  BatchNormCacheKeyWrapper(
      const Tensor& X,
      bool training,
      at::MemoryFormat memory_format) {
    setBatchNormParams(this->pod, X, training, memory_format);
  }
};

template <typename T, typename KeyType>
struct BatchNormGraphCache {
  std::unordered_map<KeyType, T, ParamsWrapperHash<KeyType>> engine_cache;

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

thread_local BatchNormGraphCache<
    graph_and_tensors_forward,
    BatchNormCacheKeyWrapper>
    batchnorm_forward_graph_cache;

thread_local BatchNormGraphCache<
    graph_and_tensors_backward,
    BatchNormCacheKeyWrapper>
    batchnorm_backward_graph_cache;

void raw_cudnn_batchnorm_forward_out(
    const Tensor& X,
    const Tensor& weight,
    const Tensor& bias,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double epsilon,
    Tensor* Y,
    Tensor* save_mean,
    Tensor* save_inv_var,
    Tensor* running_mean_new,
    Tensor* running_var_new,
    Tensor* /* reserve space - not used in frontend API */,
    at::MemoryFormat memory_format) {
  
  auto key = BatchNormCacheKeyWrapper(X, training, memory_format);
  auto graph_and_tensors_forward_ptr = batchnorm_forward_graph_cache.find(key);
  auto batchnorm_graph = std::make_shared<fe::graph::Graph>();
  graph_and_tensors_forward graph_and_tensors_forward_values;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack;

  if (graph_and_tensors_forward_ptr) {
    auto [graph, X_fe, scale_fe, bias_fe, running_mean_fe, running_var_fe, 
          Y_fe, saved_mean_fe, saved_inv_var_fe, next_running_mean_fe, next_running_var_fe] =
        *graph_and_tensors_forward_ptr;
    
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {scale_fe, weight.data_ptr()},
            {bias_fe, bias.data_ptr()},
            {Y_fe, Y->data_ptr()}};
    
    if (training) {
      variant_pack_[saved_mean_fe] = save_mean->data_ptr();
      variant_pack_[saved_inv_var_fe] = save_inv_var->data_ptr();
      if (running_mean_opt.has_value() && running_var_opt.has_value()) {
        variant_pack_[running_mean_fe] = running_mean_opt.value().data_ptr();
        variant_pack_[running_var_fe] = running_var_opt.value().data_ptr();
        variant_pack_[next_running_mean_fe] = running_mean_new->data_ptr();
        variant_pack_[next_running_var_fe] = running_var_new->data_ptr();
      }
    } else {
      // Inference mode
      if (running_mean_opt.has_value() && running_var_opt.has_value()) {
        variant_pack_[running_mean_fe] = running_mean_opt.value().data_ptr();
        variant_pack_[running_var_fe] = running_var_opt.value().data_ptr();
      }
    }
    
    variant_pack = std::move(variant_pack_);
    batchnorm_graph = std::move(graph);
  } else {
    batchnorm_graph->set_io_data_type(get_fe_dtype(X))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Convert tensors to 4D format as required by cuDNN
    std::vector<int64_t> x_dim(X.sizes().begin(), X.sizes().end());
    std::vector<int64_t> x_stride(X.strides().begin(), X.strides().end());
    
    // Pad dimensions to 4D if necessary
    while (x_dim.size() < 4) {
      x_dim.push_back(1);
      x_stride.push_back(1);
    }

    auto X_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("X")
            .set_dim(x_dim)
            .set_stride(x_stride));

    auto scale_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("scale")
            .set_dim({1, X.size(1), 1, 1})
            .set_stride({X.size(1), 1, X.size(1), X.size(1)})
            .set_data_type(get_fe_dtype(weight)));

    auto bias_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("bias")
            .set_dim({1, X.size(1), 1, 1})
            .set_stride({X.size(1), 1, X.size(1), X.size(1)})
            .set_data_type(get_fe_dtype(bias)));

    auto epsilon_fe = batchnorm_graph->tensor(static_cast<float>(epsilon));
    auto momentum_fe = batchnorm_graph->tensor(static_cast<float>(momentum));

    std::shared_ptr<fe::graph::Tensor_attributes> Y_fe, saved_mean_fe, saved_inv_var_fe,
        next_running_mean_fe, next_running_var_fe, running_mean_fe, running_var_fe;

    if (training) {
      auto batchnorm_options = fe::graph::Batchnorm_attributes()
          .set_epsilon(epsilon_fe);

      if (running_mean_opt.has_value() && running_var_opt.has_value()) {
        running_mean_fe = batchnorm_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("running_mean")
                .set_dim({1, X.size(1), 1, 1})
                .set_stride({X.size(1), 1, X.size(1), X.size(1)})
                .set_data_type(get_fe_dtype(running_mean_opt.value())));

        running_var_fe = batchnorm_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("running_var")
                .set_dim({1, X.size(1), 1, 1})
                .set_stride({X.size(1), 1, X.size(1), X.size(1)})
                .set_data_type(get_fe_dtype(running_var_opt.value())));

        batchnorm_options.set_previous_running_stats(running_mean_fe, running_var_fe, momentum_fe);
      }

      auto [Y_temp, mean_temp, inv_var_temp, next_mean_temp, next_var_temp] =
          batchnorm_graph->batchnorm(X_fe, scale_fe, bias_fe, batchnorm_options);

      Y_fe = Y_temp;
      saved_mean_fe = mean_temp;
      saved_inv_var_fe = inv_var_temp;
      next_running_mean_fe = next_mean_temp;
      next_running_var_fe = next_var_temp;

      Y_fe->set_output(true);
      saved_mean_fe->set_output(true).set_data_type(get_fe_dtype(*save_mean));
      saved_inv_var_fe->set_output(true).set_data_type(get_fe_dtype(*save_inv_var));

      if (running_mean_opt.has_value() && running_var_opt.has_value()) {
        next_running_mean_fe->set_output(true).set_data_type(get_fe_dtype(*running_mean_new));
        next_running_var_fe->set_output(true).set_data_type(get_fe_dtype(*running_var_new));
      }
    } else {
      // Inference mode
      TORCH_CHECK(running_mean_opt.has_value() && running_var_opt.has_value(),
                  "running_mean and running_var must be provided in inference mode");

      running_mean_fe = batchnorm_graph->tensor(
          fe::graph::Tensor_attributes()
              .set_name("running_mean")
              .set_dim({1, X.size(1), 1, 1})
              .set_stride({X.size(1), 1, X.size(1), X.size(1)})
              .set_data_type(get_fe_dtype(running_mean_opt.value())));

      running_var_fe = batchnorm_graph->tensor(
          fe::graph::Tensor_attributes()
              .set_name("running_var")
              .set_dim({1, X.size(1), 1, 1})
              .set_stride({X.size(1), 1, X.size(1), X.size(1)})
              .set_data_type(get_fe_dtype(running_var_opt.value())));

      auto batchnorm_inference_options = fe::graph::Batchnorm_inference_attributes()
          .set_epsilon(epsilon_fe);

      Y_fe = batchnorm_graph->batchnorm_inference(
          X_fe, running_mean_fe, running_var_fe, scale_fe, bias_fe, batchnorm_inference_options);
      Y_fe->set_output(true);
    }

    cudnnHandle_t handle = getCudnnHandle();
    TORCH_INTERNAL_ASSERT(batchnorm_graph->validate().is_good());
    TORCH_INTERNAL_ASSERT(
        batchnorm_graph->build_operation_graph(handle).is_good());
    TORCH_INTERNAL_ASSERT(batchnorm_graph
                              ->create_execution_plans(
                                  {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
                              .is_good());
    TORCH_INTERNAL_ASSERT(
        batchnorm_graph->check_support(handle).is_good(),
        batchnorm_graph->check_support(handle).get_message());
    TORCH_INTERNAL_ASSERT(batchnorm_graph->build_plans(handle).is_good());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {scale_fe, weight.data_ptr()},
            {bias_fe, bias.data_ptr()},
            {Y_fe, Y->data_ptr()}};

    if (training) {
      variant_pack_[saved_mean_fe] = save_mean->data_ptr();
      variant_pack_[saved_inv_var_fe] = save_inv_var->data_ptr();
      if (running_mean_opt.has_value() && running_var_opt.has_value()) {
        variant_pack_[running_mean_fe] = running_mean_opt.value().data_ptr();
        variant_pack_[running_var_fe] = running_var_opt.value().data_ptr();
        variant_pack_[next_running_mean_fe] = running_mean_new->data_ptr();
        variant_pack_[next_running_var_fe] = running_var_new->data_ptr();
      }
    } else {
      // Inference mode
      if (running_mean_opt.has_value() && running_var_opt.has_value()) {
        variant_pack_[running_mean_fe] = running_mean_opt.value().data_ptr();
        variant_pack_[running_var_fe] = running_var_opt.value().data_ptr();
      }
    }

    variant_pack = std::move(variant_pack_);
    auto result = std::make_tuple(
        batchnorm_graph,
        X_fe,
        scale_fe,
        bias_fe,
        running_mean_fe,
        running_var_fe,
        Y_fe,
        saved_mean_fe,
        saved_inv_var_fe,
        next_running_mean_fe,
        next_running_var_fe);
    batchnorm_forward_graph_cache.update(key, std::move(result));
  }

  cudnnHandle_t handle = getCudnnHandle();
  size_t workspace_size = batchnorm_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_INTERNAL_ASSERT(!workspace_size || workspace_ptr);
  TORCH_INTERNAL_ASSERT(
      batchnorm_graph->execute(handle, variant_pack, workspace_ptr.get())
          .is_good());
}

void raw_cudnn_batchnorm_backward_out(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& weight,
    const Tensor& saved_mean,
    const Tensor& saved_inv_var,
    bool training,
    double epsilon,
    Tensor* dX,
    Tensor* dweight,
    Tensor* dbias,
    at::MemoryFormat memory_format) {
  
  auto key = BatchNormCacheKeyWrapper(X, training, memory_format);
  auto graph_and_tensors_backward_ptr = batchnorm_backward_graph_cache.find(key);
  auto batchnorm_graph = std::make_shared<fe::graph::Graph>();
  graph_and_tensors_backward graph_and_tensors_backward_values;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack;

  if (graph_and_tensors_backward_ptr) {
    auto [graph, X_fe, DY_fe, scale_fe, mean_fe, inv_variance_fe, DX_fe, dscale_fe, dbias_fe] =
        *graph_and_tensors_backward_ptr;
    
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {DY_fe, dY.data_ptr()},
            {scale_fe, weight.data_ptr()},
            {mean_fe, saved_mean.data_ptr()},
            {inv_variance_fe, saved_inv_var.data_ptr()},
            {DX_fe, dX->data_ptr()},
            {dscale_fe, dweight->data_ptr()},
            {dbias_fe, dbias->data_ptr()}};
    
    variant_pack = std::move(variant_pack_);
    batchnorm_graph = std::move(graph);
  } else {
    batchnorm_graph->set_io_data_type(get_fe_dtype(X))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Convert tensors to 4D format as required by cuDNN
    std::vector<int64_t> x_dim(X.sizes().begin(), X.sizes().end());
    std::vector<int64_t> x_stride(X.strides().begin(), X.strides().end());
    std::vector<int64_t> dy_dim(dY.sizes().begin(), dY.sizes().end());
    std::vector<int64_t> dy_stride(dY.strides().begin(), dY.strides().end());
    
    // Pad dimensions to 4D if necessary
    while (x_dim.size() < 4) {
      x_dim.push_back(1);
      x_stride.push_back(1);
    }
    while (dy_dim.size() < 4) {
      dy_dim.push_back(1);
      dy_stride.push_back(1);
    }

    auto X_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("X")
            .set_dim(x_dim)
            .set_stride(x_stride));

    auto DY_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("DY")
            .set_dim(dy_dim)
            .set_stride(dy_stride));

    auto scale_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("scale")
            .set_dim({1, X.size(1), 1, 1})
            .set_stride({X.size(1), 1, X.size(1), X.size(1)})
            .set_data_type(get_fe_dtype(weight)));

    auto mean_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("mean")
            .set_dim({1, X.size(1), 1, 1})
            .set_stride({X.size(1), 1, X.size(1), X.size(1)})
            .set_data_type(get_fe_dtype(saved_mean)));

    auto inv_variance_fe = batchnorm_graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("inv_variance")
            .set_dim({1, X.size(1), 1, 1})
            .set_stride({X.size(1), 1, X.size(1), X.size(1)})
            .set_data_type(get_fe_dtype(saved_inv_var)));

    auto batchnorm_backward_options = fe::graph::Batchnorm_backward_attributes()
        .set_saved_mean_and_inv_variance(mean_fe, inv_variance_fe);

    auto [DX_fe, dscale_fe, dbias_fe] = batchnorm_graph->batchnorm_backward(
        DY_fe, X_fe, scale_fe, batchnorm_backward_options);

    DX_fe->set_output(true);
    dscale_fe->set_output(true).set_data_type(get_fe_dtype(*dweight));
    dbias_fe->set_output(true).set_data_type(get_fe_dtype(*dbias));

    cudnnHandle_t handle = getCudnnHandle();
    TORCH_INTERNAL_ASSERT(batchnorm_graph->validate().is_good());
    TORCH_INTERNAL_ASSERT(
        batchnorm_graph->build_operation_graph(handle).is_good());
    TORCH_INTERNAL_ASSERT(batchnorm_graph
                              ->create_execution_plans(
                                  {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})
                              .is_good());
    TORCH_INTERNAL_ASSERT(
        batchnorm_graph->check_support(handle).is_good(),
        batchnorm_graph->check_support(handle).get_message());
    TORCH_INTERNAL_ASSERT(batchnorm_graph->build_plans(handle).is_good());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack_ = {
            {X_fe, X.data_ptr()},
            {DY_fe, dY.data_ptr()},
            {scale_fe, weight.data_ptr()},
            {mean_fe, saved_mean.data_ptr()},
            {inv_variance_fe, saved_inv_var.data_ptr()},
            {DX_fe, dX->data_ptr()},
            {dscale_fe, dweight->data_ptr()},
            {dbias_fe, dbias->data_ptr()}};

    variant_pack = std::move(variant_pack_);
    auto result = std::make_tuple(
        batchnorm_graph,
        X_fe,
        DY_fe,
        scale_fe,
        mean_fe,
        inv_variance_fe,
        DX_fe,
        dscale_fe,
        dbias_fe);
    batchnorm_backward_graph_cache.update(key, std::move(result));
  }

  cudnnHandle_t handle = getCudnnHandle();
  size_t workspace_size = batchnorm_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_INTERNAL_ASSERT(!workspace_size || workspace_ptr);
  TORCH_INTERNAL_ASSERT(
      batchnorm_graph->execute(handle, variant_pack, workspace_ptr.get())
          .is_good());
}

} // namespace

size_t _get_cudnn_batch_norm_reserve_space_size(
    const Tensor& input_t,
    bool training) {
  // Frontend API doesn't use reserve space
  return 0;
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> cudnn_batch_norm_out(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t_opt,
    const std::optional<Tensor>& running_mean_t_opt,
    const std::optional<Tensor>& running_var_t_opt,
    bool training,
    double exponential_average_factor,
    double epsilon,
    Tensor& output_t,
    Tensor& save_mean,
    Tensor& save_var,
    Tensor& reserve) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned =
      at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = running_mean_t_opt.value_or(Tensor());
  const Tensor& running_var_t = running_var_t_opt.value_or(Tensor());

  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2},
      bias{bias_t, "bias", 3}, running_mean{running_mean_t, "running_mean", 4},
      running_var{running_var_t, "running_var", 5};
  CheckedFrom c = "cudnn_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));

  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  // Frontend API doesn't use reserve space
  reserve = at::empty({0}, input->options().dtype(kByte));

  Tensor running_mean_new = running_mean_t.defined() ? at::empty_like(running_mean_t) : Tensor();
  Tensor running_var_new = running_var_t.defined() ? at::empty_like(running_var_t) : Tensor();

  raw_cudnn_batchnorm_forward_out(
      *input,
      *weight,
      *bias,
      running_mean_t_opt,
      running_var_t_opt,
      training,
      exponential_average_factor,
      epsilon,
      &output_t,
      &save_mean,
      &save_var,
      &running_mean_new,
      &running_var_new,
      &reserve,
      input->suggest_memory_format());

  // Update running stats if in training mode
  if (training && running_mean_t.defined() && running_var_t.defined()) {
    running_mean_t.copy_(running_mean_new);
    running_var_t.copy_(running_var_new);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>{
      output_t, save_mean, save_var, reserve};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t_opt,
    const std::optional<Tensor>& running_mean_t_opt,
    const std::optional<Tensor>& running_var_t_opt,
    bool training,
    double exponential_average_factor,
    double epsilon) {
  auto output_t = at::empty_like(
      input_t, input_t.options(), input_t.suggest_memory_format());
  Tensor save_mean, save_var, reserve;

  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({num_features}, weight_t.options());
    save_var = at::empty({num_features}, weight_t.options());
  } else {
    // This keeps a consistent output with native_batch_norm
    save_mean = at::empty({0}, weight_t.options());
    save_var = at::empty({0}, weight_t.options());
  }

  reserve = at::empty({0}, input_t.options().dtype(kByte));

  return cudnn_batch_norm_out(
      input_t,
      weight_t,
      bias_t_opt,
      running_mean_t_opt,
      running_var_t_opt,
      training,
      exponential_average_factor,
      epsilon,
      output_t,
      save_mean,
      save_var,
      reserve);
}

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input_t,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_t_opt,
    const std::optional<Tensor>& save_var_t_opt,
    double epsilon,
    const Tensor& reserveSpace) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& save_mean_t = save_mean_t_opt.value_or(Tensor());
  const Tensor& save_var_t = save_var_t_opt.value_or(Tensor());

  auto grad_output_contig =
      grad_output_t.contiguous(input_t.suggest_memory_format());
  TensorArg input{input_t, "input", 1},
      grad_output{grad_output_contig, "grad_output", 2},
      weight{weight_t, "weight", 3}, save_mean{save_mean_t, "save_mean", 4},
      save_var{save_var_t, "save_var", 5},
      reserve{reserveSpace, "reserve_space", 6};
  CheckedFrom c = "cudnn_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {save_mean, save_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  TORCH_CHECK(grad_output->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  auto grad_input_t = at::empty(
      input->sizes(), input->options(), input->suggest_memory_format());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t = at::empty(weight->sizes(), weight->options());

  raw_cudnn_batchnorm_backward_out(
      *grad_output,
      *input,
      *weight,
      *save_mean,
      *save_var,
      true, // training mode for backward
      epsilon,
      &grad_input_t,
      &grad_weight_t,
      &grad_bias_t,
      input->suggest_memory_format());

  return std::tuple<Tensor, Tensor, Tensor>{
      grad_input_t, grad_weight_t, grad_bias_t};
}

} // namespace native
} // namespace at

#endif // AT_CUDNN_ENABLED
