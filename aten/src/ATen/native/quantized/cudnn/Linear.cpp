#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <c10/util/ArrayRef.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/TensorUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cudnn_frontend.h>
#include <torch/library.h>

#include <iostream>
#include <unordered_map>

int register_linear_params();

// TODO: there is a table from input dtype and weight dtype to operator dtype,
// we can derive the operator dtype based on input dtype
cudnn_frontend::MatMulDesc_v8 getLinearDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::MatMulDescBuilder()
    .setMathPrecision(dataType)
    .build();
}

// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
namespace {
// we currently set the maximum number of input dimensions to 5
// this can be increased, if necessary
constexpr uint8_t max_num_input_dim = 5;
struct LinearParams {
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int64_t input_size[max_num_input_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int64_t weight_size[2];
  bool deterministic;
  bool allow_tf32;
};
struct CacheKey {
  LinearParams params;
  uint8_t input_alignment;
  uint8_t weight_alignment;
  uint8_t output_alignment;
  // default to -1 when no bias
  int8_t bias_alignment;
  bool kReluFused;
};
void setLinearParams(
    LinearParams* params, const at::Tensor& input, const at::Tensor& weight,
    bool deterministic, bool allow_tf32) {
  // operator datatype needs to be int32 for int8 matmul, but we can
  // set the datatype for output tensor to int32 or fp32
  memset(params, 0, sizeof(LinearParams));
  params->device_id = at::cuda::current_device();
  params->dataType = CUDNN_DATA_INT32;
  params->input_dim = input.dim();
  params->memory_format = input.suggest_memory_format();
  for (int i = 0; i < params->input_dim; ++i) {
    params->input_size[i] = input.sizes()[i];
  }
  for (int i = 0; i < 2; ++i) {
    params->weight_size[i] = weight.sizes()[i];
  }
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
}
std::unordered_map<CacheKey, cudnn_frontend::ExecutionPlan, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;
}
// TODO: we can use cudnn_frontend::ExecutionPlanCache when it supports caching
// multiple operators
// reference: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/conv_sample.cpp#L293
//static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

// currently we only support int8 symmetric (zero_point = 0 for inputs and output) quantized linear op
// We implement relu(act_int8 * transpose(w_int8) + [bias_fp32/(act_scale * w_scale] ) * ( act_scale * w_scale / out_scale )
// which requires 5 cudnn ops (1 matmul, 2 multiplication, 1 add, and 1 relu ops)
// matmul op: linear_op
// Multiplication ops: rhs_mult_op, requant_op
// Addition op: add_op
// Relu op: relu_op
template <bool kReluFused>
void PackedLinearWeightCudnn::apply_impl_helper(const at::Tensor& quantized_output, const at::Tensor& input, double output_scale) {
  if (quantized_output.numel() == 0) {
    return;
  }
  auto act_scale = input.q_scale();
  auto weight_scale = orig_weight.q_scale();
  auto requantize_multiplier = act_scale * weight_scale / output_scale;
  at::Tensor requantize_multiplier_tensor = cudnn_utils::getRequantMultiplierTensor(requantize_multiplier, quantized_output.dim());
  std::optional<at::Tensor> bias_multiplier_tensor;
  std::optional<at::Tensor> broadcasted_bias;
  if (bias_.has_value()) {
    // the input bias is a 1-D tensor whose size is the same as the size of the last dimension of quantized_output
    // we need to add trailing dimensions in order to properly broadcast bias, otherwise broadcast_to will fail.
    // the number of trailing dimensions is quantized_output.dim() - 2. We also prepend a leading dimension for clarity
    std::vector<int64_t> new_size(quantized_output.dim(), 1);
    new_size.back() = bias_.value().size(0);
    broadcasted_bias = bias_.value().clone().reshape(new_size);
    broadcasted_bias.value() = broadcasted_bias.value().broadcast_to(quantized_output.sizes()).contiguous();
    bias_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat));
    auto bias_multiplier = 1.0 / (act_scale * weight_scale);
    bias_multiplier_tensor.value().fill_(bias_multiplier);
  }

  cudnnHandle_t handle = at::native::getCudnnHandle();
  CacheKey key{};
  // memset is needed here because there is implicit packing added for CacheKey, and this can result in uninitialized padded values that are
  // used for hashing (see how at::native::ParamsHash is defined). without memset, we can potentially come across a situation where two
  // CacheKey objects have the same user defined parameters, but
  // different padded values, resulting in different hash outputs.
  memset(&key, 0, sizeof(key));
  bool deterministic{true};
  bool allow_tf32{false};
  setLinearParams(&key.params, input, orig_weight, deterministic, allow_tf32);

  key.input_alignment = cudnn_utils::getAlignment(input);
  key.output_alignment = cudnn_utils::getAlignment(quantized_output);
  key.weight_alignment = cudnn_utils::getAlignment(orig_weight);
  if (bias_.has_value()) {
    key.bias_alignment = static_cast<int8_t>(cudnn_utils::getAlignment(broadcasted_bias.value()));
  } else {
    key.bias_alignment = -1;
  }
  key.kReluFused = kReluFused;
  // the matmul operation is input * transpose(weight), so we will work with the transposed weight
  auto weight_transposed = transpose(orig_weight, 0, 1);
  // cudnn expects tensors to be at least 3D. weight_transposed is currently 2D. we will create a 3D view
  // by prepending a leading dummy dimension (cudnn expects leading dimensions to be the dummy dimensions)
  std::vector<int64_t> new_sizes(3, 1);
  new_sizes.back() = weight_transposed.size(1);
  new_sizes[1] = weight_transposed.size(0);
  weight_transposed = weight_transposed.view(new_sizes);

  auto run = [&](const cudnn_frontend::ExecutionPlan& plan_desc) {
    auto workspace_size = plan_desc.getWorkspaceSize();
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    at::SmallVector<void *, 8> data_ptrs;
    at::SmallVector<int64_t, 8> uids;
    data_ptrs = {input.data_ptr<int8_t>(), weight_transposed.data_ptr<int8_t>(),
                 requantize_multiplier_tensor.data_ptr(), quantized_output.data_ptr<int8_t>()};
    uids = {'x', 'w', 's', 'r'};
    if (bias_.has_value()) {
      data_ptrs.insert(data_ptrs.end(), {broadcasted_bias.value().data_ptr(), bias_multiplier_tensor.value().data_ptr(),
                                         broadcasted_bias.value().data_ptr(), broadcasted_bias.value().data_ptr()});
      uids.insert(uids.end(), {'b', 'c', 'd', 'n'});
    }
    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace_size ? workspace_ptr.get() : nullptr)
      .setDataPointers(static_cast<int64_t>(uids.size()), data_ptrs.data())
      .setUids(static_cast<int64_t>(uids.size()), uids.data())
      .build();
    auto variant_pack_desc = variantPack.get_raw_desc();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc.get_raw_desc(), variant_pack_desc));
  };

  auto search = execution_plan_cache.find(key);
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ExecutionPlan plan_desc = search->second;
    run(plan_desc);
    return;
  }

  // linear_op computes act_int8 * tranpose(w_int8) (matrix multiplication)
  // where act_int8 and w_int8 are the input and weight variables, resp.
  // output is a fp32 tensor
  auto linear_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
      .setaMatDesc(cudnn_utils::getTensorDescriptor(input.sizes(), input.strides(), CUDNN_DATA_INT8, 'x', key.input_alignment))
      .setbMatDesc(cudnn_utils::getTensorDescriptor(weight_transposed.sizes(), weight_transposed.strides(), CUDNN_DATA_INT8, 'w', key.weight_alignment))
      // for virtual tensors, the alignment is not used, so we can just put an arbitrary value here, e.g., key.output_alignment
      .setcMatDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'y', key.output_alignment, true))
      .setmatmulDesc(getLinearDescriptor(key.params.dataType))
      .build();
  // std::cout << "operator:" << linear_op.describe() << std::endl;

  std::optional<cudnn_frontend::Operation> bias_mult_op;
  std::optional<cudnn_frontend::Operation> sum_linear_bias_op;
  if (bias_.has_value()) {
    // we can't directly assign bias_mult_op because operator= is deleted for cudnn_frontend::Operation;
    // alternatively, I think we can use std::unique_ptr and dynamically allocate these builder ops
    // but here, we chose to do it statically. std::optional<T>::emplace() enables this approach

    // bias_mult_op computes bias_fp32 / (act_scale * w_scale) or bias_fp32 * (1 / (act_scale * w_scale))
    // where bias_multiplier = (1 / (act_scale * w_scale))
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    bias_mult_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'b', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setbDesc(cudnn_utils::getTensorDescriptor(bias_multiplier_tensor.value(), 'c', cudnn_utils::getAlignment(bias_multiplier_tensor.value())))
      // TODO: I think we should be able to make this a virtual tensor, but we would need cudnn to support
      // setbdesc(ManagedOpaqueDescriptor const &raw_tensor) first
      .setyDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'd', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(bias_multiplier_tensor.value())))
      .build());

    // computes (act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)])
    // where the 1st and 2nd summands is output of linear op and broadcasted_bias, resp.
    // output is a fp32 tensor
    // we use inplace operation here where the output is assigned to the input
    sum_linear_bias_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(linear_op.getOutputTensor())
      // TODO: An additional entry for broadcasted_bias in the uid-data_ptr pairing
      // appears to be needed in the current version of cudnn (8.4.0). Without it, some
      // test cases are failing. NVIDIA is currently investigating this issue.
      // When this issue is fixed, we can change 'n' back to 'd' and remove the additional entry in uid and data_ptrs in variant pack above
      .setbDesc(cudnn_utils::getTensorDescriptor(broadcasted_bias.value(), 'n', cudnn_utils::getAlignment(broadcasted_bias.value())))
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'e', key.output_alignment, true))
      .setpwDesc(cudnn_utils::getPointWiseAddDescriptor(at::native::getCudnnDataType(broadcasted_bias.value())))
      .build());
  }

  // relu_op computes relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]
  // or relu(act_int8 * w_int8) if bias is not present.
  // output is a fp32 tensor
  std::optional<cudnn_frontend::Operation> relu_op;
  std::shared_ptr<cudnn_frontend::OpaqueBackendPointer> tensor2requant_ptr = bias_.has_value() ? sum_linear_bias_op.value().getOutputTensor() : linear_op.getOutputTensor();
  if constexpr (kReluFused) {
    // we use inplace operation here where the output is assigned to the input
    relu_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(tensor2requant_ptr)
      // for virtual tensors, the alignment is not used, so we can just put an arbitrary value here, e.g., key.output_alignment
      .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_FLOAT, 'f', key.output_alignment, true))
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(CUDNN_DATA_FLOAT))
      .build());
  }

  // requant_op computes relu(act_int8 * w_int8 + [bias_fp32/(act_scale * w_scale)]) / (out_scale / (act_scale * w_scale))
  // or relu(act_int8 * w_int8) / (out_scale / (act_scale * w_scale))) if bias is not present.
  // output is a fp32 tensor
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : tensor2requant_ptr)
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 's', cudnn_utils::getAlignment(requantize_multiplier_tensor)))
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'r', key.output_alignment))
    .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(requantize_multiplier_tensor)))
    .build();
  // // std::cout << "operator:" << requant_op.describe() << std::endl;

  std::vector<cudnn_frontend::Operation const *> ops{&linear_op};
  if (bias_.has_value()) {
    ops.emplace_back(&(bias_mult_op.value()));
    ops.emplace_back(&(sum_linear_bias_op.value()));
  }
  if constexpr (kReluFused) {
    ops.emplace_back(&(relu_op.value()));
  }
  ops.emplace_back(&requant_op);

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(static_cast<int64_t>(ops.size()), ops.data())
      .build();
  // std::cout << "opGraph: " << opGraph.describe() << std::endl;

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
      .build();
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)
                    .setOperation(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                    .build();

  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  auto& fallback_list = fallback.getFallbackList();

  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_utils::filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, at::kChar);
  cudnn_utils::filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, at::kChar);

  for (auto &cfg : engine_configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();
      run(plan);
      execution_plan_cache.emplace(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << '\n';} catch(c10::CuDNNError &e) { std::cout << "other error" << e.what() << '\n';}
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation Quantized Linear Cudnn");
}

// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
// Numerics are the same as conv (see aten/src/ATen/native/quantized/Conv.cpp):
template <bool kReluFused>
at::Tensor PackedLinearWeightCudnn::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  std::vector<int64_t> original_output_shape{act.sizes().vec()}; // 2D
  original_output_shape.back() = orig_weight.size(0); // output channels
  // cudnn expects tensors to be at least 3D. we will prepend a dummy dimension for quantized_output
  std::vector<int64_t> output_shape(3, 1);
  output_shape[1] = original_output_shape[0];
  output_shape[2] = original_output_shape[1];
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
      output_scale,
      output_zero_point);
  // cudnn expects tensors to be at least 3D. act is currently 2D. We will create a 3D view
  std::vector<int64_t> new_sizes(3, 1);
  // cudnn expects leading dimensions to be the dummy dimensions
  new_sizes.back() = act.sizes().back();
  new_sizes[1] = act.size(0);
  apply_impl_helper<kReluFused>(
      quantized_output, act.view(new_sizes), output_scale);
  return quantized_output.view(original_output_shape);
}

at::Tensor PackedLinearWeightCudnn::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightCudnn::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point);
}


namespace at::native {
namespace {

template <bool kReluFused>
class QLinearInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    // TODO: check all zero_points are zero/all tensors are symmetrically quantized
    if constexpr (kReluFused) {
      return packed_weight->apply_relu(std::move(act), output_scale, output_zero_point);
    } else {
      return packed_weight->apply(std::move(act), output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear"), QLinearInt8<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu"), QLinearInt8<true>::run);
}

} // namespace
} // namespace at::native


#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
