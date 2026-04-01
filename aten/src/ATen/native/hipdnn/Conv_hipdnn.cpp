#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED() || !defined(USE_HIPDNN)

// No forward stubs needed: dispatch stubs (REGISTER_NO_CPU_DISPATCH) handle
// the non-CUDA case, and backend selection (use_hipdnn()) prevents dispatch
// to hipDNN when not compiled with hipDNN support.

#else // AT_ROCM_ENABLED && USE_HIPDNN

#include <hipdnn_frontend.hpp>
#include <ATen/hipdnn/Types.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/Exceptions.h>
#include <ATen/hipdnn/Utils.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <list>
#include <unordered_map>
#include <utility>

namespace at::native {

// ---------------------------------------------------------------------------
// Cache key: captures everything that determines graph topology
// ---------------------------------------------------------------------------
constexpr int hipdnn_max_dim = 3;

struct HipdnnConvParams {
  c10::DeviceIndex device_id;
  hipdnn_frontend::DataType dataType;
  int input_size[2 + hipdnn_max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + hipdnn_max_dim];
  int output_size[2 + hipdnn_max_dim]; // dgrad/wgrad: disambiguates output_padding
  int padding[hipdnn_max_dim];
  int stride[hipdnn_max_dim];
  int dilation[hipdnn_max_dim];
  int64_t groups;
  bool has_bias;
  int operation; // 0=fprop, 1=dgrad, 2=wgrad
};

static void setHipdnnConvParams(
    HipdnnConvParams* params,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool has_bias,
    at::MemoryFormat memory_format,
    int operation,
    IntArrayRef output_size = {}) {
  memset(params, 0, sizeof(*params));
  params->device_id = input.device().index();
  params->dataType = getHipdnnDataType(input);
  params->input_dim = static_cast<uint8_t>(input.dim());
  params->memory_format = memory_format;
  params->groups = groups;
  params->has_bias = has_bias;
  params->operation = operation;
  for (int i = 0; i < input.dim(); i++) {
    params->input_size[i] = static_cast<int>(input.size(i));
  }
  for (int i = 0; i < weight.dim(); i++) {
    params->weight_size[i] = static_cast<int>(weight.size(i));
  }
  for (size_t i = 0; i < output_size.size(); i++) {
    params->output_size[i] = static_cast<int>(output_size[i]);
  }
  int spatial_dims = input.dim() - 2;
  for (int i = 0; i < spatial_dims; i++) {
    params->padding[i] = static_cast<int>(padding[i]);
    params->stride[i] = static_cast<int>(stride[i]);
    params->dilation[i] = static_cast<int>(dilation[i]);
  }
}

// ---------------------------------------------------------------------------
// Cached graph value
// ---------------------------------------------------------------------------
struct HipdnnConvCachedGraph {
  std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
  int64_t workspace_size;
};

// ---------------------------------------------------------------------------
// Thread-local LRU cache (same pattern as cuDNN v8 Conv_v8.cpp)
// ---------------------------------------------------------------------------
// Returns the LRU cache limit for convolution graphs.
// Special values (matching cuDNN v8):
//   0       = unlimited (no eviction)
//   negative = caching disabled
static int getHipdnnConvCacheLimit() {
  static int limit = []{
    constexpr int DEFAULT_LIMIT = 10000;
    const auto val = c10::utils::get_env("TORCH_HIPDNN_CONV_LRU_CACHE_LIMIT");
    if (!val) {
      return DEFAULT_LIMIT;
    }
    try {
      return std::stoi(val.value());
    } catch (std::invalid_argument const&) {
      TORCH_WARN(
          "invalid TORCH_HIPDNN_CONV_LRU_CACHE_LIMIT,",
          " using default LRU cache limit of ",
          DEFAULT_LIMIT,
          " entries.");
    } catch (std::out_of_range const&) {
      TORCH_WARN(
          "invalid TORCH_HIPDNN_CONV_LRU_CACHE_LIMIT,",
          " using default LRU cache limit of ",
          DEFAULT_LIMIT,
          " entries.");
    }
    return DEFAULT_LIMIT;
  }();
  return limit;
}

// LRU cache for hipDNN convolution graph lookups. Keyed by convolution
// parameters (POD struct), valued by compiled graph. When we add hipDNN
// batch-norm support, this can move to a shared header.
template <typename KeyType, typename ValueType>
struct ParamsLRUCache {
  using KeyWrapper = ParamsWrapper<KeyType>;

  int cache_limit;
  std::list<KeyWrapper> cache_order;
  std::unordered_map<
      KeyWrapper,
      std::pair<ValueType, typename std::list<KeyWrapper>::iterator>,
      ParamsWrapperHash<KeyWrapper>> cache;

  explicit ParamsLRUCache(int limit) : cache_limit(limit) {}

  ValueType* find(const KeyType& key) {
    if (cache_limit < 0) return nullptr;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) return nullptr;
    if (cache_limit) {
      cache_order.splice(cache_order.begin(), cache_order, it->second.second);
    }
    return &(it->second.first);
  }

  void update(const KeyType& key, ValueType entry) {
    if (cache_limit < 0) return;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) {
      if (cache_limit == 0) {
        cache.emplace(wrapped, std::make_pair(std::move(entry), cache_order.end()));
      } else {
        if (static_cast<long>(cache.size()) >= cache_limit) {
          auto count = cache.erase(cache_order.back());
          TORCH_INTERNAL_ASSERT(count == 1, "LRU cache eviction failed to erase key");
          cache_order.pop_back();
        }
        cache_order.emplace_front(wrapped);
        cache.emplace(wrapped, std::make_pair(std::move(entry), cache_order.begin()));
      }
    } else {
      it->second.first = std::move(entry);
      if (cache_limit) {
        cache_order.splice(cache_order.begin(), cache_order, it->second.second);
      }
    }
  }
};

using HipdnnConvCache = ParamsLRUCache<HipdnnConvParams, HipdnnConvCachedGraph>;

static HipdnnConvCache* getHipdnnConvCache() {
  static thread_local auto* cache = new HipdnnConvCache(getHipdnnConvCacheLimit());
  return cache;
}

// ---------------------------------------------------------------------------
// Deterministic UID assignment for graph tensors
// ---------------------------------------------------------------------------
enum HipdnnConvUid : int64_t {
  // Forward (fprop)
  UID_INPUT  = 1,
  UID_WEIGHT = 2,
  UID_OUTPUT = 3,
  UID_BIAS   = 4,
  // Backward data (dgrad) — aliases
  UID_DGRAD_GRAD_OUTPUT = UID_INPUT,
  UID_DGRAD_WEIGHT      = UID_WEIGHT,
  UID_DGRAD_GRAD_INPUT  = UID_OUTPUT,
  // Backward weight (wgrad) — aliases
  UID_WGRAD_GRAD_OUTPUT = UID_INPUT,
  UID_WGRAD_INPUT       = UID_WEIGHT,
  UID_WGRAD_GRAD_WEIGHT = UID_OUTPUT,
};

// ---------------------------------------------------------------------------
// Graph builders
//
// Note: groups are not explicitly passed to graph builders. HipDNN infers
// groupCount from tensor dimensions (input_channels / weight_channels_per_group).
// PyTorch provides correctly-shaped weight tensors [C_out, C_in/groups, kH, kW].
// ---------------------------------------------------------------------------
static HipdnnConvCachedGraph buildConvFpropGraph(
    hipdnnHandle_t handle,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output,
    const Tensor* bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {

  auto inputType = getHipdnnDataType(input);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType)
      .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto x_attr = createTensorAttributes(input);
  x_attr->set_uid(UID_INPUT);
  auto w_attr = createTensorAttributes(weight);
  w_attr->set_uid(UID_WEIGHT);

  hipdnn_frontend::graph::ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding(std::vector<int64_t>(padding.begin(), padding.end()));
  conv_attrs.set_stride(std::vector<int64_t>(stride.begin(), stride.end()));
  conv_attrs.set_dilation(std::vector<int64_t>(dilation.begin(), dilation.end()));

  auto conv_out = graph->conv_fprop(x_attr, w_attr, conv_attrs);

  if (bias) {
    conv_out->set_dim(output.sizes().vec());
    conv_out->set_stride(output.strides().vec());

    auto bias_reshaped = reshape_bias(input.dim(), *bias);
    auto b_attr = createTensorAttributes(bias_reshaped);
    b_attr->set_uid(UID_BIAS);

    hipdnn_frontend::graph::PointwiseAttributes add_attrs;
    add_attrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    add_attrs.set_compute_data_type(inputType);

    auto y_attr = graph->pointwise(conv_out, b_attr, add_attrs);
    y_attr->set_output(true).set_uid(UID_OUTPUT);
  } else {
    conv_out->set_output(true).set_uid(UID_OUTPUT);
  }

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));

  return {std::move(graph), ws};
}

static HipdnnConvCachedGraph buildConvDgradGraph(
    hipdnnHandle_t handle,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& output,
    const Tensor* bias,
    IntArrayRef input_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {

  auto inputType = getHipdnnDataType(grad_output);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType)
      .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto dy_attr = createTensorAttributes(grad_output);
  dy_attr->set_uid(UID_INPUT);
  auto w_attr = createTensorAttributes(weight);
  w_attr->set_uid(UID_WEIGHT);

  hipdnn_frontend::graph::ConvDgradAttributes conv_attrs;
  conv_attrs.set_padding(std::vector<int64_t>(padding.begin(), padding.end()));
  conv_attrs.set_stride(std::vector<int64_t>(stride.begin(), stride.end()));
  conv_attrs.set_dilation(std::vector<int64_t>(dilation.begin(), dilation.end()));

  auto dx_attr = graph->conv_dgrad(dy_attr, w_attr, conv_attrs);
  dx_attr->set_dim(std::vector<int64_t>(input_size.begin(), input_size.end()));

  if (bias) {
    dx_attr->set_stride(output.strides().vec());

    auto bias_reshaped = reshape_bias(grad_output.dim(), *bias);
    auto b_attr = createTensorAttributes(bias_reshaped);
    b_attr->set_uid(UID_BIAS);

    hipdnn_frontend::graph::PointwiseAttributes add_attrs;
    add_attrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    add_attrs.set_compute_data_type(inputType);

    auto y_attr = graph->pointwise(dx_attr, b_attr, add_attrs);
    y_attr->set_output(true).set_uid(UID_OUTPUT);
  } else {
    dx_attr->set_output(true).set_uid(UID_OUTPUT);
  }

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));

  return {std::move(graph), ws};
}

static HipdnnConvCachedGraph buildConvWgradGraph(
    hipdnnHandle_t handle,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {

  auto inputType = getHipdnnDataType(input);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  // No set_intermediate_data_type needed: single-op graph has no virtual tensors.
  graph->set_io_data_type(inputType)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

  auto dy_attr = createTensorAttributes(grad_output);
  dy_attr->set_uid(UID_INPUT);
  auto x_attr = createTensorAttributes(input);
  x_attr->set_uid(UID_WEIGHT);

  hipdnn_frontend::graph::ConvWgradAttributes conv_attrs;
  conv_attrs.set_padding(std::vector<int64_t>(padding.begin(), padding.end()));
  conv_attrs.set_stride(std::vector<int64_t>(stride.begin(), stride.end()));
  conv_attrs.set_dilation(std::vector<int64_t>(dilation.begin(), dilation.end()));

  auto dw_attr = graph->conv_wgrad(dy_attr, x_attr, conv_attrs);
  dw_attr->set_dim(std::vector<int64_t>(weight_size.begin(), weight_size.end()));
  dw_attr->set_output(true).set_uid(UID_OUTPUT);

  HIPDNN_FE_CHECK(graph->build(handle));

  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));

  return {std::move(graph), ws};
}

// ---------------------------------------------------------------------------
// Graph execution helpers (cache-check-then-build-and-execute)
// ---------------------------------------------------------------------------
static void runHipdnnConvFprop(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output,
    const Tensor* bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format,
    bool benchmark,
    bool deterministic) {

  TORCH_CHECK(
      !deterministic,
      "hipdnn_convolution does not support deterministic mode yet. "
      "hipDNN does not currently provide engine-level determinism guarantees.");
  if (benchmark) {
    TORCH_WARN_ONCE(
        "hipdnn_convolution: benchmark mode is not supported yet and will be ignored. "
        "hipDNN does not currently support algorithm search.");
  }

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  bool has_bias = bias != nullptr;
  HipdnnConvParams key;
  setHipdnnConvParams(&key, input, weight, padding, stride, dilation,
                      groups, has_bias, memory_format, /*operation=*/0);

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry = buildConvFpropGraph(
        handle, input, weight, output, bias, padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[UID_INPUT] = input.data_ptr();
  variantPack[UID_WEIGHT] = weight.data_ptr();
  variantPack[UID_OUTPUT] = output.data_ptr();
  if (bias) {
    variantPack[UID_BIAS] = bias->data_ptr();
  }

  // Workspace inherits device from input.options()
  auto workspace = at::empty({cached->workspace_size}, input.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

static void runHipdnnConvDgrad(
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& grad_input,
    const Tensor* bias,
    IntArrayRef input_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format,
    bool benchmark,
    bool deterministic) {

  TORCH_CHECK(
      !deterministic,
      "hipdnn_convolution does not support deterministic mode yet. "
      "hipDNN does not currently provide engine-level determinism guarantees.");
  if (benchmark) {
    TORCH_WARN_ONCE(
        "hipdnn_convolution: benchmark mode is not supported yet and will be ignored. "
        "hipDNN does not currently support algorithm search.");
  }

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  bool has_bias = bias != nullptr;
  HipdnnConvParams key;
  // For dgrad, use grad_output as the "input" for the cache key.
  // input_size disambiguates cases with different output_padding.
  setHipdnnConvParams(&key, grad_output, weight, padding, stride, dilation,
                      groups, has_bias, memory_format, /*operation=*/1,
                      input_size);

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry = buildConvDgradGraph(handle, grad_output, weight, grad_input,
                                     bias, input_size, padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[UID_DGRAD_GRAD_OUTPUT] = grad_output.data_ptr();
  variantPack[UID_DGRAD_WEIGHT] = weight.data_ptr();
  variantPack[UID_DGRAD_GRAD_INPUT] = grad_input.data_ptr();
  if (bias) {
    variantPack[UID_BIAS] = bias->data_ptr();
  }

  // Workspace inherits device from grad_output.options()
  auto workspace = at::empty({cached->workspace_size}, grad_output.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

static void runHipdnnConvWgrad(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grad_weight,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format,
    bool benchmark,
    bool deterministic) {

  TORCH_CHECK(
      !deterministic,
      "hipdnn_convolution does not support deterministic mode yet. "
      "hipDNN does not currently provide engine-level determinism guarantees.");
  if (benchmark) {
    TORCH_WARN_ONCE(
        "hipdnn_convolution: benchmark mode is not supported yet and will be ignored. "
        "hipDNN does not currently support algorithm search.");
  }

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  HipdnnConvParams key;
  setHipdnnConvParams(&key, grad_output, input, padding, stride, dilation,
                      groups, /*has_bias=*/false, memory_format, /*operation=*/2,
                      weight_size);

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry = buildConvWgradGraph(handle, grad_output, input, weight_size,
                                     padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[UID_WGRAD_GRAD_OUTPUT] = grad_output.data_ptr();
  variantPack[UID_WGRAD_INPUT] = input.data_ptr();
  variantPack[UID_WGRAD_GRAD_WEIGHT] = grad_weight.data_ptr();

  // Workspace inherits device from grad_output.options()
  auto workspace = at::empty({cached->workspace_size}, grad_output.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------
Tensor hipdnn_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  TensorArg input{input_t, "input", 1};
  TensorArg weight{weight_t, "weight", 2};
  CheckedFrom c = "hipdnn_convolution";
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = hipdnn_conv_suggest_memory_format(input_t, weight_t);
  auto input_c = input_t.contiguous(memory_format);
  auto weight_c = weight_t.contiguous(memory_format);

  auto output_size = conv_output_size(
      input_c.sizes(), weight_c.sizes(), padding, stride, dilation);
  auto output = at::empty(output_size, input_c.options(), memory_format);

  bool has_bias = bias_opt.has_value() && bias_opt->defined();
  const Tensor* bias_ptr = has_bias ? &(*bias_opt) : nullptr;
  runHipdnnConvFprop(input_c, weight_c, output, bias_ptr,
                     padding, stride, dilation, groups, memory_format,
                     benchmark, deterministic);

  return output;
}

Tensor hipdnn_convolution_transpose(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  TensorArg input{input_t, "input", 1};
  TensorArg weight{weight_t, "weight", 2};
  CheckedFrom c = "hipdnn_convolution_transpose";
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = hipdnn_conv_suggest_memory_format(input_t, weight_t);
  auto input_c = input_t.contiguous(memory_format);
  auto weight_c = weight_t.contiguous(memory_format);

  auto trans_output_size = conv_input_size(
      input_c.sizes(), weight_c.sizes(), padding, output_padding, stride, dilation, groups);
  auto output = at::empty(trans_output_size, input_c.options(), memory_format);

  bool has_bias = bias_opt.has_value() && bias_opt->defined();
  const Tensor* bias_ptr = has_bias ? &(*bias_opt) : nullptr;
  runHipdnnConvDgrad(input_c, weight_c, output, bias_ptr,
                     trans_output_size, padding, stride, dilation,
                     groups, memory_format, benchmark, deterministic);

  return output;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
std::tuple<Tensor, Tensor, Tensor> hipdnn_convolution_backward(
    const Tensor& input,
    const Tensor& grad_output_t,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {

  auto memory_format = hipdnn_conv_suggest_memory_format(input, weight);
  auto grad_output = grad_output_t.contiguous(memory_format);
  auto input_c = input.contiguous(memory_format);
  auto weight_c = weight.contiguous(memory_format);

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty(input_c.sizes(), input_c.options(), memory_format);
    runHipdnnConvDgrad(grad_output, weight_c, grad_input, /*bias=*/nullptr,
                       input_c.sizes(), padding, stride, dilation,
                       groups, memory_format, benchmark, deterministic);
  }

  if (output_mask[1]) {
    grad_weight = at::empty(weight_c.sizes(), weight_c.options(), memory_format);
    runHipdnnConvWgrad(grad_output, input_c, grad_weight, weight_c.sizes(),
                       padding, stride, dilation, groups, memory_format,
                       benchmark, deterministic);
  }

  if (output_mask[2]) {
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    for (int64_t i = 2; i < grad_output.dim(); i++) {
      reduce_dims.push_back(i);
    }
    grad_bias = grad_output.sum(reduce_dims);
  }

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

std::tuple<Tensor, Tensor, Tensor> hipdnn_convolution_transpose_backward(
    const Tensor& input,
    const Tensor& grad_output_t,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {

  auto memory_format = hipdnn_conv_suggest_memory_format(input, weight);
  auto grad_output = grad_output_t.contiguous(memory_format);
  auto input_c = input.contiguous(memory_format);
  auto weight_c = weight.contiguous(memory_format);

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    // Transpose backward-input = fprop
    grad_input = at::empty(input_c.sizes(), input_c.options(), memory_format);
    runHipdnnConvFprop(grad_output, weight_c, grad_input, /*bias=*/nullptr,
                       padding, stride, dilation, groups, memory_format,
                       benchmark, deterministic);
  }

  if (output_mask[1]) {
    // Transpose backward-weight = wgrad
    grad_weight = at::empty(weight_c.sizes(), weight_c.options(), memory_format);
    runHipdnnConvWgrad(input_c, grad_output, grad_weight, weight_c.sizes(),
                       padding, stride, dilation, groups, memory_format,
                       benchmark, deterministic);
  }

  if (output_mask[2]) {
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    for (int64_t i = 2; i < grad_output.dim(); i++) {
      reduce_dims.push_back(i);
    }
    grad_bias = grad_output.sum(reduce_dims);
  }

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

// ---------------------------------------------------------------------------
// Dispatch stub registration
// ---------------------------------------------------------------------------
REGISTER_CUDA_DISPATCH(hipdnn_convolution_stub, &hipdnn_convolution)
REGISTER_CUDA_DISPATCH(hipdnn_convolution_transpose_stub, &hipdnn_convolution_transpose)
REGISTER_CUDA_DISPATCH(hipdnn_convolution_backward_stub, &hipdnn_convolution_backward)
REGISTER_CUDA_DISPATCH(hipdnn_convolution_transpose_backward_stub, &hipdnn_convolution_transpose_backward)

} // namespace at::native

#endif
