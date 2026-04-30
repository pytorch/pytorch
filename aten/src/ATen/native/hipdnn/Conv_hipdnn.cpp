#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>

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

#include <ATen/hipdnn/Exceptions.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/Utils.h>
#include <hipdnn_frontend.hpp>

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
// Generic LRU cache.
// ---------------------------------------------------------------------------

// TODO: lift this to a shared utility across other graph-caching DNN backends.
template <typename KeyType, typename ValueType>
struct ParamsLRUCache {
  using KeyWrapper = ParamsWrapper<KeyType>;

  int cache_limit;
  std::list<KeyWrapper> cache_order;
  std::unordered_map<
      KeyWrapper,
      std::pair<ValueType, typename std::list<KeyWrapper>::iterator>,
      ParamsWrapperHash<KeyWrapper>>
      cache;

  explicit ParamsLRUCache(int limit) : cache_limit(limit) {}

  ValueType* find(const KeyType& key) {
    if (cache_limit < 0)
      return nullptr;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end())
      return nullptr;
    if (cache_limit) {
      cache_order.splice(cache_order.begin(), cache_order, it->second.second);
    }
    return &(it->second.first);
  }

  void update(const KeyType& key, ValueType entry) {
    if (cache_limit < 0)
      return;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) {
      if (cache_limit == 0) {
        cache.emplace(
            wrapped, std::make_pair(std::move(entry), cache_order.end()));
      } else {
        if (static_cast<long>(cache.size()) >= cache_limit) {
          auto count = cache.erase(cache_order.back());
          TORCH_INTERNAL_ASSERT(
              count == 1, "LRU cache eviction failed to erase key");
          cache_order.pop_back();
        }
        cache_order.emplace_front(wrapped);
        cache.emplace(
            wrapped, std::make_pair(std::move(entry), cache_order.begin()));
      }
    } else {
      it->second.first = std::move(entry);
      if (cache_limit) {
        cache_order.splice(cache_order.begin(), cache_order, it->second.second);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Cache details specific to Conv.
// ---------------------------------------------------------------------------

constexpr int hipdnn_max_dim = 3;

// The cache key type.
// Instances should have exactly enough information to uniquely determine a Conv
// graph.
struct HipdnnConvParams {
  c10::DeviceIndex device_id;
  hipdnn_frontend::DataType dataType;
  int input_size[2 + hipdnn_max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + hipdnn_max_dim];
  int output_size[2 + hipdnn_max_dim];
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

// Allows cache limit control via environment variable.
// Note: this could be lifted to a (cuDNN/hipDNN) shared graph-cache threshold
// controller. Special values (matching cuDNN v8):
//   0 = unlimited (no eviction)
//   negative = caching disabled
static int getHipdnnConvCacheLimit() {
  static int limit = [] {
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

// The cache value type.
struct HipdnnConvCachedGraph {
  std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
  int64_t workspace_size;
};

using HipdnnConvCache = ParamsLRUCache<HipdnnConvParams, HipdnnConvCachedGraph>;

static HipdnnConvCache* getHipdnnConvCache() {
  static thread_local HipdnnConvCache cache(getHipdnnConvCacheLimit());
  return &cache;
}

// Stable UIDs for graph tensors.
enum HipdnnConvUid : int64_t {
  UID_A = 1, // fprop: x;  dgrad: dy; wgrad: dy
  UID_B = 2, // fprop: w;  dgrad: w;  wgrad: x
  UID_OUTPUT = 3, // fprop: y;  dgrad: dx; wgrad: dw
  UID_BIAS = 4, // fprop/dgrad bias-fuse; wgrad has no bias path
};

enum class HipdnnConvOp : int { Fprop = 0, Dgrad = 1, Wgrad = 2 };

// ---------------------------------------------------------------------------
// Build a hipDNN graph for fprop / dgrad / wgrad with optional bias fuse on
// fprop and dgrad. Groups are inferred by hipDNN from tensor shapes.
// ---------------------------------------------------------------------------
static HipdnnConvCachedGraph buildConvGraph(
    hipdnnHandle_t handle,
    HipdnnConvOp op,
    const Tensor& a,
    const Tensor& b,
    const Tensor& out,
    const Tensor* bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {
  TORCH_INTERNAL_ASSERT(
      !(op == HipdnnConvOp::Wgrad && bias != nullptr),
      "hipdnn wgrad has no bias-fuse path");

  auto inputType = getHipdnnDataType(a);
  auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
  graph->set_io_data_type(inputType).set_compute_data_type(
      hipdnn_frontend::DataType::FLOAT);

  auto a_attr = createTensorAttributes(a);
  a_attr->set_uid(UID_A);
  auto b_attr = createTensorAttributes(b);
  b_attr->set_uid(UID_B);

  std::vector<int64_t> p(padding.begin(), padding.end());
  std::vector<int64_t> s(stride.begin(), stride.end());
  std::vector<int64_t> d(dilation.begin(), dilation.end());
  std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> conv_out;
  switch (op) {
    case HipdnnConvOp::Fprop: {
      hipdnn_frontend::graph::ConvFpropAttributes attrs;
      attrs.set_padding(p).set_stride(s).set_dilation(d);
      conv_out = graph->conv_fprop(a_attr, b_attr, attrs);
      break;
    }
    case HipdnnConvOp::Dgrad: {
      hipdnn_frontend::graph::ConvDgradAttributes attrs;
      attrs.set_padding(p).set_stride(s).set_dilation(d);
      conv_out = graph->conv_dgrad(a_attr, b_attr, attrs);
      conv_out->set_dim(out.sizes().vec());
      break;
    }
    case HipdnnConvOp::Wgrad: {
      hipdnn_frontend::graph::ConvWgradAttributes attrs;
      attrs.set_padding(p).set_stride(s).set_dilation(d);
      conv_out = graph->conv_wgrad(a_attr, b_attr, attrs);
      conv_out->set_dim(out.sizes().vec());
      break;
    }
  }

  if (bias) {
    graph->set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);
    conv_out->set_dim(out.sizes().vec()).set_stride(out.strides().vec());
    auto bias_reshaped = reshape_bias(a.dim(), *bias);
    auto bias_attr = createTensorAttributes(bias_reshaped);
    bias_attr->set_uid(UID_BIAS);
    hipdnn_frontend::graph::PointwiseAttributes add_attrs;
    add_attrs.set_mode(hipdnn_frontend::PointwiseMode::ADD)
        .set_compute_data_type(inputType);
    auto y = graph->pointwise(conv_out, bias_attr, add_attrs);
    y->set_output(true).set_uid(UID_OUTPUT);
  } else {
    conv_out->set_output(true).set_uid(UID_OUTPUT);
  }

  HIPDNN_FE_CHECK(graph->build(handle));
  int64_t ws = 0;
  HIPDNN_FE_CHECK(graph->get_workspace_size(ws));
  return {std::move(graph), ws};
}

// ---------------------------------------------------------------------------
// Cache-lookup-then-build-and-execute. Caller must allocate `out` with the
// destination shape. For dgrad/wgrad the shape is also baked into the cache
// key to disambiguate cases with the same (a, b) shapes but different
// produced-tensor shapes (e.g. transpose conv with output_padding).
// ---------------------------------------------------------------------------
static void runHipdnnConv(
    HipdnnConvOp op,
    const Tensor& a,
    const Tensor& b,
    const Tensor& out,
    const Tensor* bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    at::MemoryFormat memory_format,
    bool benchmark,
    bool deterministic) {
  // TODO: enable these options in the near future.
  TORCH_CHECK(
      !deterministic,
      "hipdnn_convolution does not support deterministic mode yet. "
      "hipDNN does not currently provide engine-level determinism guarantees.");
  if (benchmark) {
    TORCH_WARN_ONCE(
        "hipdnn_convolution: benchmark mode is not supported yet and will be "
        "ignored. hipDNN does not currently support algorithm search.");
  }

  auto handle = getHipdnnHandle();
  auto* cache = getHipdnnConvCache();

  HipdnnConvParams key;
  setHipdnnConvParams(
      &key,
      a,
      b,
      padding,
      stride,
      dilation,
      groups,
      bias != nullptr,
      memory_format,
      static_cast<int>(op),
      out.sizes());

  auto* cached = cache->find(key);
  if (!cached) {
    auto entry =
        buildConvGraph(handle, op, a, b, out, bias, padding, stride, dilation);
    cache->update(key, std::move(entry));
    cached = cache->find(key);
  }

  std::unordered_map<int64_t, void*> variantPack;
  variantPack[UID_A] = a.data_ptr();
  variantPack[UID_B] = b.data_ptr();
  variantPack[UID_OUTPUT] = out.data_ptr();
  if (bias) {
    variantPack[UID_BIAS] = bias->data_ptr();
  }

  auto workspace =
      at::empty({cached->workspace_size}, a.options().dtype(at::kByte));
  HIPDNN_FE_CHECK(
      cached->graph->execute(handle, variantPack, workspace.data_ptr()));
}

// Sum-reduce grad_output over batch and spatial dims to get the bias gradient.
static Tensor compute_grad_bias(const Tensor& grad_output) {
  std::vector<int64_t> reduce_dims;
  reduce_dims.push_back(0);
  for (int64_t i = 2; i < grad_output.dim(); i++) {
    reduce_dims.push_back(i);
  }
  return grad_output.sum(reduce_dims);
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
  runHipdnnConv(
      HipdnnConvOp::Fprop,
      input_c,
      weight_c,
      output,
      bias_ptr,
      padding,
      stride,
      dilation,
      groups,
      memory_format,
      benchmark,
      deterministic);

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
      input_c.sizes(),
      weight_c.sizes(),
      padding,
      output_padding,
      stride,
      dilation,
      groups);
  auto output = at::empty(trans_output_size, input_c.options(), memory_format);

  bool has_bias = bias_opt.has_value() && bias_opt->defined();
  const Tensor* bias_ptr = has_bias ? &(*bias_opt) : nullptr;
  runHipdnnConv(
      HipdnnConvOp::Dgrad,
      input_c,
      weight_c,
      output,
      bias_ptr,
      padding,
      stride,
      dilation,
      groups,
      memory_format,
      benchmark,
      deterministic);

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
    runHipdnnConv(
        HipdnnConvOp::Dgrad,
        grad_output,
        weight_c,
        grad_input,
        /*bias=*/nullptr,
        padding,
        stride,
        dilation,
        groups,
        memory_format,
        benchmark,
        deterministic);
  }

  if (output_mask[1]) {
    grad_weight =
        at::empty(weight_c.sizes(), weight_c.options(), memory_format);
    runHipdnnConv(
        HipdnnConvOp::Wgrad,
        grad_output,
        input_c,
        grad_weight,
        /*bias=*/nullptr,
        padding,
        stride,
        dilation,
        groups,
        memory_format,
        benchmark,
        deterministic);
  }

  if (output_mask[2]) {
    grad_bias = compute_grad_bias(grad_output);
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
    runHipdnnConv(
        HipdnnConvOp::Fprop,
        grad_output,
        weight_c,
        grad_input,
        /*bias=*/nullptr,
        padding,
        stride,
        dilation,
        groups,
        memory_format,
        benchmark,
        deterministic);
  }

  if (output_mask[1]) {
    // Transpose backward-weight = wgrad
    grad_weight =
        at::empty(weight_c.sizes(), weight_c.options(), memory_format);
    runHipdnnConv(
        HipdnnConvOp::Wgrad,
        input_c,
        grad_output,
        grad_weight,
        /*bias=*/nullptr,
        padding,
        stride,
        dilation,
        groups,
        memory_format,
        benchmark,
        deterministic);
  }

  if (output_mask[2]) {
    grad_bias = compute_grad_bias(grad_output);
  }

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

// ---------------------------------------------------------------------------
// Dispatch stub registration
// ---------------------------------------------------------------------------
REGISTER_CUDA_DISPATCH(hipdnn_convolution_stub, &hipdnn_convolution)
REGISTER_CUDA_DISPATCH(
    hipdnn_convolution_transpose_stub,
    &hipdnn_convolution_transpose)
REGISTER_CUDA_DISPATCH(
    hipdnn_convolution_backward_stub,
    &hipdnn_convolution_backward)
REGISTER_CUDA_DISPATCH(
    hipdnn_convolution_transpose_backward_stub,
    &hipdnn_convolution_transpose_backward)

} // namespace at::native

#endif
