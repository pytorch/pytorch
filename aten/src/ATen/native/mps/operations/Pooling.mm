//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Pool.h>
#include <ATen/native/PoolingChecks.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Pooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool3d_native.h>
#include <ATen/ops/adaptive_avg_pool2d_native.h>
#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/max_pool2d_backward_native.h>
#include <ATen/ops/max_pool2d_native.h>
#include <ATen/ops/max_pool2d_with_indices_backward_native.h>
#include <ATen/ops/max_pool2d_with_indices_native.h>
#include <ATen/ops/max_pool3d_with_indices_backward_native.h>
#include <ATen/ops/max_pool3d_with_indices_native.h>
#include <ATen/ops/max_unpool2d_native.h>
#include <ATen/ops/max_unpool3d_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Pooling_metallib.h>
#endif
using namespace std::string_view_literals;

namespace mps {

static std::vector<int32_t> copy_and_maybe_expand(IntArrayRef a, int32_t pooling_dims) {
  std::vector<int32_t> b(pooling_dims);
  for (const auto dim : c10::irange(pooling_dims)) {
    b[dim] = safe_downcast<int32_t, int64_t>(a[a.size() == 1 ? 0 : dim]);
  }
  return b;
}

using PoolSizes = std::tuple<int32_t,
                             std::vector<int64_t>,
                             std::vector<int32_t>,
                             std::vector<int32_t>,
                             std::vector<int32_t>,
                             std::optional<std::vector<int32_t>>>;

static void check_pool_dims(const Tensor& t, const std::string& op_name) {
  for (const auto size : t.sizes()) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        size <= std::numeric_limits<int32_t>::max(), op_name, ": MPS does not support tensor dims larger than INT_MAX");
  }
}

static PoolSizes process_pool_sizes(const Tensor& input,
                                    IntArrayRef kernel_size,
                                    IntArrayRef stride,
                                    IntArrayRef padding,
                                    std::optional<IntArrayRef> dilation_opt,
                                    bool ceil_mode,
                                    const int32_t pooling_dims,
                                    const std::string& op_name) {
  TORCH_INTERNAL_ASSERT(pooling_dims == 1 || pooling_dims == 2 || pooling_dims == 3);

  const int32_t dims = input.dim();

  TORCH_CHECK(dims == pooling_dims + 1 || dims == pooling_dims + 2,
              op_name,
              ": non-empty ",
              pooling_dims + 1,
              "D or ",
              pooling_dims + 2,
              "D (batch mode) tensor expected for input");

  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == pooling_dims,
              op_name,
              ": kernel_size must either be a single int, or a tuple of ",
              pooling_dims,
              " ints");

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == pooling_dims,
              op_name,
              ": stride must either be omitted, a single int, or a tuple of ",
              pooling_dims,
              " ints");

  TORCH_CHECK(padding.size() == 1 || padding.size() == pooling_dims,
              op_name,
              ": padding must either be a single int, or a tuple of ",
              pooling_dims,
              " ints");

  if (dilation_opt.has_value()) {
    auto dilation = dilation_opt.value();
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == pooling_dims,
                op_name,
                ": dilation must be either a single int, or a tuple of ",
                pooling_dims,
                " ints");
  }

  int32_t leading_dims = input.dim() - pooling_dims;

  const auto kernel_size_expanded = copy_and_maybe_expand(kernel_size, pooling_dims);
  const auto stride_expanded = copy_and_maybe_expand(stride.empty() ? kernel_size : stride, pooling_dims);
  const auto padding_expanded = copy_and_maybe_expand(padding, pooling_dims);
  const auto dilation_expanded = dilation_opt.has_value() ? copy_and_maybe_expand(dilation_opt.value(), pooling_dims)
                                                          : std::vector<int32_t>(pooling_dims, 1);

  check_pool_dims(input, op_name);
  for (const auto dim : c10::irange(pooling_dims)) {
    const auto window_end = input.size(leading_dims + dim) + padding_expanded[dim] +
        (kernel_size_expanded[dim] - 1) * int64_t(dilation_expanded[dim]);
    TORCH_CHECK_NOT_IMPLEMENTED(window_end <= std::numeric_limits<int32_t>::max(),
                                op_name,
                                ": MPS does not support pooling windows reaching past INT_MAX");
  }

  std::vector<int64_t> output_pooling_size(pooling_dims);
  for (const auto dim : c10::irange(pooling_dims)) {
    output_pooling_size[dim] = pooling_output_shape<int64_t>(input.size(leading_dims + dim),
                                                             kernel_size_expanded[dim],
                                                             padding_expanded[dim],
                                                             stride_expanded[dim],
                                                             dilation_expanded[dim],
                                                             ceil_mode);
  }

  // pool2d/pool3d_shape_check below check the non-batch dimensions themselves
  if (pooling_dims == 1) {
    check_non_empty_dims(input, /*first_dim=*/leading_dims == 2 ? 1 : 0, op_name.c_str(), "input");
  }

  std::vector<int64_t> output_size(dims);
  for (const auto dim : c10::irange(leading_dims)) {
    output_size[dim] = input.size(dim);
  }
  for (const auto dim : c10::irange(pooling_dims)) {
    output_size[leading_dims + dim] = output_pooling_size[dim];
  }

  if (pooling_dims == 2) {
    const auto memory_format = input.suggest_memory_format();
    pool2d_shape_check(input,
                       kernel_size_expanded[0],
                       kernel_size_expanded[1],
                       stride_expanded[0],
                       stride_expanded[1],
                       padding_expanded[0],
                       padding_expanded[1],
                       dilation_expanded[0],
                       dilation_expanded[1],
                       input.size(leading_dims - 1),
                       input.size(leading_dims),
                       input.size(leading_dims + 1),
                       output_pooling_size[0],
                       output_pooling_size[1],
                       memory_format);
  } else if (pooling_dims == 3) {
    pool3d_shape_check(input,
                       input.size(leading_dims - 1),
                       kernel_size_expanded[0],
                       kernel_size_expanded[1],
                       kernel_size_expanded[2],
                       stride_expanded[0],
                       stride_expanded[1],
                       stride_expanded[2],
                       padding_expanded[0],
                       padding_expanded[1],
                       padding_expanded[2],
                       dilation_expanded[0],
                       dilation_expanded[1],
                       dilation_expanded[2],
                       input.size(leading_dims),
                       input.size(leading_dims + 1),
                       input.size(leading_dims + 2),
                       output_pooling_size[0],
                       output_pooling_size[1],
                       output_pooling_size[2],
                       op_name.c_str());
  }

  return PoolSizes(dims,
                   output_size,
                   kernel_size_expanded,
                   stride_expanded,
                   padding_expanded,
                   dilation_opt.has_value() ? std::make_optional(dilation_expanded) : std::nullopt);
}

static void check_pool_dtype(const Tensor& input, const std::string& op_name) {
  const auto dtype = input.scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(dtype) && dtype != kUInt16 && dtype != kUInt32 && dtype != kUInt64,
                              "\"",
                              op_name,
                              "\" not implemented for '",
                              dtype,
                              "'");
}

static void resize_pool_output(const Tensor& t, IntArrayRef size, MemoryFormat memory_format) {
  if (t.sizes() != size) {
    t.resize_(size, memory_format);
  }
}

static void fill_pool_size_strides(PoolingParams<5>& params,
                                   const Tensor& input,
                                   const Tensor& output,
                                   const std::optional<Tensor>& indices_opt) {
  const Tensor& indices = *(at::borrow_from_optional_tensor(indices_opt));
  params.channels_last = input.suggest_memory_format() == MemoryFormat::ChannelsLast;
  for (const auto dim : c10::irange(input.dim())) {
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = input.stride(dim);
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = output.stride(dim);
    if (indices.defined()) {
      params.indices_sizes[dim] = safe_downcast<int32_t, int64_t>(indices.size(dim));
      params.indices_strides[dim] = indices.stride(dim);
    }
  }
}

static MTLSize pool_linear_grid(const Tensor& t, int64_t threads) {
  int64_t height = 1;
  for (int64_t dim = 0; threads / height > std::numeric_limits<uint32_t>::max(); dim++) {
    height *= t.size(dim);
  }
  return MTLSizeMake(threads / height, height, 1);
}

// 2D kernels run one thread per tile_rows x tile_cols tile of t as (c, w, n * H + h) for channels_last and
// (w, h, n * C + c) otherwise, so that consecutive threads follow the smallest stride; the other kernels decompose
// their thread index themselves
static MTLSize pool_grid(const Tensor& t,
                         bool is_2d,
                         bool channels_last,
                         int64_t tile_rows = 1,
                         int64_t tile_cols = 1) {
  if (!is_2d) {
    return pool_linear_grid(t, t.numel());
  }
  const auto N = t.dim() == 4 ? t.size(0) : 1;
  const auto C = t.size(-3), H = ceil_div(t.size(-2), tile_rows), W = ceil_div(t.size(-1), tile_cols);
  return channels_last ? MTLSizeMake(C, W, N * H) : MTLSizeMake(W, H, N * C);
}

static int64_t pool2d_grid_max_batches(const Tensor& t) {
  const auto plane = t.suggest_memory_format() == MemoryFormat::ChannelsLast ? t.size(-2) : t.size(-3);
  constexpr int64_t max_axis = std::numeric_limits<uint32_t>::max();
  return t.dim() == 4 ? max_axis / std::max<int64_t>(plane, 1) : t.size(0);
}

// Adaptive pooling window bounds need (in_size + 1) * (out_size + 1) to fit in the kernel index type
static bool adaptive_pool_needs_int64(const Tensor& input, const Tensor& output, const int32_t pooling_dims) {
  for (const auto dim : c10::irange(input.dim() - pooling_dims, input.dim())) {
    if ((input.size(dim) + 1) * (output.size(dim) + 1) > std::numeric_limits<int32_t>::max()) {
      return true;
    }
  }
  return false;
}

static void launch_max_pool_kernel(const Tensor& input,
                                   const Tensor& output,
                                   const std::optional<Tensor>& indices_opt,
                                   const PoolingParams<5>& params,
                                   const std::string& op_name) {
  const auto numThreads = output.numel();
  if (numThreads == 0) {
    return;
  }

  const bool channels_last = params.channels_last;
  const auto memory_format = channels_last ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous;
  const auto dims = params.dims;
  // Dense kernels use 32-bit offsets within a plane (NCHW) or an image (NHWC)
  const auto plane_numel = (channels_last ? input.size(dims - 3) : 1) *
      std::max(input.size(dims - 2) * input.size(dims - 1), output.size(dims - 2) * output.size(dims - 1));
  const bool is_dense = !params.adaptive && params.pooling_dims == 2 && input.is_contiguous(memory_format) &&
      output.is_contiguous(memory_format) && (!indices_opt || indices_opt->is_contiguous(memory_format)) &&
      plane_numel <= std::numeric_limits<int32_t>::max() && output.size(0) <= pool2d_grid_max_batches(output);
  const auto kw = params.kernel_size[1];
  // With indices the dense kernel is already bound by the int64 writes
  const bool use_s1 = is_dense && !channels_last && !params.return_indices && params.stride[0] == 1 &&
      params.stride[1] == 1 && params.dilation[1] == 1;
  std::string kernel_name;
  if (use_s1) {
    kernel_name = fmt::format(
        "max_pool2d_s1_nchw_k{}_{}", kw == 2 || kw == 3 || kw == 5 ? kw : 0, scalarToMetalTypeString(input));
  } else if (is_dense) {
    kernel_name = fmt::format("max_pool2d_dense_{}{}_{}",
                              channels_last ? "nhwc"sv : "nchw"sv,
                              params.return_indices ? "_idx"sv : ""sv,
                              scalarToMetalTypeString(input));
  } else {
    const bool use_32bit = canUse32BitIndexMath(input) && canUse32BitIndexMath(output) &&
        (!indices_opt || canUse32BitIndexMath(*indices_opt)) &&
        !(params.adaptive && adaptive_pool_needs_int64(input, output, params.pooling_dims));
    kernel_name = fmt::format("max_pool{}_{}", use_32bit ? "" : "_i64", scalarToMetalTypeString(input));
  }

  const auto grid = use_s1 ? pool_grid(output, true, false, pool2d_s1_tile_rows, pool2d_s1_tile_cols)
                           : pool_grid(output, is_dense, channels_last);
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto maxPoolPSO = lib.getPipelineStateForFunc(kernel_name);

      getMPSProfiler().beginProfileKernel(maxPoolPSO, op_name, {input}, mpsStream);
      [computeEncoder setComputePipelineState:maxPoolPSO];
      mtl_setArgs(computeEncoder, input, output, indices_opt, params);

      mtl_dispatch3DJob(computeEncoder, maxPoolPSO, grid.width, grid.height, grid.depth);
      getMPSProfiler().endProfileKernel(maxPoolPSO, mpsStream);
    }
  });
}

static void max_pool_with_indices_out_mps_template(const Tensor& output,
                                                   const std::optional<Tensor>& indices_opt,
                                                   const Tensor& input,
                                                   IntArrayRef _kernel_size,
                                                   IntArrayRef _stride,
                                                   IntArrayRef _padding,
                                                   IntArrayRef _dilation,
                                                   bool ceil_mode,
                                                   const int32_t pooling_dims,
                                                   const std::string& op_name) {
  check_pool_dtype(input, op_name);
  auto [dims, output_size, kernel_size, stride, padding, dilation_opt] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, _dilation, ceil_mode, pooling_dims, op_name);
  TORCH_INTERNAL_ASSERT(dilation_opt.has_value());
  auto dilation = dilation_opt.value();
  const Tensor& indices = *(at::borrow_from_optional_tensor(indices_opt));
  const bool return_indices = indices.defined();

  const auto memory_format = input.suggest_memory_format();
  resize_pool_output(output, output_size, memory_format);
  if (return_indices) {
    resize_pool_output(indices, output_size, memory_format);
  }

  const std::optional<Tensor> indices_arg = return_indices ? std::make_optional(indices) : std::nullopt;

  PoolingParams<5> params;

  params.dims = dims;
  params.pooling_dims = pooling_dims;
  params.return_indices = return_indices;
  params.adaptive = false;
  fill_pool_size_strides(params, input, output, indices_arg);

  memcpy(params.kernel_size.data(), kernel_size.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.stride.data(), stride.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.padding.data(), padding.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.dilation.data(), dilation.data(), pooling_dims * sizeof(int32_t));

  launch_max_pool_kernel(input, output, indices_arg, params, op_name);
}

static void adaptive_max_pool_out_mps_template(const Tensor& output,
                                               const Tensor& indices,
                                               const Tensor& input,
                                               const int32_t pooling_dims,
                                               const std::string& op_name) {
  check_pool_dtype(input, op_name);
  check_pool_dims(input, op_name);
  check_pool_dims(output, op_name);
  const auto dims = static_cast<int32_t>(input.dim());

  PoolingParams<5> params;

  params.dims = dims;
  params.pooling_dims = pooling_dims;
  params.return_indices = true;
  params.adaptive = true;
  fill_pool_size_strides(params, input, output, indices);

  for (const auto dim : c10::irange(pooling_dims)) {
    params.kernel_size[dim] = 0;
    params.stride[dim] = 0;
    params.padding[dim] = 0;
    params.dilation[dim] = 1;
  }

  launch_max_pool_kernel(input, output, indices, params, op_name);
}

static int64_t pool_backward_num_threads(const Tensor& grad_output, int32_t dims, int32_t pooling_dims, bool serial) {
  auto numThreads = grad_output.numel();
  if (numThreads > 0 && serial) {
    for (const auto dim : c10::irange(dims - pooling_dims, dims)) {
      numThreads /= grad_output.size(dim);
    }
  }
  return numThreads;
}

static void max_pool_backward_out_mps_template(Tensor& grad_input,
                                               const Tensor& indices,
                                               const Tensor& input,
                                               const Tensor& grad_output,
                                               PoolingParams<5>& params,
                                               const std::string& op_name) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      supportedFloatingType(input), "\"", op_name, "\" not implemented for '", input.scalar_type(), "'");
  TORCH_CHECK_TYPE(grad_output.scalar_type() == input.scalar_type() && indices.scalar_type() == kLong,
                   op_name,
                   ": expected grad_output of dtype ",
                   input.scalar_type(),
                   " and int64 indices, got ",
                   grad_output.scalar_type(),
                   " and ",
                   indices.scalar_type());
  resize_pool_output(grad_input, input.sizes(), input.suggest_memory_format());
  const bool is_2d = params.pooling_dims == 2 && !params.adaptive;
  const auto max_batches = pool2d_grid_max_batches(grad_input);
  if (is_2d && grad_input.size(0) > max_batches) {
    for (int64_t n = 0; n < grad_input.size(0); n += max_batches) {
      const auto batches = std::min(max_batches, grad_input.size(0) - n);
      auto grad_input_batches = grad_input.narrow(0, n, batches);
      max_pool_backward_out_mps_template(grad_input_batches,
                                         indices.narrow(0, n, batches),
                                         input.narrow(0, n, batches),
                                         grad_output.narrow(0, n, batches),
                                         params,
                                         op_name);
    }
    return;
  }
  const bool use_32bit = offsetsFitIn<int32_t>(grad_input, grad_output, indices);
  if (grad_input.numel() == 0) {
    return;
  }
  params.dims = static_cast<int32_t>(input.dim());

  // The 2D kernel gathers and is deterministic. See Note [Enabling Deterministic Operations]
  // The other kernel is normally nondeterministic because of atomic accumulation
  // across multiple threads. To make it deterministic, dispatch only one thread
  // per batch, so the accumulations are serialized.
  const bool serial = !is_2d && at::globalContext().deterministicAlgorithms();
  TORCH_CHECK_NOT_IMPLEMENTED(!serial || grad_output.numel() <= std::numeric_limits<int32_t>::max(),
                              op_name,
                              ": MPS does not support deterministic gradients of more than INT_MAX outputs");
  if (serial && input.scalar_type() != kFloat) {
    Tensor grad_input_float = at::empty({0}, grad_input.options().dtype(kFloat));
    max_pool_backward_out_mps_template(
        grad_input_float, indices, input.to(kFloat), grad_output.to(kFloat), params, op_name);
    grad_input.copy_(grad_input_float);
    return;
  }
  PoolingBackwardParams<5> backward_params{};
  if (is_2d) {
    fill_pool_size_strides(params, grad_input, grad_output, indices);
  } else {
    grad_input.fill_(0);
    backward_params.dims = params.dims;
    backward_params.pooling_dims = params.pooling_dims;
    for (const auto dim : c10::irange(params.dims)) {
      backward_params.grad_input_sizes[dim] = safe_downcast<int32_t, int64_t>(grad_input.size(dim));
      backward_params.grad_input_strides[dim] = grad_input.stride(dim);
      backward_params.grad_output_sizes[dim] = safe_downcast<int32_t, int64_t>(grad_output.size(dim));
      backward_params.grad_output_strides[dim] = grad_output.stride(dim);
      backward_params.indices_strides[dim] = indices.stride(dim);
    }
  }
  const auto grid = is_2d
      ? pool_grid(grad_input, is_2d, params.channels_last, 1, pool2d_backward_tile_cols)
      : pool_linear_grid(grad_output, pool_backward_num_threads(grad_output, params.dims, params.pooling_dims, serial));

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto maxPoolPSO = lib.getPipelineStateForFunc((is_2d ? "max_pool2d_backward_" : "max_pool_backward_") +
                                                    scalarToMetalTypeString(input) + (serial ? "_serial" : "") +
                                                    (use_32bit ? "" : "_i64"));

      getMPSProfiler().beginProfileKernel(maxPoolPSO, op_name, {input}, mpsStream);
      [computeEncoder setComputePipelineState:maxPoolPSO];
      if (is_2d) {
        mtl_setArgs(computeEncoder, grad_input, grad_output, indices, params);
      } else {
        mtl_setArgs(computeEncoder, grad_input, grad_output, indices, backward_params);
      }

      mtl_dispatch3DJob(computeEncoder, maxPoolPSO, grid.width, grid.height, grid.depth);
      getMPSProfiler().endProfileKernel(maxPoolPSO, mpsStream);
    }
  });
}

static void max_pool_with_indices_backward_out_mps_template(Tensor& grad_input,
                                                            const Tensor& indices,
                                                            const Tensor& input,
                                                            const Tensor& grad_output,
                                                            IntArrayRef _kernel_size,
                                                            IntArrayRef _stride,
                                                            IntArrayRef _padding,
                                                            IntArrayRef _dilation,
                                                            bool ceil_mode,
                                                            const int32_t pooling_dims,
                                                            const std::string& op_name) {
  auto [_, __, kernel_size, stride, padding, dilation_opt] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, _dilation, ceil_mode, pooling_dims, op_name);
  PoolingParams<5> params{};
  params.pooling_dims = pooling_dims;
  memcpy(params.kernel_size.data(), kernel_size.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.stride.data(), stride.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.padding.data(), padding.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.dilation.data(), dilation_opt->data(), pooling_dims * sizeof(int32_t));
  max_pool_backward_out_mps_template(grad_input, indices, input, grad_output, params, op_name);
}

static void max_unpool_out_mps_template(const Tensor& input,
                                        const Tensor& indices,
                                        IntArrayRef output_size_,
                                        IntArrayRef stride,
                                        IntArrayRef padding,
                                        Tensor& output,
                                        const int32_t pooling_dims,
                                        const std::string& op_name) {
  check_pool_dtype(input, op_name);
  max_unpooling_shape_check(input, indices, output_size_, pooling_dims, op_name.c_str(), stride, padding);

  auto dims = input.dim();
  auto leading_dims = dims - pooling_dims;

  const auto memory_format = input.suggest_memory_format();
  std::vector<int64_t> output_size(dims);
  for (int dim : c10::irange(leading_dims)) {
    output_size[dim] = input.sizes()[dim];
  }
  for (int dim : c10::irange(pooling_dims)) {
    output_size[leading_dims + dim] = output_size_[dim];
  }

  resize_pool_output(output, output_size, memory_format);
  if (output.numel() == 0) {
    return;
  }
  check_pool_dims(input, op_name);
  check_pool_dims(output, op_name);
  output.fill_(0);

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  if (input.numel() == 0) {
    return;
  }
  const bool use_32bit = offsetsFitIn<int32_t>(output, input, indices);
  const auto grid = pool_linear_grid(input, input.numel());
  MaxUnpoolingParams<5> params;

  params.dims = dims;
  params.pooling_dims = pooling_dims;

  for (const auto dim : c10::irange(dims)) {
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = output.stride(dim);
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = input.stride(dim);
    params.indices_strides[dim] = indices.stride(dim);
  }

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto PSO =
          lib.getPipelineStateForFunc("max_unpool_" + scalarToMetalTypeString(input) + (use_32bit ? "" : "_i64"));

      getMPSProfiler().beginProfileKernel(PSO, op_name, {input}, mpsStream);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, output, input, indices, params, mpsStream->getErrorBuffer());

      mtl_dispatch3DJob(computeEncoder, PSO, grid.width, grid.height, grid.depth);
      getMPSProfiler().endProfileKernel(PSO, mpsStream);
    }
  });
}

static bool fill_avg_pool_params(AvgPoolingParams<5>& params,
                                 const Tensor& input,
                                 const Tensor& output,
                                 const int32_t pooling_dims,
                                 const std::string& op_name) {
  check_pool_dims(input, op_name);
  check_pool_dims(output, op_name);
  params.dims = static_cast<int32_t>(input.dim());
  params.pooling_dims = pooling_dims;
  params.channels_last = input.suggest_memory_format() == MemoryFormat::ChannelsLast;
  for (const auto dim : c10::irange(input.dim())) {
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = input.stride(dim);
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = output.stride(dim);
  }
  return offsetsFitIn<int32_t>(input, output);
}

template <typename... ExtraArgs>
static void launch_avg_pool_kernel(const std::string& kernel_name,
                                   const std::string& index_suffix,
                                   const Tensor& arg0,
                                   const Tensor& arg1,
                                   MTLSize grid,
                                   const AvgPoolingParams<5>& params,
                                   const std::string& op_name,
                                   const ExtraArgs&... extra_args) {
  if (grid.width * grid.height * grid.depth == 0) {
    return;
  }
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto PSO = lib.getPipelineStateForFunc(kernel_name + scalarToMetalTypeString(arg0) + index_suffix);

      getMPSProfiler().beginProfileKernel(PSO, op_name, {arg0, arg1}, mpsStream);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, arg0, arg1, params, extra_args...);

      mtl_dispatch3DJob(computeEncoder, PSO, grid.width, grid.height, grid.depth);
      getMPSProfiler().endProfileKernel(PSO, mpsStream);
    }
  });
}

static void avg_pool_out_mps_template(const Tensor& output,
                                      const Tensor& input,
                                      IntArrayRef _kernel_size,
                                      IntArrayRef _stride,
                                      IntArrayRef _padding,
                                      bool ceil_mode,
                                      bool count_include_pad,
                                      std::optional<int64_t> divisor_override,
                                      const int32_t pooling_dims,
                                      const std::string& op_name) {
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() == kLong || supportedFloatingType(input),
                              "\"",
                              op_name,
                              "\" not implemented for '",
                              input.scalar_type(),
                              "'");
  auto [dims, output_size, kernel_size, stride, padding, _] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, std::nullopt, ceil_mode, pooling_dims, op_name);

  const auto memory_format = input.suggest_memory_format();
  resize_pool_output(output, output_size, memory_format);

  AvgPoolingParams<5> params{};
  const bool use_32bit = fill_avg_pool_params(params, input, output, pooling_dims, op_name);
  params.count_include_pad = count_include_pad;
  params.has_divisor_override = divisor_override.has_value();
  if (divisor_override.has_value()) {
    params.divisor_override = divisor_override.value();
  }

  memcpy(params.kernel_size.data(), kernel_size.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.stride.data(), stride.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.padding.data(), padding.data(), pooling_dims * sizeof(int32_t));

  const bool is_dense =
      pooling_dims == 2 && use_32bit && input.is_contiguous(memory_format) && output.is_contiguous(memory_format);
  const bool use_s1 = is_dense && !params.channels_last && stride[0] == 1 && stride[1] == 1;
  std::string kernel_name = "avg_pool_";
  auto grid = pool_grid(output, is_dense, params.channels_last, avg_pool2d_tile_rows);
  if (use_s1) {
    const auto k = kernel_size[1];
    const auto kw = k == 2 || k == 3 || k == 5 || k == 7 ? k : 0;
    const bool aligned = input.size(-1) % pool2d_s1_tile_cols == 0 && output.size(-1) % pool2d_s1_tile_cols == 0 &&
        input.storage_offset() % pool2d_s1_tile_cols == 0 && output.storage_offset() % pool2d_s1_tile_cols == 0;
    const bool vec = kw % 2 == 1 && padding[1] == kw / 2 && aligned;
    kernel_name = fmt::format("avg_pool2d_s1_nchw_k{}{}_", kw, vec ? "_vec"sv : ""sv);
    grid = pool_grid(output, true, false, pool2d_s1_tile_rows, pool2d_s1_tile_cols);
  } else if (is_dense) {
    kernel_name = fmt::format("avg_pool2d_dense_{}_", params.channels_last ? "nhwc"sv : "nchw"sv);
  }
  launch_avg_pool_kernel(kernel_name, use_32bit ? "" : "_i64", input, output, grid, params, op_name);
}

static void avg_pool_backward_out_mps_template(const Tensor& grad_input,
                                               const Tensor& input,
                                               const Tensor& grad_output,
                                               IntArrayRef _kernel_size,
                                               IntArrayRef _stride,
                                               IntArrayRef _padding,
                                               bool ceil_mode,
                                               bool count_include_pad,
                                               std::optional<int64_t> divisor_override,
                                               const int32_t pooling_dims,
                                               const std::string& op_name) {
  const bool is_2d = pooling_dims == 2;
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() == kLong || supportedFloatingType(input),
                              "\"",
                              op_name,
                              "\" not implemented for '",
                              input.scalar_type(),
                              "'");
  TORCH_CHECK_TYPE(grad_output.scalar_type() == input.scalar_type(),
                   op_name,
                   ": expected grad_output of dtype ",
                   input.scalar_type(),
                   ", got ",
                   grad_output.scalar_type());

  auto [dims, _, kernel_size, stride, padding, __] =
      process_pool_sizes(input, _kernel_size, _stride, _padding, std::nullopt, ceil_mode, pooling_dims, op_name);

  resize_pool_output(grad_input, input.sizes(), input.suggest_memory_format());

  const auto max_batches = pool2d_grid_max_batches(grad_input);
  if (is_2d && grad_input.size(0) > max_batches) {
    for (int64_t n = 0; n < grad_input.size(0); n += max_batches) {
      const auto batches = std::min(max_batches, grad_input.size(0) - n);
      avg_pool_backward_out_mps_template(grad_input.narrow(0, n, batches),
                                         input.narrow(0, n, batches),
                                         grad_output.narrow(0, n, batches),
                                         _kernel_size,
                                         _stride,
                                         _padding,
                                         ceil_mode,
                                         count_include_pad,
                                         divisor_override,
                                         pooling_dims,
                                         op_name);
    }
    return;
  }

  // The 2D kernel gathers and long additions commute, so both are deterministic.
  // See Note [Enabling Deterministic Operations]
  // The other kernel is normally nondeterministic because of atomic accumulation
  // across multiple threads. To make it deterministic, dispatch only one thread
  // per batch, so the accumulations are serialized.
  const bool serial = !is_2d && input.scalar_type() != kLong && at::globalContext().deterministicAlgorithms();
  TORCH_CHECK_NOT_IMPLEMENTED(!serial || grad_output.numel() <= std::numeric_limits<int32_t>::max(),
                              op_name,
                              ": MPS does not support deterministic gradients of more than INT_MAX outputs");
  if (serial && input.scalar_type() != kFloat) {
    Tensor grad_input_float = at::empty({0}, grad_input.options().dtype(kFloat));
    avg_pool_backward_out_mps_template(grad_input_float,
                                       input.to(kFloat),
                                       grad_output.to(kFloat),
                                       _kernel_size,
                                       _stride,
                                       _padding,
                                       ceil_mode,
                                       count_include_pad,
                                       divisor_override,
                                       pooling_dims,
                                       op_name);
    grad_input.copy_(grad_input_float);
    return;
  }
  if (!is_2d) {
    grad_input.fill_(0);
  }

  AvgPoolingParams<5> params{};
  const bool use_32bit = fill_avg_pool_params(params, grad_input, grad_output, pooling_dims, op_name);
  params.count_include_pad = count_include_pad;
  params.has_divisor_override = divisor_override.has_value();
  if (divisor_override.has_value()) {
    params.divisor_override = divisor_override.value();
  }

  memcpy(params.kernel_size.data(), kernel_size.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.stride.data(), stride.data(), pooling_dims * sizeof(int32_t));
  memcpy(params.padding.data(), padding.data(), pooling_dims * sizeof(int32_t));

  const auto numThreads = pool_backward_num_threads(grad_output, dims, pooling_dims, serial);
  launch_avg_pool_kernel(is_2d ? "avg_pool2d_backward_" : "avg_pool_backward_",
                         std::string(serial ? "_serial" : "") + (use_32bit ? "" : "_i64"),
                         grad_input,
                         grad_output,
                         is_2d ? pool_grid(grad_input, is_2d, params.channels_last, 1, pool2d_backward_tile_cols)
                               : pool_linear_grid(grad_output, numThreads),
                         params,
                         op_name);
}

// For 2 pooling dims, adaptive pooling with divisible sizes is regular pooling (or its gradient, when upsampling)
// with kernel_size == stride, whose dense 2D kernels are faster than the adaptive ones
static void adaptive_avg_pool_out_mps_template(const Tensor& output,
                                               const Tensor& input,
                                               const int32_t pooling_dims,
                                               const std::string& op_name) {
  if (pooling_dims == 2) {
    const auto isizeH = input.size(-2), isizeW = input.size(-1);
    const auto osizeH = output.size(-2), osizeW = output.size(-1);
    if (isizeH % osizeH == 0 && isizeW % osizeW == 0) {
      const std::array<int64_t, 2> kernel{isizeH / osizeH, isizeW / osizeW};
      at::avg_pool2d_out(const_cast<Tensor&>(output), input, kernel, kernel, {0, 0}, false, true, std::nullopt);
      return;
    }
    if (osizeH % isizeH == 0 && osizeW % isizeW == 0) {
      const std::array<int64_t, 2> kernel{osizeH / isizeH, osizeW / isizeW};
      at::avg_pool2d_backward_out(
          const_cast<Tensor&>(output), input, at::empty_like(output), kernel, kernel, {0, 0}, false, true, 1);
      return;
    }
  }
  AvgPoolingParams<5> params{};
  const bool use_32bit = fill_avg_pool_params(params, input, output, pooling_dims, op_name);
  params.adaptive = true;
  const auto index_suffix = !use_32bit || adaptive_pool_needs_int64(input, output, pooling_dims) ? "_i64" : "";
  // A simdgroup per output pays off for windows large enough to amortize it and wide enough to keep its lanes busy
  int64_t window_size = 1;
  for (const auto dim : c10::irange(input.dim() - pooling_dims, input.dim())) {
    window_size *= c10::metal::ceil_div(input.size(dim), output.size(dim));
  }
  constexpr int64_t simd_width = 32;
  if (input.size(-1) >= 16 * output.size(-1) && window_size >= 1024 &&
      output.numel() * simd_width <= std::numeric_limits<uint32_t>::max()) {
    launch_avg_pool_kernel("adaptive_avg_pool_simd_",
                           index_suffix,
                           input,
                           output,
                           MTLSizeMake(output.numel() * simd_width, 1, 1),
                           params,
                           op_name);
    return;
  }
  launch_avg_pool_kernel(
      "avg_pool_", index_suffix, input, output, pool_linear_grid(output, output.numel()), params, op_name);
}

static void adaptive_avg_pool_backward_out_mps_template(const Tensor& grad_input,
                                                        const Tensor& grad_output,
                                                        const Tensor& input,
                                                        const int32_t pooling_dims,
                                                        const std::string& op_name) {
  if (grad_input.numel() == 0) {
    return;
  }
  if (pooling_dims == 2) {
    const auto isizeH = input.size(-2), isizeW = input.size(-1);
    const auto osizeH = grad_output.size(-2), osizeW = grad_output.size(-1);
    if (isizeH % osizeH == 0 && isizeW % osizeW == 0) {
      const std::array<int64_t, 2> kernel{isizeH / osizeH, isizeW / osizeW};
      at::avg_pool2d_backward_out(
          const_cast<Tensor&>(grad_input), grad_output, input, kernel, kernel, {0, 0}, false, true, std::nullopt);
      return;
    }
    if (osizeH % isizeH == 0 && osizeW % isizeW == 0) {
      const std::array<int64_t, 2> kernel{osizeH / isizeH, osizeW / isizeW};
      at::avg_pool2d_out(const_cast<Tensor&>(grad_input), grad_output, kernel, kernel, {0, 0}, false, true, 1);
      return;
    }
  }
  AvgPoolingParams<5> params{};
  const bool use_32bit = fill_avg_pool_params(params, grad_input, grad_output, pooling_dims, op_name);
  params.adaptive = true;
  const auto index_suffix =
      !use_32bit || adaptive_pool_needs_int64(grad_input, grad_output, pooling_dims) ? "_i64" : "";
  auto scaled_grad = at::empty(grad_output.sizes(), grad_output.options().dtype(kFloat));
  int64_t num_ranges = 0;
  for (const auto dim : c10::irange(grad_input.dim() - pooling_dims, grad_input.dim())) {
    num_ranges += grad_input.size(dim);
  }
  auto out_ranges = at::empty({num_ranges, 2}, grad_output.options().dtype(kInt));
  launch_avg_pool_kernel("adaptive_avg_pool_backward_scale_",
                         index_suffix,
                         grad_output,
                         scaled_grad,
                         pool_linear_grid(grad_output, std::max(grad_output.numel(), num_ranges)),
                         params,
                         op_name,
                         out_ranges);
  for (const auto dim : c10::irange(grad_output.dim())) {
    params.output_strides[dim] = scaled_grad.stride(dim);
  }
  launch_avg_pool_kernel("adaptive_avg_pool_backward_",
                         index_suffix,
                         grad_input,
                         scaled_grad,
                         pool_linear_grid(grad_input, grad_input.numel()),
                         params,
                         op_name,
                         out_ranges);
}

static Tensor& adaptive_avg_pool_out_mps_impl(const Tensor& input,
                                              IntArrayRef output_size,
                                              Tensor& output,
                                              const int32_t pooling_dims,
                                              const std::string& op_name) {
  adaptive_avg_pool_check(input, output, output_size, pooling_dims, (op_name + "()").c_str());
  auto output_shape = input.sizes().vec();
  std::copy(output_size.begin(), output_size.end(), output_shape.end() - pooling_dims);
  // Match CPU/CUDA and meta kernels: only 2D preserves channels last
  resize_pool_output(
      output, output_shape, pooling_dims == 2 ? input.suggest_memory_format() : MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  TORCH_CHECK_NOT_IMPLEMENTED(
      isFloatingType(input.scalar_type()), "\"", op_name, "\" not implemented for '", input.scalar_type(), "'");
  adaptive_avg_pool_out_mps_template(output, input, pooling_dims, op_name);
  return output;
}

static Tensor& adaptive_avg_pool_backward_out_mps_impl(const Tensor& grad_output,
                                                       const Tensor& input,
                                                       Tensor& grad_input,
                                                       const int32_t pooling_dims,
                                                       const std::string& op_name) {
  adaptive_avg_pool_backward_check(grad_input, grad_output, input, pooling_dims, (op_name + "()").c_str());
  TORCH_CHECK_NOT_IMPLEMENTED(
      isFloatingType(input.scalar_type()), "\"", op_name, "\" not implemented for '", input.scalar_type(), "'");
  resize_pool_output(
      grad_input, input.sizes(), pooling_dims == 2 ? input.suggest_memory_format() : MemoryFormat::Contiguous);
  adaptive_avg_pool_backward_out_mps_template(grad_input, grad_output, input, pooling_dims, op_name);
  return grad_input;
}

} // namespace mps

Tensor mps_max_pool2d(const Tensor& input,
                      IntArrayRef kernel_size,
                      IntArrayRef stride,
                      IntArrayRef padding,
                      IntArrayRef dilation,
                      bool ceil_mode) {
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "Max pooling for complex is not supported for MPS");
  Tensor output = at::empty({0}, input.options(), MemoryFormat::Contiguous);
  mps::max_pool_with_indices_out_mps_template(output,
                                              std::nullopt,
                                              input,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              ceil_mode,
                                              /*pooling_dims=*/2,
                                              "max_pool2d");
  return output;
}

Tensor mps_max_pool2d_backward(const Tensor& grad_output,
                               const Tensor& input,
                               IntArrayRef kernel_size,
                               IntArrayRef stride,
                               IntArrayRef padding,
                               IntArrayRef dilation,
                               bool ceil_mode) {
  using namespace mps;
  TORCH_CHECK_NOT_IMPLEMENTED(
      supportedFloatingType(input), "\"max_pool2d_backward\" not implemented for '", input.scalar_type(), "'");
  auto output = at::empty({0}, input.options());
  auto indices = at::empty({0}, input.options().dtype(kLong));
  max_pool_with_indices_out_mps_template(output,
                                         indices,
                                         input,
                                         kernel_size,
                                         stride,
                                         padding,
                                         dilation,
                                         ceil_mode,
                                         /*pooling_dims=*/2,
                                         "max_pool2d_backward");
  auto grad_input = at::empty({0}, input.options());
  max_pool_with_indices_backward_out_mps_template(grad_input,
                                                  indices,
                                                  input,
                                                  grad_output,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  ceil_mode,
                                                  /*pooling_dims=*/2,
                                                  "max_pool2d_backward");
  return grad_input;
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_mps)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& output,
 const Tensor& indices) {
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "Max pooling for complex is not supported for MPS");
  mps::max_pool_with_indices_out_mps_template(output,
                                              indices,
                                              input,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              ceil_mode,
                                              /*pooling_dims=*/2,
                                              "max_pool2d");
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices,
 const Tensor& grad_input) {
  mps::max_pool_with_indices_backward_out_mps_template(const_cast<Tensor&>(grad_input),
                                                       indices,
                                                       input,
                                                       grad_output,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       dilation,
                                                       ceil_mode,
                                                       /*pooling_dims=*/2,
                                                       "max_pool2d_backward");
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_mps)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "Adaptive max pooling for complex is not supported for MPS");
  mps::adaptive_max_pool_out_mps_template(output, indices, input, /*pooling_dims=*/2, "adaptive_max_pool2d");
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, const Tensor& indices, const Tensor& grad_input) {
  PoolingParams<5> params{};
  params.pooling_dims = 2;
  params.adaptive = true;
  mps::max_pool_backward_out_mps_template(
      const_cast<Tensor&>(grad_input), indices, input, grad_output, params, "adaptive_max_pool2d_backward");
}

Tensor& adaptive_avg_pool2d_out_mps(const Tensor& input, IntArrayRef output_size, Tensor& output) {
  return mps::adaptive_avg_pool_out_mps_impl(input, output_size, output, /*pooling_dims=*/2, "adaptive_avg_pool2d");
}

Tensor adaptive_avg_pool2d_mps(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  return adaptive_avg_pool2d_out_mps(input, output_size, output);
}

Tensor adaptive_avg_pool2d_backward_mps(const Tensor& grad_output, const Tensor& input) {
  auto grad_input = at::empty({0}, input.options());
  return mps::adaptive_avg_pool_backward_out_mps_impl(
      grad_output, input, grad_input, /*pooling_dims=*/2, "adaptive_avg_pool2d_backward");
}

Tensor& adaptive_avg_pool3d_out_mps(const Tensor& input, IntArrayRef output_size, Tensor& output) {
  return mps::adaptive_avg_pool_out_mps_impl(input, output_size, output, /*pooling_dims=*/3, "adaptive_avg_pool3d");
}

Tensor adaptive_avg_pool3d_mps(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  return adaptive_avg_pool3d_out_mps(input, output_size, output);
}

Tensor& adaptive_avg_pool3d_backward_out_mps(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
  return mps::adaptive_avg_pool_backward_out_mps_impl(
      grad_output, input, grad_input, /*pooling_dims=*/3, "adaptive_avg_pool3d_backward");
}

Tensor adaptive_avg_pool3d_backward_mps(const Tensor& grad_output, const Tensor& input) {
  auto grad_input = at::empty({0}, input.options());
  return adaptive_avg_pool3d_backward_out_mps(grad_output, input, grad_input);
}

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_mps(const Tensor& input,
                                                             IntArrayRef kernel_size,
                                                             IntArrayRef stride,
                                                             IntArrayRef padding,
                                                             IntArrayRef dilation,
                                                             bool ceil_mode,
                                                             Tensor& output,
                                                             Tensor& indices) {
  mps::max_pool_with_indices_out_mps_template(output,
                                              indices,
                                              input,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              ceil_mode,
                                              /*pooling_dims=*/3,
                                              "max_pool3d");
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_mps(const Tensor& input,
                                                       IntArrayRef kernel_size,
                                                       IntArrayRef stride,
                                                       IntArrayRef padding,
                                                       IntArrayRef dilation,
                                                       bool ceil_mode) {
  Tensor output = at::empty({0}, input.options(), MemoryFormat::Contiguous);
  Tensor indices = at::empty({0}, input.options().dtype(kLong), MemoryFormat::Contiguous);
  mps::max_pool_with_indices_out_mps_template(output,
                                              indices,
                                              input,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              ceil_mode,
                                              /*pooling_dims=*/3,
                                              "max_pool3d");

  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& max_pool3d_with_indices_backward_out_mps(const Tensor& grad_output,
                                                 const Tensor& input,
                                                 IntArrayRef kernel_size,
                                                 IntArrayRef stride,
                                                 IntArrayRef padding,
                                                 IntArrayRef dilation,
                                                 bool ceil_mode,
                                                 const Tensor& indices,
                                                 Tensor& grad_input) {
  mps::max_pool_with_indices_backward_out_mps_template(grad_input,
                                                       indices,
                                                       input,
                                                       grad_output,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       dilation,
                                                       ceil_mode,
                                                       /*pooling_dims=*/3,
                                                       "max_pool3d_backward");
  return grad_input;
}

Tensor max_pool3d_with_indices_backward_mps(const Tensor& grad_output,
                                            const Tensor& input,
                                            IntArrayRef kernel_size,
                                            IntArrayRef stride,
                                            IntArrayRef padding,
                                            IntArrayRef dilation,
                                            bool ceil_mode,
                                            const Tensor& indices) {
  auto grad_input = at::empty({0}, input.options());
  mps::max_pool_with_indices_backward_out_mps_template(grad_input,
                                                       indices,
                                                       input,
                                                       grad_output,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       dilation,
                                                       ceil_mode,
                                                       /*pooling_dims=*/3,
                                                       "max_pool3d_backward");
  return grad_input;
}

Tensor& max_unpooling2d_forward_out_mps(const Tensor& self,
                                        const Tensor& indices,
                                        IntArrayRef output_size,
                                        Tensor& output) {
  mps::max_unpool_out_mps_template(self,
                                   indices,
                                   output_size,
                                   /*stride=*/{},
                                   /*padding=*/{},
                                   output,
                                   /*pooling_dims=*/2,
                                   "max_unpool2d");
  return output;
}

Tensor max_unpooling2d_forward_mps(const Tensor& self, const Tensor& indices, IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  mps::max_unpool_out_mps_template(self,
                                   indices,
                                   output_size,
                                   /*stride=*/{},
                                   /*padding=*/{},
                                   output,
                                   /*pooling_dims=*/2,
                                   "max_unpool2d");
  return output;
}

Tensor& max_unpooling3d_forward_out_mps(const Tensor& self,
                                        const Tensor& indices,
                                        IntArrayRef output_size,
                                        IntArrayRef stride,
                                        IntArrayRef padding,
                                        Tensor& output) {
  mps::max_unpool_out_mps_template(self,
                                   indices,
                                   output_size,
                                   stride,
                                   padding,
                                   output,
                                   /*pooling_dims=*/3,
                                   "max_unpool3d");
  return output;
}

Tensor max_unpooling3d_forward_mps(const Tensor& self,
                                   const Tensor& indices,
                                   IntArrayRef output_size,
                                   IntArrayRef stride,
                                   IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  mps::max_unpool_out_mps_template(self,
                                   indices,
                                   output_size,
                                   stride,
                                   padding,
                                   output,
                                   /*pooling_dims=*/3,
                                   "max_unpool3d");
  return output;
}

TORCH_IMPL_FUNC(avg_pool2d_out_mps)
(const Tensor& input,
 int64_t kH,
 int64_t kW,
 int64_t dH,
 int64_t dW,
 int64_t padH,
 int64_t padW,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  mps::avg_pool_out_mps_template(output,
                                 input,
                                 {kH, kW},
                                 {dH, dW},
                                 {padH, padW},
                                 ceil_mode,
                                 count_include_pad,
                                 divisor_override,
                                 /*pooling_dims=*/2,
                                 "avg_pool2d");
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& grad_input) {
  mps::avg_pool_backward_out_mps_template(grad_input,
                                          input,
                                          grad_output,
                                          kernel_size,
                                          stride,
                                          padding,
                                          ceil_mode,
                                          count_include_pad,
                                          divisor_override,
                                          /*pooling_dims=*/2,
                                          "avg_pool2d_backward");
}

TORCH_IMPL_FUNC(avg_pool3d_out_mps)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  mps::avg_pool_out_mps_template(output,
                                 input,
                                 kernel_size,
                                 stride,
                                 padding,
                                 ceil_mode,
                                 count_include_pad,
                                 divisor_override,
                                 /*pooling_dims=*/3,
                                 "avg_pool3d");
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_mps)(const Tensor& grad_output,
                                             const Tensor& input,
                                             IntArrayRef kernel_size,
                                             IntArrayRef stride,
                                             IntArrayRef padding,
                                             bool ceil_mode,
                                             bool count_include_pad,
                                             std::optional<int64_t> divisor_override,
                                             const Tensor& grad_input) {
  mps::avg_pool_backward_out_mps_template(grad_input,
                                          input,
                                          grad_output,
                                          kernel_size,
                                          stride,
                                          padding,
                                          ceil_mode,
                                          count_include_pad,
                                          divisor_override,
                                          /*pooling_dims=*/3,
                                          "avg_pool3d_backward");
}

} // namespace at::native
