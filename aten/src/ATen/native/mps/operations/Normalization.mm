//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/BatchNorm.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/_native_batch_norm_legit_native.h>
#include <ATen/ops/batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LayerNorm_metallib.h>
#endif

static void get_shapes(MPSShape* input_shape_readonly,
                       NSMutableArray<NSNumber*>*& input_shape,
                       NSMutableArray<NSNumber*>*& new_mean_shape,
                       NSMutableArray<NSNumber*>*& axes,
                       int num_input_dims,
                       c10::MemoryFormat memory_format,
                       bool isBackward) {
  // Modify the shape
  if (memory_format == at::MemoryFormat::Contiguous) {
    for (int i = 0; i < num_input_dims; i++)
      input_shape[i] = input_shape_readonly[i];
  } else { // ChannelsLast
    auto num_channels = input_shape_readonly[1];
    input_shape[0] = input_shape_readonly[0];
    for (int i = 1; i < num_input_dims - 1; i++)
      input_shape[i] = input_shape_readonly[i + 1];
    input_shape[num_input_dims - 1] = num_channels;
  }

  // Mean shape should remain unchanged in backward
  if (memory_format == at::MemoryFormat::Contiguous || isBackward) {
    new_mean_shape[0] = @1;
    new_mean_shape[1] = input_shape_readonly[1];
    for (int i = 2; i < num_input_dims; i++)
      new_mean_shape[i] = @1;
  } else if (memory_format == at::MemoryFormat::ChannelsLast) {
    for (int i = 0; i < num_input_dims - 1; i++)
      new_mean_shape[i] = @1;
    new_mean_shape[num_input_dims - 1] = input_shape[num_input_dims - 1];
  }

  // Set axes of reduction
  if (memory_format == at::MemoryFormat::Contiguous || isBackward) {
    axes[0] = @0;
    for (int i = 2; i < num_input_dims; i++)
      axes[i - 1] = [NSNumber numberWithInt:i];
  } else {
    for (int i = 0; i < num_input_dims - 1; i++)
      axes[i] = [NSNumber numberWithInt:i];
  }
}

static std::string get_mem_string(c10::MemoryFormat memory_format) {
  switch (memory_format) {
    case at::MemoryFormat::Contiguous:
      return "Contiguous";
    case at::MemoryFormat::ChannelsLast:
      return "ChannelsLast";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "Unexpected memory format", memory_format);
}

static std::string_view batch_norm_variant(const Tensor& x, const Tensor& out) {
  if (!canUse32BitIndexMath(x, std::numeric_limits<uint32_t>::max()) ||
      !canUse32BitIndexMath(out, std::numeric_limits<uint32_t>::max())) {
    return "strided64_";
  }
  return x.is_non_overlapping_and_dense() && out.strides() == x.strides() ? "" : "strided_";
}

static BatchNormLayout<> batch_norm_layout(const Tensor& x, const Tensor& out) {
  using dim_t = c10::metal::vec3<int64_t>;
  const auto dim = [&](int64_t d) { return dim_t{x.size(d), x.stride(d), out.stride(d)}; };
  std::vector<dim_t> sorted, dims;
  for (const auto d : c10::irange(x.dim())) {
    if (d != 1 && x.size(d) != 1) {
      sorted.push_back(dim(d));
    }
  }
  std::ranges::sort(sorted, {}, &dim_t::y);
  for (const auto& e : sorted) {
    const auto& f = dims.empty() ? e : dims.back();
    if (!dims.empty() && e.y == f.x * f.y && e.z == f.x * f.z) {
      dims.back().x *= e.x;
    } else {
      dims.push_back(e);
    }
  }
  const bool has_inner = !dims.empty() && (x.size(1) == 1 || dims[0].y <= x.stride(1));
  const auto inner = has_inner ? dims[0] : dim_t{1, 0, 0};
  BatchNormLayout<> layout{.channel = dim(1),
                           .inner = inner,
                           .outer_numel = x.numel() / (x.size(1) * inner.x),
                           .outer_ndim = static_cast<int32_t>(std::max<size_t>(dims.size() - has_inner, 1))};
  std::ranges::copy(dims | std::views::drop(has_inner), layout.outer_dims.begin());
  return layout;
}

static Tensor batch_norm_reduce(const Tensor& x) {
  constexpr int64_t min_elems_per_thread = 16, groups_per_core = 4;
  static const auto max_groups = groups_per_core * std::max(at::mps::MPSDevice::getInstance()->getCoreCount(), 1u);
  const auto layout = batch_norm_layout(x, x);
  const int64_t C = layout.channel.x, inner = layout.inner.x, outer = layout.outer_numel;
  const bool channels_last = inner == 1;
  const auto threads = channels_last
      ? BN_MAX_THREADS
      : std::clamp<int64_t>(outer * inner / min_elems_per_thread, c10::metal::simdgroup_size, BN_MAX_THREADS);
  const auto tx = channels_last ? c10::metal::simdgroup_size : std::min(inner, threads);
  const auto ty = std::clamp<int64_t>(outer, 1, threads / tx);
  const auto blocks = channels_last ? c10::metal::ceil_div(C, tx) : C;
  const auto chunks = std::max<int64_t>(1, std::min(max_groups / blocks, c10::metal::ceil_div(outer, ty)));
  const auto splits = std::max<int64_t>(1, std::min(max_groups / (blocks * chunks), c10::metal::ceil_div(inner, tx)));
  auto out = at::empty({chunks * splits, C, BN_PARTIAL_SIZE}, x.options().dtype(kFloat));
  const auto name = fmt::format("batch_norm_reduce_{}{}", batch_norm_variant(x, x), scalarToMetalTypeString(x));
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    @autoreleasepool {
      auto encoder = getCurrentMPSStream()->commandEncoder();
      [encoder setComputePipelineState:lib.getPipelineStateForFunc(name)];
      mtl_setArgs(encoder, x, out, layout);
      [encoder dispatchThreadgroups:MTLSizeMake(blocks, chunks, splits) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
    }
  });
  return out;
}

static Tensor batch_norm_stats(const Tensor& x,
                               const std::optional<Tensor>& part,
                               std::array<std::optional<Tensor>, BN_NUM_PARAMS> tensors,
                               double eps,
                               double momentum) {
  const auto C = x.size(1);
  const auto dtype = tensors[BN_SAVE_MEAN]->scalar_type();
  for (auto& t : tensors) {
    t = t.has_value() && t->defined() ? t : std::nullopt;
    TORCH_CHECK_TYPE(!t || t->scalar_type() == dtype, "weight, bias, running and saved stats must share a dtype");
  }
  BatchNormStatsParams<> params{.num_channels = C,
                                .num_partials = part ? part->size(0) : 0,
                                .eps = static_cast<float>(eps),
                                .momentum = static_cast<float>(momentum)};
  std::ranges::transform(tensors, params.strides.begin(), [](const auto& t) { return t ? t->stride(0) : 0; });
  auto stats = at::empty({C, BN_NUM_STATS}, x.options().dtype(kFloat));
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    @autoreleasepool {
      auto encoder = getCurrentMPSStream()->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("batch_norm_stats_" + scalarToMetalTypeString(dtype));
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, part, stats, params);
      std::apply([&](const auto&... t) { mtl_setArgs<3>(encoder, t...); }, tensors);
      mtl_dispatch1DJob(encoder, pso, C);
    }
  });
  return stats;
}

static void batch_norm_elemt(const Tensor& out, const Tensor& x, const Tensor& stats) {
  constexpr int64_t max_grid = std::numeric_limits<uint32_t>::max();
  auto layout = batch_norm_layout(x, out);
  // A plane wider than the grid is folded into the outer dims, which the 64-bit variant grid-strides over
  if (layout.inner.x > max_grid) {
    std::shift_right(layout.outer_dims.begin(), layout.outer_dims.end(), 1);
    layout.outer_dims[0] = std::exchange(layout.inner, {1, 0, 0});
    layout.outer_numel *= layout.outer_dims[0].x;
    layout.outer_ndim++;
  }
  const auto name = fmt::format("batch_norm_elemt_{}{}", batch_norm_variant(x, out), scalarToMetalTypeString(x));
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    @autoreleasepool {
      auto encoder = getCurrentMPSStream()->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(name);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, out, x, stats, layout);
      mtl_dispatch3DJob(encoder, pso, layout.inner.x, layout.channel.x, std::min(layout.outer_numel, max_grid));
    }
  });
}
} // namespace mps

// Inverse standard deviation now becomes variance (without epsilon)
std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_mps_out(const Tensor& self,
                                                         const std::optional<Tensor>& weight_opt,
                                                         const std::optional<Tensor>& bias_opt,
                                                         const std::optional<Tensor>& running_mean_opt,
                                                         const std::optional<Tensor>& running_var_opt,
                                                         bool train,
                                                         double momentum,
                                                         double epsilon,
                                                         Tensor& output,
                                                         Tensor& save_mean,
                                                         Tensor& save_var) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      isFloatingType(self.scalar_type()), "batch_norm is not implemented for ", self.scalar_type(), " on MPS");
  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK_VALUE(has_running_mean == has_running_var,
                    "running_mean and running_var must either both be None or neither be None");
  if (output.numel() == 0) {
    return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_var);
  }
  TORCH_CHECK_VALUE(train || has_running_mean, "running_mean must be defined in evaluation mode");
  save_mean.resize_({train ? self.size(1) : 0});
  save_var.resize_({train ? self.size(1) : 0});
  const auto part = train ? std::optional<Tensor>(mps::batch_norm_reduce(self)) : std::nullopt;
  const auto stats = mps::batch_norm_stats(
      self, part, {weight_opt, bias_opt, running_mean_opt, running_var_opt, save_mean, save_var}, epsilon, momentum);
  mps::batch_norm_elemt(output, self, stats);
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_var);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_mps(const Tensor& self,
                                                  const std::optional<Tensor>& weight_opt,
                                                  const std::optional<Tensor>& bias_opt,
                                                  const std::optional<Tensor>& running_mean_opt,
                                                  const std::optional<Tensor>& running_var_opt,
                                                  bool train,
                                                  double momentum,
                                                  double epsilon) {
  auto output = at::empty_like(self);

  int64_t n_input = self.size(1);
  // Saved stats follow CPU: float only when the parameters are float and the input is not
  const auto param = weight_opt.value_or(Tensor()).defined() ? *weight_opt : running_mean_opt.value_or(Tensor());
  const auto stats_dtype = param.defined() ? param.scalar_type() : self.scalar_type();

  auto save_mean = at::empty({n_input}, stats_dtype, std::nullopt, kMPS, std::nullopt, std::nullopt);
  auto save_var = at::empty({n_input}, stats_dtype, std::nullopt, kMPS, std::nullopt, std::nullopt);

  at::native::batch_norm_mps_out(self,
                                 weight_opt,
                                 bias_opt,
                                 running_mean_opt,
                                 running_var_opt,
                                 train,
                                 momentum,
                                 epsilon,
                                 output,
                                 save_mean,
                                 save_var);
  return std::make_tuple(output, save_mean, save_var);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_mps(const Tensor& input,
                                                                       const std::optional<Tensor>& weight_opt,
                                                                       const std::optional<Tensor>& bias_opt,
                                                                       Tensor& running_mean,
                                                                       Tensor& running_var,
                                                                       double momentum,
                                                                       double eps) {
  auto [output, save_mean, save_var] =
      batch_norm_mps(input, weight_opt, bias_opt, running_mean, running_var, /*train*/ true, momentum, eps);
  Tensor reserve = at::empty({0}, input.options().dtype(kByte));
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _batch_norm_with_update_mps_out(const Tensor& input,
                                                                               const std::optional<Tensor>& weight_opt,
                                                                               const std::optional<Tensor>& bias_opt,
                                                                               Tensor& running_mean,
                                                                               Tensor& running_var,
                                                                               double momentum,
                                                                               double eps,
                                                                               Tensor& out,
                                                                               Tensor& save_mean,
                                                                               Tensor& save_var,
                                                                               Tensor& reserve) {
  std::tie(out, save_mean, save_var) = batch_norm_mps_out(
      input, weight_opt, bias_opt, running_mean, running_var, /*update*/ true, momentum, eps, out, save_mean, save_var);
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(out, save_mean, save_var, reserve);
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_mps(const Tensor& self,
                                                         const std::optional<Tensor>& weight_opt,
                                                         const std::optional<Tensor>& bias_opt,
                                                         Tensor& running_mean,
                                                         Tensor& running_var,
                                                         bool train,
                                                         double momentum,
                                                         double epsilon) {
  return batch_norm_mps(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, epsilon);
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_mps(const Tensor& self,
                                                                  const std::optional<Tensor>& weight_opt,
                                                                  const std::optional<Tensor>& bias_opt,
                                                                  bool train,
                                                                  double momentum,
                                                                  double epsilon) {
  return batch_norm_mps(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, epsilon);
}

std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_mps_out(const Tensor& self,
                                                                const std::optional<Tensor>& weight_opt,
                                                                const std::optional<Tensor>& bias_opt,
                                                                Tensor& running_mean,
                                                                Tensor& running_var,
                                                                bool train,
                                                                double momentum,
                                                                double epsilon,
                                                                Tensor& output,
                                                                Tensor& save_mean,
                                                                Tensor& save_var) {
  return batch_norm_mps_out(
      self, weight_opt, bias_opt, running_mean, running_var, train, momentum, epsilon, output, save_mean, save_var);
}

std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_no_stats_mps_out(const Tensor& self,
                                                                         const std::optional<Tensor>& weight_opt,
                                                                         const std::optional<Tensor>& bias_opt,
                                                                         bool train,
                                                                         double momentum,
                                                                         double epsilon,
                                                                         Tensor& output,
                                                                         Tensor& save_mean,
                                                                         Tensor& save_var) {
  return batch_norm_mps_out(
      self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, epsilon, output, save_mean, save_var);
}

// Batch norm backward
std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_mps(const Tensor& grad_output,
                                                                const Tensor& input,
                                                                const Tensor& weight,
                                                                const std::optional<Tensor>& running_mean_opt,
                                                                const std::optional<Tensor>& running_var_opt,
                                                                const std::optional<Tensor>& save_mean_opt,
                                                                const std::optional<Tensor>& save_var_opt,
                                                                bool update,
                                                                double eps,
                                                                std::array<bool, 3> grad_input_mask,
                                                                const Tensor& reserve) {
  return batch_norm_backward_mps(grad_output,
                                 input,
                                 weight,
                                 running_mean_opt,
                                 running_var_opt,
                                 save_mean_opt,
                                 save_var_opt,
                                 update,
                                 eps,
                                 grad_input_mask);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_mps(const Tensor& grad_out,
                                                           const Tensor& input,
                                                           const std::optional<Tensor>& weight_opt,
                                                           const std::optional<Tensor>& running_mean_opt,
                                                           const std::optional<Tensor>& running_var_opt,
                                                           const std::optional<Tensor>& save_mean_opt,
                                                           const std::optional<Tensor>& save_var_opt,
                                                           bool train,
                                                           double epsilon,
                                                           std::array<bool, 3> grad_input_mask) {
  // Flatten 5D to 4D (see batch_norm_mps_out for rationale).
  if (input.dim() == 5) {
    auto input_4d =
        input.contiguous().reshape({input.size(0), input.size(1), input.size(2) * input.size(3), input.size(4)});
    auto grad_out_4d = grad_out.contiguous().reshape(input_4d.sizes());
    auto [gi, gw, gb] = batch_norm_backward_mps(grad_out_4d,
                                                input_4d,
                                                weight_opt,
                                                running_mean_opt,
                                                running_var_opt,
                                                save_mean_opt,
                                                save_var_opt,
                                                train,
                                                epsilon,
                                                grad_input_mask);
    if (gi.defined())
      gi = gi.reshape(input.sizes());
    return std::make_tuple(std::move(gi), std::move(gw), std::move(gb));
  }

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  // See https://github.com/pytorch/pytorch/issues/180984
  const auto memory_format = input.suggest_memory_format(/*channels_last_strides_exact_match=*/true);

  if (grad_input_mask[0]) {
    grad_input = at::empty(input.sizes(), input.scalar_type(), std::nullopt, kMPS, std::nullopt, memory_format);
  }
  // Assuming that if grad_input_mask of weight is 1, then the weight is available
  if (grad_input_mask[1]) {
    grad_weight = at::empty(weight_opt.value().sizes(),
                            weight_opt.value().scalar_type(),
                            std::nullopt,
                            kMPS,
                            std::nullopt,
                            at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::empty(weight_opt.value().sizes(),
                          weight_opt.value().scalar_type(),
                          std::nullopt,
                          kMPS,
                          std::nullopt,
                          at::MemoryFormat::Contiguous);
  }

  using namespace at::native::mps;

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* runningMeanTensor_ = nil;
    MPSGraphTensor* runningVarTensor_ = nil;
    MPSGraphTensor* saveMeanTensor_ = nil;
    MPSGraphTensor* saveVarTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
    MPSGraphTensor* gradBiasTensor_ = nil;
  };

  auto stream = at::mps::getCurrentMPSStream();

  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK_VALUE(has_running_mean == has_running_var,
                    "running_mean and running_var must either both be None or neither be None");
  const bool has_save_mean = (save_mean_opt.has_value() && save_mean_opt->defined());
  const bool has_save_var = (save_var_opt.has_value() && save_var_opt->defined());
  TORCH_CHECK_VALUE(has_save_mean == has_save_var,
                    "save_mean and save_var must either both be None or neither be None");

  const bool has_weight = (weight_opt.has_value() && weight_opt->defined());

  bool any_grad_needed = (grad_input_mask[0] && grad_input.numel() > 0) ||
      (grad_input_mask[1] && grad_weight.numel() > 0) || (grad_input_mask[2] && grad_bias.numel() > 0);

  if (!any_grad_needed) {
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  @autoreleasepool {
    MPSShape* input_shape_readonly = mps::getMPSShape(input);
    int num_input_dims = [input_shape_readonly count];
    NSMutableArray<NSNumber*>* input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    // Broadcast with input
    NSMutableArray<NSNumber*>* new_mean_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    // Reduction axes
    NSMutableArray<NSNumber*>* axes = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_input_dims - 1)];

    get_shapes(input_shape_readonly, input_shape, new_mean_shape, axes, num_input_dims, memory_format, true);

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    auto input_mps_dtype = getMPSDataType(input);
    auto weight_mps_dtype = has_weight ? getMPSDataType(weight_opt.value()) : input_mps_dtype;
    std::string key = fmt::format("batch_norm_backward_mps:{}:{}:{}:{}:{}:{}:{}:{}",
                                  get_mem_string(memory_format),
                                  epsilon,
                                  train,
                                  has_running_mean,
                                  has_weight,
                                  [ns_shape_key UTF8String],
                                  c10::Join(",", grad_input_mask),
                                  getTensorsStringKey({grad_out,
                                                       input,
                                                       weight_opt.value_or(Tensor()),
                                                       running_mean_opt.value_or(Tensor()),
                                                       running_var_opt.value_or(Tensor()),
                                                       save_mean_opt.value_or(Tensor()),
                                                       save_var_opt.value_or(Tensor())}));
    const auto compute_dtype = getMPSDataType(toAccumulateType(input.scalar_type(), kMPS));
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      // NCHW - Channels dim is 1
      int channelsDim = 1;

      auto inputTensorOriginal = mpsGraphRankedPlaceHolder(mpsGraph, input_mps_dtype, input_shape);
      // Shape is the ORIGINAL NCHW shape
      auto gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_out), input_shape_readonly);
      MPSGraphTensor* weightTensor = nil;
      MPSGraphTensor* weightTensorCasted = nil;
      if (has_weight) {
        weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_mps_dtype, new_mean_shape);
        weightTensorCasted = castMPSTensor(mpsGraph, weightTensor, compute_dtype);
      }
      MPSGraphTensor* runningMeanTensor = nil;
      MPSGraphTensor* runningVarTensor = nil;
      if (has_running_mean) {
        runningMeanTensor =
            mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(running_mean_opt.value()), new_mean_shape);
        runningVarTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(running_var_opt.value()), new_mean_shape);
      }

      // Mean and inv std tensors to be saved and returned
      MPSGraphTensor* saveMeanTensor = nil;
      MPSGraphTensor* saveVarTensor = nil;
      if (has_save_mean) {
        saveMeanTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(save_mean_opt.value()), new_mean_shape);
        saveVarTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(save_var_opt.value()), new_mean_shape);
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensorOriginal;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->runningMeanTensor_ = runningMeanTensor;
      newCachedGraph->runningVarTensor_ = runningVarTensor;
      newCachedGraph->saveMeanTensor_ = saveMeanTensor;
      newCachedGraph->saveVarTensor_ = saveVarTensor;

      gradOutputTensor = castMPSTensor(mpsGraph, gradOutputTensor, compute_dtype);
      if (has_running_mean) {
        runningMeanTensor = castMPSTensor(mpsGraph, runningMeanTensor, compute_dtype);
        runningVarTensor = castMPSTensor(mpsGraph, runningVarTensor, compute_dtype);
      }
      if (has_save_mean) {
        saveMeanTensor = castMPSTensor(mpsGraph, saveMeanTensor, compute_dtype);
        saveVarTensor = castMPSTensor(mpsGraph, saveVarTensor, compute_dtype);
      }

      MPSGraphTensor* gradInputTensor = nil;
      MPSGraphTensor* gradWeightTensor = nil;
      MPSGraphTensor* gradBiasTensor = nil;
      MPSGraphTensor* inputTensor = nil;

      if (memory_format == at::MemoryFormat::Contiguous)
        inputTensor = inputTensorOriginal;
      else {
        // Reshape/transpose the input as needed
        auto N = input_shape[0];
        auto H = input_shape[1];
        auto W = input_shape[2];
        auto C = input_shape[3];

        inputTensor = [mpsGraph reshapeTensor:inputTensorOriginal
                                    withShape:@[ N, ([NSNumber numberWithInt:[H intValue] * [W intValue]]), C ]
                                         name:nil];
        inputTensor = [mpsGraph transposeTensor:inputTensor dimension:1 withDimension:2 name:nil];
        inputTensor = [mpsGraph reshapeTensor:inputTensor withShape:@[ N, C, H, W ] name:nil];
      }
      inputTensor = castMPSTensor(mpsGraph, inputTensor, compute_dtype);

      if (train) {
        // Use save_mean and save_var
        auto epsilonTensor = [mpsGraph constantWithScalar:(float)epsilon dataType:compute_dtype];
        MPSGraphTensor* revertSaveVarTensor = saveVarTensor;
        revertSaveVarTensor = [mpsGraph reciprocalWithTensor:revertSaveVarTensor name:nil];
        revertSaveVarTensor = [mpsGraph multiplicationWithPrimaryTensor:revertSaveVarTensor
                                                        secondaryTensor:revertSaveVarTensor
                                                                   name:nil];
        revertSaveVarTensor = [mpsGraph subtractionWithPrimaryTensor:revertSaveVarTensor
                                                     secondaryTensor:epsilonTensor
                                                                name:nil];
        if (grad_input_mask[1]) {
          gradWeightTensor = [mpsGraph normalizationGammaGradientWithIncomingGradientTensor:gradOutputTensor
                                                                               sourceTensor:inputTensor
                                                                                 meanTensor:saveMeanTensor
                                                                             varianceTensor:revertSaveVarTensor
                                                                              reductionAxes:axes
                                                                                    epsilon:(float)epsilon
                                                                                       name:nil];
        }
        if (grad_input_mask[2]) {
          gradBiasTensor = [mpsGraph normalizationBetaGradientWithIncomingGradientTensor:gradOutputTensor
                                                                            sourceTensor:inputTensor
                                                                           reductionAxes:axes
                                                                                    name:nil];
        }
        if (grad_input_mask[0]) {
          gradInputTensor = [mpsGraph normalizationGradientWithIncomingGradientTensor:gradOutputTensor
                                                                         sourceTensor:inputTensor
                                                                           meanTensor:saveMeanTensor
                                                                       varianceTensor:revertSaveVarTensor
                                                                          gammaTensor:weightTensorCasted
                                                                  gammaGradientTensor:gradWeightTensor
                                                                   betaGradientTensor:gradBiasTensor
                                                                        reductionAxes:axes
                                                                              epsilon:(float)epsilon
                                                                                 name:nil];
        }
      } else {
        // Use running mean and running var
        MPSGraphTensor* rsqrtTensor = nil;
        MPSGraphTensor* epsilonTensor = nil;
        if (grad_input_mask[1]) {
          epsilonTensor = [mpsGraph constantWithScalar:(float)epsilon shape:@[ @1 ] dataType:compute_dtype];
          MPSGraphTensor* xMinusMean = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                              secondaryTensor:runningMeanTensor
                                                                         name:nil];
          MPSGraphTensor* varianceEpsTensor = [mpsGraph additionWithPrimaryTensor:runningVarTensor
                                                                  secondaryTensor:epsilonTensor
                                                                             name:nil];
          rsqrtTensor = [mpsGraph reciprocalSquareRootWithTensor:varianceEpsTensor name:nil];
          MPSGraphTensor* bnForwardTensor = [mpsGraph multiplicationWithPrimaryTensor:xMinusMean
                                                                      secondaryTensor:rsqrtTensor
                                                                                 name:nil];
          MPSGraphTensor* gradBnMulTensor = [mpsGraph multiplicationWithPrimaryTensor:bnForwardTensor
                                                                      secondaryTensor:gradOutputTensor
                                                                                 name:nil];
          gradWeightTensor = [mpsGraph reductionSumWithTensor:gradBnMulTensor axes:axes name:nil];
        }
        if (grad_input_mask[2]) {
          gradBiasTensor = [mpsGraph normalizationBetaGradientWithIncomingGradientTensor:gradOutputTensor
                                                                            sourceTensor:inputTensor
                                                                           reductionAxes:axes
                                                                                    name:nil];
        }
        if (grad_input_mask[0]) {
          auto unitTensor = [mpsGraph constantWithScalar:1.0 shape:input_shape_readonly dataType:compute_dtype];
          if (!epsilonTensor)
            epsilonTensor = [mpsGraph constantWithScalar:(float)epsilon shape:@[ @1 ] dataType:compute_dtype];
          if (!rsqrtTensor) {
            MPSGraphTensor* varianceEpsTensor = [mpsGraph additionWithPrimaryTensor:runningVarTensor
                                                                    secondaryTensor:epsilonTensor
                                                                               name:nil];
            rsqrtTensor = [mpsGraph reciprocalSquareRootWithTensor:varianceEpsTensor name:nil];
          }

          gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:unitTensor secondaryTensor:rsqrtTensor name:nil];
          if (has_weight)
            gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradInputTensor
                                                        secondaryTensor:weightTensorCasted
                                                                   name:nil];
          gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradInputTensor
                                                      secondaryTensor:gradOutputTensor
                                                                 name:nil];
        }
      }

      if (grad_input_mask[1]) {
        gradWeightTensor = [mpsGraph reshapeTensor:gradWeightTensor
                                         withShape:@[ input_shape_readonly[channelsDim] ]
                                              name:nil];
        gradWeightTensor = castMPSTensor(mpsGraph, gradWeightTensor, weight_mps_dtype);
      }
      if (grad_input_mask[2]) {
        gradBiasTensor = [mpsGraph reshapeTensor:gradBiasTensor
                                       withShape:@[ input_shape_readonly[channelsDim] ]
                                            name:nil];
        gradBiasTensor = castMPSTensor(mpsGraph, gradBiasTensor, weight_mps_dtype);
      }

      if (grad_input_mask[0]) {
        gradInputTensor = castMPSTensor(mpsGraph, gradInputTensor, input_mps_dtype);
      }
      MPSGraphTensor* gradInputTensorFinal = nil;

      if (memory_format == at::MemoryFormat::Contiguous)
        gradInputTensorFinal = gradInputTensor;
      else {
        // Reshape/transpose the input as needed
        auto N = input_shape[0];
        auto H = input_shape[1];
        auto W = input_shape[2];
        auto C = input_shape[3];

        gradInputTensorFinal = [mpsGraph reshapeTensor:gradInputTensor
                                             withShape:@[ N, C, ([NSNumber numberWithInt:[H intValue] * [W intValue]]) ]
                                                  name:nil];
        gradInputTensorFinal = [mpsGraph transposeTensor:gradInputTensorFinal dimension:1 withDimension:2 name:nil];
        gradInputTensorFinal = [mpsGraph reshapeTensor:gradInputTensorFinal withShape:@[ N, H, W, C ] name:nil];
      }

      newCachedGraph->gradInputTensor_ = gradInputTensorFinal;
      newCachedGraph->gradWeightTensor_ = gradWeightTensor;
      newCachedGraph->gradBiasTensor_ = gradBiasTensor;
    });

    // For channels_last, input_shape is the NHWC-packed shape; feed the raw buffer directly instead of
    // viewing the NCHW tensor to NHWC (which view() rejects). Mirrors the forward pass.
    const auto needs_gather = memory_format != MemoryFormat::ChannelsLast;
    auto inputPlaceholder =
        Placeholder(cachedGraph->inputTensor_, input, input_shape, needs_gather, MPSDataTypeInvalid, needs_gather);
    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_out, input_shape_readonly);
    auto weightPlaceholder = Placeholder();
    if (has_weight)
      weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_opt.value(), new_mean_shape);
    auto runningMeanPlaceholder = Placeholder();
    auto runningVarPlaceholder = Placeholder();
    if (has_running_mean) {
      runningMeanPlaceholder = Placeholder(cachedGraph->runningMeanTensor_, running_mean_opt.value(), new_mean_shape);
      runningVarPlaceholder = Placeholder(cachedGraph->runningVarTensor_, running_var_opt.value(), new_mean_shape);
    }
    auto saveMeanPlaceholder = Placeholder();
    auto saveVarPlaceholder = Placeholder();
    if (has_save_mean) {
      saveMeanPlaceholder = Placeholder(cachedGraph->saveMeanTensor_, save_mean_opt.value(), new_mean_shape);
      saveVarPlaceholder = Placeholder(cachedGraph->saveVarTensor_, save_var_opt.value(), new_mean_shape);
    }

    auto gradInputPlaceholder = Placeholder();
    if (grad_input_mask[0])
      gradInputPlaceholder =
          Placeholder(cachedGraph->gradInputTensor_, grad_input, input_shape, false, MPSDataTypeInvalid, needs_gather);
    auto gradWeightPlaceholder = Placeholder();
    if (grad_input_mask[1])
      gradWeightPlaceholder = Placeholder(cachedGraph->gradWeightTensor_, grad_weight);
    auto gradBiasPlaceholder = Placeholder();

    if (grad_input_mask[2])
      gradBiasPlaceholder = Placeholder(cachedGraph->gradBiasTensor_, grad_bias);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
    if (has_weight)
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    if (has_running_mean) {
      feeds[runningMeanPlaceholder.getMPSGraphTensor()] = runningMeanPlaceholder.getMPSGraphTensorData();
      feeds[runningVarPlaceholder.getMPSGraphTensor()] = runningVarPlaceholder.getMPSGraphTensorData();
    }
    if (has_save_mean) {
      feeds[saveMeanPlaceholder.getMPSGraphTensor()] = saveMeanPlaceholder.getMPSGraphTensorData();
      feeds[saveVarPlaceholder.getMPSGraphTensor()] = saveVarPlaceholder.getMPSGraphTensorData();
    }

    NSMutableDictionary* results = [[NSMutableDictionary new] autorelease];
    if (grad_input_mask[0])
      results[gradInputPlaceholder.getMPSGraphTensor()] = gradInputPlaceholder.getMPSGraphTensorData();
    if (grad_input_mask[1])
      results[gradWeightPlaceholder.getMPSGraphTensor()] = gradWeightPlaceholder.getMPSGraphTensorData();
    if (grad_input_mask[2])
      results[gradBiasPlaceholder.getMPSGraphTensor()] = gradBiasPlaceholder.getMPSGraphTensorData();

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// Layer norm forward for MPS
std::tuple<Tensor, Tensor, Tensor> layer_norm_mps(const Tensor& input,
                                                  IntArrayRef normalized_shape,
                                                  const std::optional<Tensor>& weight_opt,
                                                  const std::optional<Tensor>& bias_opt,
                                                  double eps) {
  auto N = c10::multiply_integers(normalized_shape);
  auto out = at::empty_like(input, MemoryFormat::Contiguous);
  auto batch_dim = input.dim() - normalized_shape.size();
  IntArrayRef batch_shape = input.sizes().slice(0, batch_dim);
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto X = input.expect_contiguous();
  // The Metal kernels bind gamma/beta at the input dtype, so mixed-dtype
  // affine params (e.g. fp32 gamma/beta with an fp16 input, the
  // keep-LayerNorm-in-fp32 recipe) must be cast, not reinterpreted.
  const auto bias_contig =
      bias.defined() ? std::make_optional(bias.to(input.scalar_type()).contiguous()) : std::nullopt;
  const auto gamma = weight.defined() ? std::make_optional(weight.to(input.scalar_type()).contiguous()) : std::nullopt;
  auto mean = at::empty(batch_shape, input.options(), MemoryFormat::Contiguous);
  auto rstd = at::empty(batch_shape, input.options(), MemoryFormat::Contiguous);

  auto input_shape = input.sizes();
  uint64_t axis_size = static_cast<uint64_t>(N);
  float epsilon_buf = static_cast<float>(eps);
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int axis = input_ndim - normalized_ndim;
  MPSStream* stream = getCurrentMPSStream();
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() != kLong, "Not implemented for long on MPS");
  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      const int N_READS = 4;
      auto metalType = mps::scalarToMetalTypeString(input);
      // Use 32-bit index math unless the tensor is too large for it.
      const bool use32 = at::native::canUse32BitIndexMath(*X) && at::native::canUse32BitIndexMath(out);
      const char* idx_str = use32 ? "i32" : "i64";
      const char* variant = axis_size <= 1024 * N_READS ? "single_row" : "looped";
      id<MTLComputePipelineState> layerNormKernel =
          mps::lib.getPipelineStateForFunc(fmt::format("layer_norm_{}_{}_{}", variant, idx_str, metalType));
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:layerNormKernel];

      auto setLayerNormArgs = [&](auto idx_tag) {
        using IDX_T = decltype(idx_tag);
        mps::mtl_setArgs(
            computeEncoder, *X, out, mean, rstd, static_cast<IDX_T>(axis_size), epsilon_buf, gamma, bias_contig);
      };
      if (use32) {
        setLayerNormArgs(uint32_t{});
      } else {
        setLayerNormArgs(uint64_t{});
      }
      MTLSize numThreads = MTLSizeMake(std::min<uint64_t>((axis_size + N_READS - 1) / N_READS, 1024), 1, 1);
      MTLSize numThreadgroups = MTLSizeMake(M, 1, 1);
      [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:numThreads];
    });
  }
  out = out.view(input_shape);
  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for ([[maybe_unused]] auto idx : c10::irange(axis, input.dim())) {
    stat_shape.push_back(1);
  }
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  return std::make_tuple(std::move(out), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_mps(const Tensor& grad_out,
                                                           const Tensor& input,
                                                           IntArrayRef normalized_shape,
                                                           const Tensor& mean,
                                                           const Tensor& rstd,
                                                           const std::optional<Tensor>& weight_opt /* optional */,
                                                           const std::optional<Tensor>& bias_opt /* optional */,
                                                           std::array<bool, 3> grad_input_mask) {
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();
  auto dOut = grad_out.expect_contiguous();

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::empty_like(*X,
                                std::nullopt /* dtype */,
                                std::nullopt /* layout */,
                                kMPS /* device */,
                                std::nullopt /* pin_memory */,
                                at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    grad_weight = M > 0 ? at::empty_like(*gamma,
                                         std::nullopt /* dtype */,
                                         std::nullopt /* layout */,
                                         kMPS /* device */,
                                         std::nullopt /* pin_memory */,
                                         at::MemoryFormat::Contiguous)
                        : at::zeros_like(*gamma,
                                         std::nullopt /* dtype */,
                                         std::nullopt /* layout */,
                                         kMPS /* device */,
                                         std::nullopt /* pin_memory */,
                                         at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    grad_bias = M > 0 ? at::empty_like(*beta,
                                       std::nullopt /* dtype */,
                                       std::nullopt /* layout */,
                                       kMPS /* device */,
                                       std::nullopt /* pin_memory */,
                                       at::MemoryFormat::Contiguous)
                      : at::zeros_like(*beta,
                                       std::nullopt /* dtype */,
                                       std::nullopt /* layout */,
                                       kMPS /* device */,
                                       std::nullopt /* pin_memory */,
                                       at::MemoryFormat::Contiguous);
  }
  if (M > 0) {
    using namespace at::native::mps;

    // Derive from MPSCachedGraph
    struct CachedGraph : public MPSCachedGraph {
      CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor* gradOutputTensor_ = nil;
      MPSGraphTensor* inputTensor_ = nil;
      MPSGraphTensor* weightTensor_ = nil;
      MPSGraphTensor* meanTensor_ = nil;
      MPSGraphTensor* rstdTensor_ = nil;
      MPSGraphTensor* gradInputTensor_ = nil;
      MPSGraphTensor* gradWeightTensor_ = nil;
      MPSGraphTensor* gradBiasTensor_ = nil;
    };

    auto stream = at::mps::getCurrentMPSStream();

    const bool has_weight = (weight_opt.has_value() && weight_opt->defined());

    if (X->numel() == 0) {
      return std::make_tuple(grad_input, grad_weight, grad_bias);
    }

    // const auto memory_format = input.suggest_memory_format();

    @autoreleasepool {
      MPSShape* input_shape = mps::getMPSShape(*X);
      MPSShape* gamma_shape = mps::getMPSShape(normalized_shape);

      auto num_normalized_dims = [gamma_shape count];
      auto num_channel_dims = [input_shape count] - num_normalized_dims;

      NSMutableArray<NSNumber*>* gamma_axes = [NSMutableArray<NSNumber*> arrayWithCapacity:num_channel_dims];

      for (const auto i : c10::irange(num_channel_dims))
        gamma_axes[i] = [NSNumber numberWithInt:static_cast<int>(i)];

      // Axes along which to reduce to get "batch norm" gradient
      // This will be applied on shape [1, M, -1]
      NSMutableArray<NSNumber*>* bn_axes = [NSMutableArray<NSNumber*> arrayWithCapacity:num_normalized_dims];
      for (const auto i : c10::irange(num_normalized_dims))
        bn_axes[i] = [NSNumber numberWithInt:static_cast<int>(1 + 1 + i)];

      // Shape of input to do "batch norm" backward
      // This is [1, M, -1]
      NSMutableArray<NSNumber*>* bn_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_normalized_dims + 2)];
      bn_shape[0] = [NSNumber numberWithInt:1];
      bn_shape[1] = [NSNumber numberWithInt:M];
      for (const auto i : c10::irange(num_normalized_dims))
        bn_shape[i + 2] = input_shape[i + num_channel_dims];

      // Shape of mean to do "batch norm" backward
      // This is [1, M, [1,1,1..1]]
      NSMutableArray<NSNumber*>* bn_mean_shape =
          [NSMutableArray<NSNumber*> arrayWithCapacity:(num_normalized_dims + 2)];
      bn_mean_shape[0] = [NSNumber numberWithInt:1];
      bn_mean_shape[1] = [NSNumber numberWithInt:M];
      for (const auto i : c10::irange(num_normalized_dims))
        bn_mean_shape[i + 2] = [NSNumber numberWithInt:1];

      // Shape of gamma to multiply with "batch norm" backward
      // This is [1, 1, -1]
      NSMutableArray<NSNumber*>* bn_gamma_shape =
          [NSMutableArray<NSNumber*> arrayWithCapacity:(num_normalized_dims + 2)];
      bn_gamma_shape[0] = [NSNumber numberWithInt:1];
      bn_gamma_shape[1] = [NSNumber numberWithInt:1];
      for (const auto i : c10::irange(num_normalized_dims))
        bn_gamma_shape[i + 2] = input_shape[i + num_channel_dims];

      std::string key = fmt::format("layer_norm_backward_mps:{}:{}:{}:{}:{}:{}:{}",
                                    has_weight,
                                    getArrayRefString(normalized_shape),
                                    getArrayRefString((*X).sizes()),
                                    c10::Join(",", grad_input_mask),
                                    getMPSTypeString(*X),
                                    has_weight ? getMPSTypeString(*gamma) : "",
                                    bias.defined() ? getMPSTypeString(*beta) : "");
      auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
        MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, *X);
        MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, *dOut);
        MPSGraphTensor* weightTensor = nil;
        if (has_weight)
          weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, *gamma);
        // Compute in the input dtype; MPSGraph rejects mixed-dtype arithmetic
        // (fp32 gamma with an fp16 input asserts inside the MLIR verifier).
        MPSGraphTensor* weightComputeTensor =
            has_weight ? castMPSTensor(mpsGraph, weightTensor, getMPSDataType(*X)) : nil;

        // Mean and inv std tensors to be saved and returned
        MPSGraphTensor* meanTensor = mpsGraphRankedPlaceHolder(mpsGraph, mean);
        MPSGraphTensor* rstdTensor = mpsGraphRankedPlaceHolder(mpsGraph, rstd);

        MPSGraphTensor* gradInputTensor = nil;
        MPSGraphTensor* gradWeightTensor = nil;
        MPSGraphTensor* gradBiasTensor = nil;

        if (grad_input_mask[1]) {
          MPSGraphTensor* xMinusMean = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                              secondaryTensor:meanTensor
                                                                         name:nil];
          MPSGraphTensor* bnForwardTensor = [mpsGraph multiplicationWithPrimaryTensor:xMinusMean
                                                                      secondaryTensor:rstdTensor
                                                                                 name:nil];
          MPSGraphTensor* gradBnMulTensor = [mpsGraph multiplicationWithPrimaryTensor:bnForwardTensor
                                                                      secondaryTensor:gradOutputTensor
                                                                                 name:nil];
          gradWeightTensor = [mpsGraph reductionSumWithTensor:gradBnMulTensor axes:gamma_axes name:nil];
        }
        if (grad_input_mask[2]) {
          gradBiasTensor = [mpsGraph reductionSumWithTensor:gradOutputTensor axes:gamma_axes name:nil];
        }
        if (grad_input_mask[0]) {
          // Reshape input to [1, M, -1]
          // Reshape mean and rstd to [1, M, -1]
          // Reshape gamma to [1, 1, -1] (-1 has N dims)

          MPSGraphTensor* bnInputTensor = [mpsGraph reshapeTensor:inputTensor withShape:bn_shape name:nil];
          MPSGraphTensor* bnGradOutputTensor = [mpsGraph reshapeTensor:gradOutputTensor withShape:bn_shape name:nil];
          // Do this at the end
          if (has_weight) {
            MPSGraphTensor* bnGammaTensor = [mpsGraph reshapeTensor:weightComputeTensor
                                                          withShape:bn_gamma_shape
                                                               name:nil];
            bnGradOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:bnGradOutputTensor
                                                           secondaryTensor:bnGammaTensor
                                                                      name:nil];
          }
          MPSGraphTensor* bnMeanTensor = [mpsGraph reshapeTensor:meanTensor withShape:bn_mean_shape name:nil];
          MPSGraphTensor* bnRstdTensor = [mpsGraph reshapeTensor:rstdTensor withShape:bn_mean_shape name:nil];

          MPSGraphTensor* mulTensor = [mpsGraph constantWithScalar:N shape:@[ @1 ] dataType:MPSDataTypeInt32];

          MPSGraphTensor* numberToReduceTensor = mulTensor;

          MPSGraphTensor* cast2Tensor = [mpsGraph castTensor:numberToReduceTensor
                                                      toType:bnInputTensor.dataType
                                                        name:@"cast2Tensor"];

          MPSGraphTensor* sizeReciprocalTensor = [mpsGraph reciprocalWithTensor:cast2Tensor name:nil];

          // TODO: Reduce redundant computation
          MPSGraphTensor* xMinusMean = [mpsGraph subtractionWithPrimaryTensor:bnInputTensor
                                                              secondaryTensor:bnMeanTensor
                                                                         name:nil];

          MPSGraphTensor* normalizedTensor = [mpsGraph multiplicationWithPrimaryTensor:xMinusMean
                                                                       secondaryTensor:bnRstdTensor
                                                                                  name:nil];

          MPSGraphTensor* bnGradMulTensor = [mpsGraph multiplicationWithPrimaryTensor:bnGradOutputTensor
                                                                      secondaryTensor:normalizedTensor
                                                                                 name:nil];

          MPSGraphTensor* gammaGradient = [mpsGraph reductionSumWithTensor:bnGradMulTensor axes:bn_axes name:nil];

          MPSGraphTensor* betaGradient = [mpsGraph reductionSumWithTensor:bnGradOutputTensor axes:bn_axes name:nil];

          MPSGraphTensor* gradient1 = [mpsGraph multiplicationWithPrimaryTensor:bnGradOutputTensor
                                                                secondaryTensor:bnRstdTensor
                                                                           name:nil];

          MPSGraphTensor* gradient2_1 = [mpsGraph multiplicationWithPrimaryTensor:sizeReciprocalTensor
                                                                  secondaryTensor:xMinusMean
                                                                             name:nil];

          // reverseVariance is square of rstd
          MPSGraphTensor* reverseVariance = [mpsGraph squareWithTensor:bnRstdTensor name:nil];
          MPSGraphTensor* gradient2_2 = [mpsGraph multiplicationWithPrimaryTensor:gammaGradient
                                                                  secondaryTensor:reverseVariance
                                                                             name:nil];

          MPSGraphTensor* gradient2 = [mpsGraph multiplicationWithPrimaryTensor:gradient2_1
                                                                secondaryTensor:gradient2_2
                                                                           name:nil];

          MPSGraphTensor* gradient3_1 = [mpsGraph multiplicationWithPrimaryTensor:sizeReciprocalTensor
                                                                  secondaryTensor:betaGradient
                                                                             name:nil];

          MPSGraphTensor* gradient3 = [mpsGraph multiplicationWithPrimaryTensor:gradient3_1
                                                                secondaryTensor:bnRstdTensor
                                                                           name:nil];

          MPSGraphTensor* gradient4 = [mpsGraph subtractionWithPrimaryTensor:gradient1
                                                             secondaryTensor:gradient2
                                                                        name:nil];

          MPSGraphTensor* gradient = [mpsGraph subtractionWithPrimaryTensor:gradient4
                                                            secondaryTensor:gradient3
                                                                       name:nil];

          gradInputTensor = [mpsGraph reshapeTensor:gradient withShape:input_shape name:nil];
        }

        // Gradients are computed in the input dtype; return them in the
        // affine params' dtype like CPU does.
        if (grad_input_mask[1]) {
          gradWeightTensor = [mpsGraph reshapeTensor:castMPSTensor(mpsGraph, gradWeightTensor, getMPSDataType(*gamma))
                                           withShape:gamma_shape
                                                name:nil];
        }
        if (grad_input_mask[2]) {
          gradBiasTensor = [mpsGraph reshapeTensor:castMPSTensor(mpsGraph, gradBiasTensor, getMPSDataType(*beta))
                                         withShape:gamma_shape
                                              name:nil];
        }

        newCachedGraph->gradOutputTensor_ = gradOutputTensor;
        newCachedGraph->inputTensor_ = inputTensor;
        newCachedGraph->weightTensor_ = weightTensor;
        newCachedGraph->meanTensor_ = meanTensor;
        newCachedGraph->rstdTensor_ = rstdTensor;
        newCachedGraph->gradInputTensor_ = gradInputTensor;
        newCachedGraph->gradWeightTensor_ = gradWeightTensor;
        newCachedGraph->gradBiasTensor_ = gradBiasTensor;
      });

      auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, *X);
      auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, *dOut);
      auto weightPlaceholder = Placeholder();
      if (has_weight)
        weightPlaceholder = Placeholder(cachedGraph->weightTensor_, *gamma);
      auto saveMeanPlaceholder = Placeholder(cachedGraph->meanTensor_, mean);
      auto saveVarPlaceholder = Placeholder(cachedGraph->rstdTensor_, rstd);

      auto gradInputPlaceholder = Placeholder();
      if (grad_input_mask[0])
        gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);
      auto gradWeightPlaceholder = Placeholder();
      if (grad_input_mask[1])
        gradWeightPlaceholder = Placeholder(cachedGraph->gradWeightTensor_, grad_weight);
      auto gradBiasPlaceholder = Placeholder();
      ;
      if (grad_input_mask[2])
        gradBiasPlaceholder = Placeholder(cachedGraph->gradBiasTensor_, grad_bias);

      NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
      feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
      feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
      if (has_weight)
        feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
      feeds[saveMeanPlaceholder.getMPSGraphTensor()] = saveMeanPlaceholder.getMPSGraphTensorData();
      feeds[saveVarPlaceholder.getMPSGraphTensor()] = saveVarPlaceholder.getMPSGraphTensorData();

      NSMutableDictionary* results = [[NSMutableDictionary new] autorelease];
      if (grad_input_mask[0])
        results[gradInputPlaceholder.getMPSGraphTensor()] = gradInputPlaceholder.getMPSGraphTensorData();
      if (grad_input_mask[1])
        results[gradWeightPlaceholder.getMPSGraphTensor()] = gradWeightPlaceholder.getMPSGraphTensorData();
      if (grad_input_mask[2])
        results[gradBiasPlaceholder.getMPSGraphTensor()] = gradBiasPlaceholder.getMPSGraphTensorData();

      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

} // namespace at::native
