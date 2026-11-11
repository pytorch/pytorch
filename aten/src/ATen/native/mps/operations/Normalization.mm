//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/BatchNorm.h>
#include <c10/util/safe_conv.h>

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
#include <ATen/native/mps/Normalization_metallib.h>
#endif
using namespace std::string_view_literals;

static std::string_view batch_norm_variant(const Tensor& x, const Tensor& out) {
  if (!offsetsFitIn<uint32_t>(x, out)) {
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

static Tensor batch_norm_reduce(const Tensor& x,
                                const std::optional<Tensor>& dy = std::nullopt,
                                const std::optional<Tensor>& mean = std::nullopt) {
  constexpr int64_t min_elems_per_thread = 16;
  const auto max_groups = occupancy_threadgroups();
  const auto layout = batch_norm_layout(x, dy.value_or(x));
  const int64_t C = layout.channel.x, inner = layout.inner.x, outer = layout.outer_numel;
  const bool channels_last = inner == 1;
  const auto threads = channels_last
      ? c10::metal::max_threadgroup_size
      : std::clamp<int64_t>(
            outer * inner / min_elems_per_thread, c10::metal::simdgroup_size, c10::metal::max_threadgroup_size);
  const auto tx = channels_last ? c10::metal::simdgroup_size : std::min(inner, threads);
  const auto ty = std::clamp<int64_t>(outer, 1, threads / tx);
  const auto blocks = channels_last ? c10::metal::ceil_div(C, tx) : C;
  const auto chunks = std::max<int64_t>(1, std::min(max_groups / blocks, c10::metal::ceil_div(outer, ty)));
  const auto splits = std::max<int64_t>(1, std::min(max_groups / (blocks * chunks), c10::metal::ceil_div(inner, tx)));
  auto out = at::empty({chunks * splits, C, BN_PARTIAL_SIZE}, x.options().dtype(kFloat));
  const auto name = fmt::format("batch_norm_{}reduce_{}{}",
                                dy ? "backward_"sv : ""sv,
                                batch_norm_variant(x, dy.value_or(x)),
                                scalarToMetalTypeString(x));
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    @autoreleasepool {
      auto encoder = getCurrentMPSStream()->commandEncoder();
      [encoder setComputePipelineState:lib.getPipelineStateForFunc(name)];
      mtl_setArgs(encoder, x, out, layout, dy, mean);
      [encoder dispatchThreadgroups:MTLSizeMake(blocks, chunks, splits) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
    }
  });
  return out;
}

template <size_t N>
static Tensor batch_norm_stats(const Tensor& x,
                               const std::optional<Tensor>& part,
                               std::array<std::optional<Tensor>, N> tensors,
                               double eps,
                               double momentum,
                               std::optional<bool> backward_train = std::nullopt) {
  const auto C = c10::safe_conv<uint32_t>(x.size(1), "uint32_t");
  const auto dtype =
      (backward_train ? tensors[bn_idx(BNBackwardParam::Mean)] : tensors[bn_idx(BNParam::SaveMean)])->scalar_type();
  for (auto& t : tensors) {
    t = t.has_value() && t->defined() ? t : std::nullopt;
    TORCH_CHECK_TYPE(!t || t->scalar_type() == dtype, "weight, bias, running and saved stats must share a dtype");
  }
  BatchNormStatsParams params{.num_channels = C,
                              .num_partials = part ? c10::safe_conv<uint32_t>(part->size(0)) : 0u,
                              .eps = static_cast<float>(eps),
                              .momentum = static_cast<float>(momentum)};
  std::ranges::transform(tensors, params.strides.begin(), [](const auto& t) { return t ? t->stride(0) : 0; });
  auto stats = at::empty({C, bn_idx(BNStat::NumStats)}, x.options().dtype(kFloat));
  const auto name =
      fmt::format("batch_norm_{}stats_{}", backward_train ? "backward_"sv : ""sv, scalarToMetalTypeString(dtype));
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    @autoreleasepool {
      auto encoder = getCurrentMPSStream()->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(name);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, part, stats, params);
      std::apply([&](const auto&... t) { mtl_setArgs<3>(encoder, t...); }, tensors);
      if (backward_train) {
        mtl_setArgs<3 + N>(encoder, *backward_train);
      }
      mtl_dispatch1DJob(encoder, pso, C);
    }
  });
  return stats;
}

static void batch_norm_elemt(const Tensor& out,
                             const Tensor& x,
                             const Tensor& stats,
                             const std::optional<Tensor>& dy = std::nullopt) {
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
      mtl_setArgs(encoder, out, x, stats, layout, dy);
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
  const auto stats = mps::batch_norm_stats<bn_idx(BNParam::NumParams)>(
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
  const auto& mean = train ? save_mean_opt : running_mean_opt;
  const auto& var = train ? save_var_opt : running_var_opt;
  TORCH_CHECK_VALUE(mean.has_value() && mean->defined() && var.has_value() && var->defined(),
                    train ? "save_mean and save_var" : "running_mean and running_var",
                    " must be defined");
  auto grad_input = grad_input_mask[0] ? at::empty_like(input) : Tensor();
  auto grad_weight = grad_input_mask[1] ? at::empty({input.size(1)}, mean->options()) : Tensor();
  auto grad_bias = grad_input_mask[2] ? at::empty({input.size(1)}, mean->options()) : Tensor();
  if (input.numel() == 0) {
    return std::make_tuple(grad_input,
                           grad_weight.defined() ? grad_weight.zero_() : grad_weight,
                           grad_bias.defined() ? grad_bias.zero_() : grad_bias);
  }
  const bool reuse_grad_out = grad_out.scalar_type() == input.scalar_type() &&
      (!train || !grad_input.defined() || grad_out.strides() == grad_input.strides());
  const auto dy = reuse_grad_out ? grad_out : at::empty_like(input).copy_(grad_out);
  const auto part = train || grad_input_mask[1] || grad_input_mask[2]
      ? std::optional<Tensor>(mps::batch_norm_reduce(input, dy, mean->to(kFloat).contiguous()))
      : std::nullopt;
  const auto stats = mps::batch_norm_stats<bn_idx(BNBackwardParam::NumParams)>(
      input, part, {weight_opt, mean, var, grad_weight, grad_bias}, epsilon, 0, train);
  if (grad_input.defined()) {
    train ? mps::batch_norm_elemt(grad_input, input, stats, dy) : mps::batch_norm_elemt(grad_input, dy, stats);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
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
  using namespace mps;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const auto [M, N] = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto X = input.expect_contiguous();
  auto dOut = grad_out.expect_contiguous();
  const auto dtype = X->scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(
      supportedFloatingType(dtype), "native_layer_norm_backward: unsupported dtype ", dtype, " on MPS");
  TORCH_CHECK_TYPE(dOut->scalar_type() == dtype,
                   "native_layer_norm_backward: expected grad_out dtype ",
                   dtype,
                   " but got ",
                   dOut->scalar_type());
  TORCH_CHECK_VALUE(dOut->numel() == M * N && mean.numel() == M && rstd.numel() == M,
                    "native_layer_norm_backward: grad_out, mean and rstd must match the input");
  if (grad_input_mask[1] && grad_input_mask[2]) {
    TORCH_CHECK_TYPE(weight.scalar_type() == bias.scalar_type(),
                     "native_layer_norm_backward: weight and bias gradients are computed in one dtype, got ",
                     weight.scalar_type(),
                     " and ",
                     bias.scalar_type());
  }

  Tensor grad_input, grad_weight, grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::empty_like(*X, MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    grad_weight =
        M > 0 ? at::empty_like(weight, MemoryFormat::Contiguous) : at::zeros_like(weight, MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    grad_bias = M > 0 ? at::empty_like(bias, MemoryFormat::Contiguous) : at::zeros_like(bias, MemoryFormat::Contiguous);
  }
  if (X->numel() == 0) {
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
  }

  const auto mean_c = mean.to(dtype).contiguous();
  const auto rstd_c = rstd.to(dtype).contiguous();
  const auto gamma = weight.defined() ? std::make_optional(weight.to(dtype).contiguous()) : std::nullopt;
  const auto rows = c10::safe_conv<uint32_t>(M, "layer_norm rows");
  const auto row_size = c10::safe_conv<uint32_t>(N, "layer_norm normalized size");
  const auto threads =
      std::min(c10::metal::ceil_div(row_size, c10::metal::ILP_PER_THREAD), c10::metal::max_threadgroup_size);
  const auto variant = threads * c10::metal::ILP_PER_THREAD < row_size ? "looped"sv : "single_row"sv;
  const auto col_groups = c10::metal::ceil_div(row_size, c10::metal::simdgroup_size);
  const auto splits = std::clamp<uint32_t>(
      occupancy_threadgroups() / col_groups, 1, c10::metal::ceil_div(rows, c10::metal::simdgroup_size));
  const auto has_affine_grads = grad_weight.defined() || grad_bias.defined();
  const auto dgamma = grad_weight.defined() ? std::make_optional(grad_weight) : std::nullopt;
  const auto dbeta = grad_bias.defined() ? std::make_optional(grad_bias) : std::nullopt;
  constexpr int64_t affine_grads = 2;
  const auto partials = has_affine_grads ? at::empty({splits, N, affine_grads}, X->options().dtype(kFloat)) : Tensor();
  const auto type_str = scalarToMetalTypeString(*X);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto enc = stream->commandEncoder();
      if (grad_input.defined()) {
        auto pso = lib.getPipelineStateForFunc(fmt::format("layer_norm_backward_{}_{}", variant, type_str));
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, *dOut, *X, mean_c, rstd_c, grad_input, row_size, gamma);
        [enc dispatchThreadgroups:MTLSizeMake(1, rows, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
      if (has_affine_grads) {
        auto pso = lib.getPipelineStateForFunc(fmt::format("layer_norm_backward_affine_{}", type_str));
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, *dOut, *X, partials, std::array<uint32_t, 2>{rows, row_size}, mean_c, rstd_c);
        [enc dispatchThreadgroups:MTLSizeMake(col_groups, splits, 1)
            threadsPerThreadgroup:MTLSizeMake(c10::metal::simdgroup_size, c10::metal::simdgroup_size, 1)];
        const auto& affine = grad_weight.defined() ? grad_weight : grad_bias;
        pso = lib.getPipelineStateForFunc(
            fmt::format("layer_norm_backward_affine_combine_{}", scalarToMetalTypeString(affine)));
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, partials, dgamma, dbeta, std::array<uint32_t, 2>{splits, row_size});
        mtl_dispatch1DJob(enc, pso, row_size);
      }
    }
  });
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

} // namespace at::native
