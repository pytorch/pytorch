// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

// batching rules translated from jax: https://github.com/google/jax/blob/master/jax/_src/lax/lax.py#L3143

// Does not support batch_group_count (needed for convolution backwards)
std::tuple<Tensor,optional<int64_t>>
convolution_batch_rule(const Tensor& lhs, optional<int64_t> lhs_bdim, const Tensor& rhs, optional<int64_t> rhs_bdim, const optional<Tensor>& bias, optional<int64_t> bias_bdim, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  DimVector lhs_spec(stride.size() + 2);
  std::iota(lhs_spec.begin(), lhs_spec.end(), 0);
  DimVector rhs_spec = lhs_spec;
  DimVector out_spec = lhs_spec;
  if (transposed) {
    rhs_spec[0] = 1;
    rhs_spec[1] = 0;
  }

  // If we have a batched bias or weight, we need to perform the computation separately.
  optional<Tensor> unbatched_bias;
  bool separate_bias;
  if ((rhs_bdim && bias && bias->defined()) || bias_bdim) {
    TORCH_INTERNAL_ASSERT(bias.has_value());
    TORCH_INTERNAL_ASSERT(bias->defined());
    unbatched_bias = nullopt;
    separate_bias = true;
  } else {
    unbatched_bias = bias;
    separate_bias = false;
  }
  std::tuple<Tensor, optional<int64_t>> result;
  if (lhs_bdim && !rhs_bdim) {
    auto new_x = reshape_dim_into(*lhs_bdim, lhs_spec[0], lhs);
    auto out = at::convolution(new_x, rhs, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
    out = reshape_dim_outof(out_spec[0], lhs.sizes()[*lhs_bdim], out);
    result = std::make_tuple(out, out_spec[0]);
  } else if (!lhs_bdim && rhs_bdim) {
    if (groups == 1) {
      auto new_w = reshape_dim_into(*rhs_bdim, rhs_spec[0], rhs);
      auto out = at::convolution(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
      out = reshape_dim_outof(out_spec[1], rhs.sizes()[*rhs_bdim], out);
      result = std::make_tuple(out, out_spec[1]);
    } else {
      auto dim_with_groups = transposed ? 1 : 0;
      auto new_w = reshape_dim_outof(rhs_spec[dim_with_groups] + (*rhs_bdim <= rhs_spec[0]), groups, rhs);
      new_w = reshape_dim_into(*rhs_bdim + (rhs_spec[0] < rhs_bdim), rhs_spec[0] + 1, new_w);
      new_w = reshape_dim_into(rhs_spec[0], rhs_spec[0], new_w);
      auto out = at::convolution(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
      out = reshape_dim_outof(out_spec[1], groups, out);
      out = reshape_dim_outof(out_spec[1] + 1, rhs.sizes()[*rhs_bdim], out);
      out = reshape_dim_into(out_spec[1], out_spec[1] + 1, out);
      result = std::make_tuple(out, out_spec[1]);
    }
  } else if (lhs_bdim && rhs_bdim) {
    auto new_x = reshape_dim_into(*lhs_bdim, lhs_spec[1], lhs);
    groups *= lhs.sizes()[*lhs_bdim];
    auto dim_with_groups = transposed ? 1 : 0;
    auto new_w = reshape_dim_into(*rhs_bdim, rhs_spec[dim_with_groups], rhs);
    auto out = at::convolution(new_x, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
    out = reshape_dim_outof(out_spec[1], lhs.sizes()[*lhs_bdim], out);
    result = std::make_tuple(out, out_spec[1]);
  } else {
    result = std::make_tuple(at::convolution(lhs, rhs, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups), nullopt);
  }
  if (separate_bias) {
    auto A = std::get<0>(result);
    auto A_batch_dim = std::get<1>(result);
    auto B = *bias;
    auto B_batch_dim = bias_bdim;
    A = moveBatchDimToFront(A, A_batch_dim);
    B = moveBatchDimToFront(B, B_batch_dim);
    for (size_t i = 0; i < out_spec.size() - 2; i++) {
      B = B.unsqueeze(-1);
    }
    B = maybePadToLogicalRank(B, B_batch_dim, rankWithoutBatchDim(A, A_batch_dim));

    return std::make_tuple(at::add(A, B), 0);
  } else {
    return result;
  }
}

Tensor _convolution_decomp(
    const Tensor& input_r, const Tensor& weight_r, const c10::optional<Tensor>& bias_r_opt,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  // Ignore everything. If the user called this in the normal way,
  // then they should be fine.
  (void*) benchmark;
  (void*) deterministic;
  (void*) cudnn_enabled;
  (void*) allow_tf32;
  return at::convolution(
      input_r, weight_r, bias_r_opt, stride_, padding_, dilation_, transposed_, output_padding_, groups_);
}

// Tensor convNd_transpose_decomp(const Tensor &self, const Tensor &weight, const optional<Tensor>& bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
//   std::vector<int64_t> t(self.dim() - 2, 0);
//   IntArrayRef out_padding(t);
//   return at::convolution(self, weight, bias, stride, padding, dilation, true, out_padding, groups);
// }

Tensor mkldnn_convolution_decomp(const Tensor &self, const Tensor &weight, const optional<Tensor>& bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  std::vector<int64_t> t(self.dim() - 2, 0);
  IntArrayRef out_padding(t);
  return at::convolution(self, weight, bias, stride, padding, dilation, false, out_padding, groups);
}

Tensor cudnn_convolution_plumbing(
    const Tensor & self, const Tensor & weight, IntArrayRef padding,
    IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);

  // conv2d that we have a batch rule for
  if (self.dim() == 4) {
    // Contiguous because usually conv is followed by BN and BN calls .contiguous
    // which can fail due to https://github.com/facebookresearch/functorch/issues/55
    return at::conv2d(self, weight, nullopt, stride, padding, dilation, groups).contiguous();
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::cudnn_convolution", "");
  return slow_fallback<Tensor>(op, { self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32 });
}

bool first_dim_has_size_1(const Tensor& value, int64_t bdim) {
  if (bdim == 0) {
    return value.size(1) == 1;
  }
  return value.size(0) == 1;
}

std::tuple<Tensor,int64_t,Tensor,int64_t> cudnn_conv_per_sample_grad_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark,
    bool deterministic, bool allow_tf32, std::array<bool, 2> output_mask) {
  TORCH_INTERNAL_ASSERT(self_bdim && grad_output_bdim && !weight_bdim);
  // TODO: No clue if this works if the first non-batch dim isn't size 1
  TORCH_INTERNAL_ASSERT(first_dim_has_size_1(self, *self_bdim));
  TORCH_INTERNAL_ASSERT(self.dim() == 5);

  auto bdim_size = self.size(*self_bdim);
  auto self_ = reshape_dim_into(*self_bdim, 0, self);
  auto in_channels = self_.size(1);
  auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);

  auto grad_self = at::cudnn_convolution_backward_input(
      self_.sizes(), grad_output_, weight,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  grad_self = reshape_dim_outof(0, bdim_size, grad_self);

  // Copied from https://github.com/pytorch/opacus/blob/master/opacus/grad_sample/conv.py
  auto A = at::im2col(self_, {weight.size(2), weight.size(3)}, dilation, padding, stride);
  auto B = grad_output_.reshape({bdim_size, -1, A.size(-1)});
  auto grad_sample = at::einsum("noq,npq->nop", {B, A});
  grad_sample = grad_sample.view({
      bdim_size, groups, -1, groups, in_channels / groups,
      weight.size(2) * weight.size(3) });
  grad_sample = at::einsum("ngrg...->ngr...", {grad_sample});
  grad_sample = grad_sample.reshape(
      {bdim_size, weight.size(0), weight.size(1), weight.size(2), weight.size(3)});

  return std::make_tuple(grad_self, 0, grad_sample, 0);
}

std::tuple<Tensor,Tensor> cudnn_convolution_backward_plumbing(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, std::array<bool, 2> output_mask) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);

  if (self_bdim.has_value() && self_value.dim() == 5 && first_dim_has_size_1(self_value, *self_bdim) && grad_output_bdim.has_value() && !weight_bdim.has_value()) {
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto result = cudnn_conv_per_sample_grad_rule(
        self_value, self_bdim,
        grad_output_value, grad_output_bdim,
        weight_value, weight_bdim,
        padding, stride, dilation, groups,
        benchmark, deterministic, allow_tf32, output_mask);
    return std::make_tuple(
        makeBatched(std::get<0>(result), std::get<1>(result), cur_level),
        makeBatched(std::get<2>(result), std::get<3>(result), cur_level));
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::cudnn_convolution_backward", "");
  return slow_fallback<Tensor,Tensor>(op, { self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask });
}

std::tuple<Tensor,optional<int64_t>> embedding_batch_rule(
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& indices, optional<int64_t> indices_bdim,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (!weight_bdim && indices_bdim) {
    // B*, ED -> B*D
    const auto result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
    return std::make_tuple(result, indices_bdim);
  } else if (weight_bdim && !indices_bdim) {
    // *, BED -> *, E(BD) -> *(BD) -> *BD
    const auto batch_size = weight.size(*weight_bdim);
    const auto weight_ = reshape_dim_into(*weight_bdim, /*embedding_dim*/1, weight);
    auto result = at::embedding(weight_, indices, padding_idx, scale_grad_by_freq, sparse);
    result = reshape_dim_outof(-1, batch_size, result);
    return std::make_tuple(result, result.dim() - 2);
  }
  TORCH_INTERNAL_ASSERT(weight_bdim && indices_bdim);
  // B*, BED -> B*, (BE)D -> B*D
  // We'll need to do something extra: add (0, E, 2*E, ...) to the indices.
  const auto batch_size = weight.size(*weight_bdim);
  const auto num_embeddings = weight.size((*weight_bdim == 0) ? 1 : 0);
  const auto weight_ = reshape_dim_into(*weight_bdim, 0, weight);
  auto indices_ = moveBatchDimToFront(indices, indices_bdim);

  // [batch_size, 1, 1, 1, ..., 1]
  DimVector view_shape(indices_.dim(), 1);
  view_shape[0] = batch_size;

  auto range = at::arange(0, batch_size * num_embeddings, num_embeddings, indices_.options());
  range = range.view(view_shape);

  indices_ = indices_ + range;
  const auto result = at::embedding(weight_, indices_, padding_idx, scale_grad_by_freq, sparse);
  return std::make_tuple(result, 0);
}

/**
 * grid sample batch rule breaks down into 3 cases:
 *   case 1 (input is batched, grid is not):
 *     batch input along first dimension, unpack along first dimension
 *     2d:
 *       input: N(BC)H_{in}W_{in}, grid: NH_{out}W_{out}2
 *       output: N(BC)H_{out}W_{out}
 *     3d:
 *       input: N(BC)D_{in}H_{in}W_{in}, grid: ND_{out}H_{out}W_{out}3
 *       output: N(BC)D_{out}H_{out}W_{out}
 *   case 2 (input is not batched, grid is batched):
 *     batch grid along second dimension, unpack along second dimension
 *     2d:
 *       input: NCH_{in}W_{in}, grid: N(BH_{out})W_{out}2
 *       output: NC(BH_{out})W_{out}
 *     3d:
 *       input: NCD_{in}H_{in}W_{in}, grid: N(BD_{out})H_{out}W_{out}3
 *       output: NC(BD_{out})H_{out}W_{out}
 *   case 3 (input and grid are both batched):
 *     batch grid and input along 0th dimension, unpack along 0th dimension
 *     2d:
 *       input: (BN)CH_{in}W_{in}, grid: (BN)H_{out}W_{out}2
 *       output: (BN)CH_{out}W_{out}
 *     3d:
 *       input: (BN)CD_{in}H_{in}W_{in}, grid: (BN)D_{out}H_{out}W_{out}3
 *       output: (BN)CD_{out}H_{out}W_{out}
 */
template<typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>>
grid_sample_batch_rule(const Tensor& input, optional<int64_t> input_bdim, const Tensor& grid, optional<int64_t> grid_bdim, ExtraArgs... extra_args) {
  std::tuple<Tensor, optional<int64_t>> result;
  if (input_bdim && !grid_bdim) {
    auto new_input = reshape_dim_into(*input_bdim, 1, input);
    auto out = Func(new_input, grid, std::forward<ExtraArgs>(extra_args)...);
    out = reshape_dim_outof(1, input.sizes()[*input_bdim], out);
    result = std::make_tuple(out, 1);
  } else if (!input_bdim && grid_bdim) {
    // grid of N(BH)W2 -> NC(BH)W or grid of N(BD)HBW3 -> NC(BD)HW
    auto new_grid = reshape_dim_into(*grid_bdim, 1, grid);
    auto out = Func(input, new_grid, std::forward<ExtraArgs>(extra_args)...);
    out = reshape_dim_outof(2, grid.sizes()[*grid_bdim], out);
    result = std::make_tuple(out, 2);
  } else if (input_bdim && grid_bdim) {
    auto new_input = reshape_dim_into(*input_bdim, 0, input);
    auto new_grid = reshape_dim_into(*grid_bdim, 0, grid);
    auto out = Func(new_input, new_grid, std::forward<ExtraArgs>(extra_args)...);
    out = reshape_dim_outof(0, input.sizes()[*grid_bdim], out);
    result = std::make_tuple(out, 0);
  } else {
    result = std::make_tuple(Func(input, grid, std::forward<ExtraArgs>(extra_args)...), nullopt);
  }
  return result;
}

std::tuple<Tensor, Tensor, Tensor, int64_t>
grid_sample_backward_helper_in(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& grid, optional<int64_t> grid_bdim) {

  auto batch_size = get_bdim_size3(
      grad_output, grad_output_bdim, input, input_bdim, grid, grid_bdim);

  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  grad_output_ = ensure_has_bdim(grad_output_, grad_output_bdim.has_value(), batch_size);
  grad_output_ = reshape_dim_into(0, 0, grad_output_);

  auto input_ = moveBatchDimToFront(input, input_bdim);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);
  input_ = reshape_dim_into(0, 0, input_);

  auto grid_ = moveBatchDimToFront(grid, grid_bdim);
  grid_ = ensure_has_bdim(grid_, grid_bdim.has_value(), batch_size);
  grid_ = reshape_dim_into(0, 0, grid_);

  return std::make_tuple(grad_output_, input_, grid_, batch_size);
}

std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
grid_sample_backward_helper_out(
    const std::tuple<Tensor, Tensor> & bw_out,
    optional<int64_t> grad_input_out_bdim,
    optional<int64_t> grad_grid_out_bdim,
    int64_t bdim_size) {
  auto grad_input = std::get<0>(bw_out);
  auto grad_grid = std::get<1>(bw_out);
  grad_input = reshape_dim_outof(*grad_input_out_bdim, bdim_size, grad_input);
  grad_grid = reshape_dim_outof(*grad_grid_out_bdim, bdim_size, grad_grid);
  auto result = std::make_tuple(grad_input, grad_input_out_bdim, grad_grid, grad_grid_out_bdim);
  return result;
}


template<typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
grid_sample_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& grid, optional<int64_t> grid_bdim,
    ExtraArgs... extra_args) {

  auto new_bw_input = grid_sample_backward_helper_in(
      grad_output, grad_output_bdim, input, input_bdim, grid, grid_bdim);

  auto new_grad_output = std::get<0>(new_bw_input);
  auto new_input = std::get<1>(new_bw_input);
  auto new_grid = std::get<2>(new_bw_input);
  int64_t batch_size = std::get<3>(new_bw_input);

  auto bw_out = Func(new_grad_output, new_input, new_grid, std::forward<ExtraArgs>(extra_args)...);

  return grid_sample_backward_helper_out(bw_out, 0, 0, batch_size);
}

template<typename F, F Func>
std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
cudnn_grid_sample_backward_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& grid, optional<int64_t> grid_bdim,
    const Tensor& grad_output, optional<int64_t> grad_output_bdim) {

  auto new_bw_input = grid_sample_backward_helper_in(
      grad_output, grad_output_bdim, input, input_bdim, grid, grid_bdim);

  auto new_grad_output = std::get<0>(new_bw_input);
  auto new_input = std::get<1>(new_bw_input);
  auto new_grid = std::get<2>(new_bw_input);
  int64_t bdim_size = std::get<3>(new_bw_input);

  auto bw_out = Func(new_input, new_grid, new_grad_output);

  return grid_sample_backward_helper_out(bw_out, 0, 0, bdim_size);
}

std::tuple<Tensor, optional<int64_t>> cross_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    const optional<int64_t> dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);

  if (other_bdim.has_value() && !self_bdim.has_value()) {
    self_ = self_.expand_as(other_);
  }
  if (self_bdim.has_value() && !other_bdim.has_value()) {
    other_ = other_.expand_as(self_);
  }
  auto new_dim = dim;
  if (dim.has_value()) {
    auto t = (self_bdim.has_value()) ? self_ : other_;
    bool flag = (self_bdim.has_value()) ? true : other_bdim.has_value();
    new_dim = getPhysicalDim(t, flag, *dim);
  } else {
    // if batch size is 3 we have to avoid that bdim is used as cross' dim argument
    // according to cross API:
    // > If dim is not given, it defaults to the first dimension found with the size 3
    // we have to skip batch dim and find another dim with size 3
    auto bs = (self_bdim.has_value()) ? self_.size(0) : (other_bdim.has_value()) ? other_.size(0) : -1;
    if (bs == 3) {
      auto t = (self_bdim.has_value()) ? self_ : other_;
      int64_t idx = 1;
      for (auto it = t.sizes().begin() + 1; it < t.sizes().end(); ++it, ++idx) {
        if (*it == 3) {
          new_dim = idx;
          break;
        }
      }
    }
  }
  optional<int64_t> out_dim = (self_bdim.has_value() || other_bdim.has_value()) ? 0 : (optional<int64_t>) nullopt;
  return std::make_tuple(at::cross(self_, other_, new_dim), out_dim);
}

// TODO: replace with targetable functionalization
Tensor one_hot_decomposition_hack(const Tensor &self, int64_t num_classes) {
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor.");
    auto shape = self.sizes().vec();

    // empty tensor could be converted to one hot representation,
    // but shape inference is not possible.
    if (self.numel() == 0) {
        if (num_classes <= 0) {
            AT_ERROR("Can not infer total number of classes from empty tensor.");
        } else {
            shape.push_back(num_classes);
            return at::empty(shape, self.options());
        }
    }

    TORCH_CHECK(num_classes > 0, "When vmap-ing torch.nn.functional.one_hot, please "
        "provide an explicit positive num_classes argument.");

    // Disabling all of the following checks. This is OK because scatter has checks too.
    // Maybe one_hot should be a primitive wrt autograd so we don't have to deal with this.
    // // non-empty tensor
    // if (self.device().type() != at::kCUDA) {
    //   //for cuda, rely on device assert thrown by scatter
    //   TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    // }
    // if (self.device().type() != at::kCUDA) {
    //   //rely on device asserts from scatter to avoid sync here
    //   TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
    // }

    shape.push_back(num_classes);
    Tensor ret = at::zeros(shape, self.options());
    return ret.scatter(-1, self.unsqueeze(-1), 1);
}

template <typename A, A a, typename C>
struct UpsampleBackwardBatchRuleHelper;

template <typename F, F Func, typename A, typename B, typename C, typename... T>
struct UpsampleBackwardBatchRuleHelper<F, Func, typelist<A, B, C, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& grad_output, optional<int64_t> grad_output_bdim,
      optional<IntArrayRef> output_size, IntArrayRef input_size,
      T... extra_args) {
    auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
    TORCH_INTERNAL_ASSERT(input_size.size() > 0);

    // input_size is wrong so we correct it
    DimVector physical_input_size(input_size.begin(), input_size.end());
    physical_input_size[0] = grad_output_.sizes()[0];

    auto out = Func(
        grad_output_,
        output_size,
        physical_input_size,
        std::forward<T>(extra_args)...);
    return std::make_tuple(reshape_dim_outof(0, grad_output.sizes()[*grad_output_bdim], out), 0);
  }

};

template <typename A, A a, typename C>
struct GridSampleBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename... T>
struct GridSampleBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& input, optional<int64_t> input_batch_dim,
      const Tensor& grid, optional<int64_t> grid_batch_dim,
      T... extra_args) {
    return grid_sample_batch_rule<F, Func, T...>(
        input, input_batch_dim, grid, grid_batch_dim, std::forward<T>(extra_args)...);
  }
};

template <typename A, A a, typename C>
struct GridSampleBackwardBatchRuleHelper;

template <typename F, F Func, typename T1, typename T2, typename T3, typename... T>
struct GridSampleBackwardBatchRuleHelper<F, Func, typelist<T1, T2, T3, T...>> {
  static std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> apply(
      const Tensor& grad_output, optional<int64_t> grad_output_batch_dim,
      const Tensor& input, optional<int64_t> input_batch_dim,
      const Tensor& grid, optional<int64_t> grid_batch_dim,
      T... extra_args) {
    return grid_sample_backward_batch_rule<F, Func, T...>(
        grad_output, grad_output_batch_dim,
        input, input_batch_dim,
        grid, grid_batch_dim,
        std::forward<T>(extra_args)...);
  }
};

template <typename F, F Func>
struct CudnnGridSampleBackwardBatchRuleHelper {
  static std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> apply(
      const Tensor& input, optional<int64_t> input_batch_dim,
      const Tensor& grid, optional<int64_t> grid_batch_dim,
      const Tensor& grad_output, optional<int64_t> grad_output_batch_dim) {
    return cudnn_grid_sample_backward_batch_rule<F, Func>(
        input, input_batch_dim,
        grid, grid_batch_dim,
        grad_output, grad_output_batch_dim
    );
  }
};

#define GRID_SAMPLE_BATCH_RULE(fn) SINGLE_ARG(\
    GridSampleBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn),\
      c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply)

#define GRID_SAMPLE_BW_BATCH_RULE(fn) SINGLE_ARG(\
    GridSampleBackwardBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn),\
      c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply)

#define CUDNN_GRID_SAMPLE_BW_BATCH_RULE(fn)\
    CudnnGridSampleBackwardBatchRuleHelper<decltype(&ATEN_FN(fn)), &ATEN_FN(fn)>::apply

#define UPSAMPLE_BACKWARD(op, overload) VMAP_SUPPORT(#op"."#overload, SINGLE_ARG(\
    UpsampleBackwardBatchRuleHelper<\
      decltype(&ATEN_FN2(op, overload)),\
      &ATEN_FN2(op, overload),\
      c10::guts::function_traits<decltype(ATEN_FN2(op, overload))>::parameter_types>::apply))

#define UPSAMPLE_BATCH(op) \
  EXISTING_BDIM2(op, vec); \
  EXISTING_BDIM(op);


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("convolution", convolution_batch_rule);
  m.impl("_convolution", _convolution_decomp);
  m.impl("mkldnn_convolution", mkldnn_convolution_decomp);
  m.impl("cudnn_convolution_backward", cudnn_convolution_backward_plumbing);
  m.impl("cudnn_convolution", cudnn_convolution_plumbing);

  EXISTING_BDIM(im2col);
  EXISTING_BDIM(im2col_backward);

  VMAP_SUPPORT("embedding", embedding_batch_rule);

  VMAP_SUPPORT("grid_sampler_2d", GRID_SAMPLE_BATCH_RULE(grid_sampler));
  VMAP_SUPPORT("grid_sampler_2d_backward", GRID_SAMPLE_BW_BATCH_RULE(grid_sampler_2d_backward));

  VMAP_SUPPORT("grid_sampler_3d", GRID_SAMPLE_BATCH_RULE(grid_sampler));
  VMAP_SUPPORT("grid_sampler_3d_backward", GRID_SAMPLE_BW_BATCH_RULE(grid_sampler_3d_backward));
  VMAP_SUPPORT("cudnn_grid_sampler_backward", CUDNN_GRID_SAMPLE_BW_BATCH_RULE(cudnn_grid_sampler_backward));

  VMAP_SUPPORT("cudnn_grid_sampler", GRID_SAMPLE_BATCH_RULE(cudnn_grid_sampler));
  VMAP_SUPPORT("cross", cross_batch_rule);

  EXISTING_BDIM(pixel_shuffle);
  EXISTING_BDIM(pixel_unshuffle);

  VARIADIC_BDIMS(constant_pad_nd);
  EXISTING_BDIM(reflection_pad1d);
  EXISTING_BDIM(reflection_pad2d);
  EXISTING_BDIM(reflection_pad3d);
  EXISTING_BDIM(replication_pad1d);
  EXISTING_BDIM(replication_pad2d);
  EXISTING_BDIM(replication_pad3d);

  EXISTING_BDIM_ALL_BOXED(replication_pad1d_backward);
  EXISTING_BDIM_ALL_BOXED(replication_pad2d_backward);
  EXISTING_BDIM_ALL_BOXED(replication_pad3d_backward);

  EXISTING_BDIM_ALL_BOXED(reflection_pad1d_backward);
  EXISTING_BDIM_ALL_BOXED(reflection_pad2d_backward);
  EXISTING_BDIM_ALL_BOXED(reflection_pad3d_backward);

  UPSAMPLE_BATCH(upsample_bicubic2d);
  UPSAMPLE_BATCH(upsample_bilinear2d);
  UPSAMPLE_BATCH(upsample_linear1d);
  UPSAMPLE_BATCH(upsample_nearest1d);
  UPSAMPLE_BATCH(upsample_nearest2d);
  UPSAMPLE_BATCH(upsample_nearest3d);
  UPSAMPLE_BATCH(upsample_trilinear3d);

  UPSAMPLE_BACKWARD(upsample_bicubic2d_backward, vec);
  UPSAMPLE_BACKWARD(upsample_bilinear2d_backward, vec);
  UPSAMPLE_BACKWARD(upsample_linear1d_backward, vec);
  UPSAMPLE_BACKWARD(upsample_nearest1d_backward, vec);
  UPSAMPLE_BACKWARD(upsample_nearest2d_backward, vec);
  UPSAMPLE_BACKWARD(upsample_nearest3d_backward, vec);
  UPSAMPLE_BACKWARD(upsample_trilinear3d_backward, vec);
  m.impl("one_hot", one_hot_decomposition_hack);
}
}}
