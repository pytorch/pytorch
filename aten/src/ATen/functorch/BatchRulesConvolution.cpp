// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at::functorch {

// convolution_batch_rule translated from jax with modifications:
// https://github.com/google/jax/blob/master/jax/_src/lax/lax.py#L3143

// PyTorch's convolution is different from JAX's conv_general_dilated:
// we do not support batch_group_count (which is needed for convolution backwards).
// Instead, there's a convolution_backward op that needs a batching rule.
static std::tuple<Tensor, std::optional<int64_t>>
convolution_batch_rule(const Tensor& lhs, std::optional<int64_t> lhs_bdim, const Tensor& rhs, std::optional<int64_t> rhs_bdim, const std::optional<Tensor>& bias, std::optional<int64_t> bias_bdim, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  DimVector lhs_spec(stride.size() + 2);
  std::iota(lhs_spec.begin(), lhs_spec.end(), 0);
  DimVector rhs_spec = lhs_spec;
  DimVector out_spec = lhs_spec;
  if (transposed) {
    rhs_spec[0] = 1;
    rhs_spec[1] = 0;
  }

  // If we have a batched bias or weight, we need to perform the computation separately.
  std::optional<Tensor> unbatched_bias;
  bool separate_bias = false;
  if ((rhs_bdim && bias && bias->defined()) || bias_bdim) {
    TORCH_INTERNAL_ASSERT(bias.has_value());
    TORCH_INTERNAL_ASSERT(bias->defined());
    unbatched_bias = std::nullopt;
    separate_bias = true;
  } else {
    unbatched_bias = bias;
    separate_bias = false;
  }
  std::tuple<Tensor, std::optional<int64_t>> result;
  if (lhs_bdim && !rhs_bdim) {
    auto new_x = reshape_dim_into(*lhs_bdim, lhs_spec[0], lhs);
    auto out = at::convolution_symint(new_x, rhs, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
    out = reshape_dim_outof_symint(out_spec[0], lhs.sizes()[*lhs_bdim], out);
    result = std::make_tuple(out, out_spec[0]);
  } else if (!lhs_bdim && rhs_bdim) {
    if (groups == 1) {
      auto new_w = reshape_dim_into(*rhs_bdim, rhs_spec[0], rhs);
      auto out = at::convolution_symint(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
      out = reshape_dim_outof_symint(out_spec[1], rhs.size(*rhs_bdim), out);
      result = std::make_tuple(out, out_spec[1]);
    } else {
      if (transposed) {
        // conv_transpose with groups is normally NIHW, IOHW -> N(GO)HW
        // With RHS batched, we do the following:
        // NIHW, BIOHW -> NIHW, I(BO)HW -> N(GBO)HW -> BN(GO)HW
        // NB: the following isn't written using rhs_spec
        // (PyTorch convs have a fixed dimension order)

        // BIOHW -> I(BO)HW
        auto new_w = reshape_dim_into(*rhs_bdim, 1, rhs);
        // NIHW, I(BO)HW -> N(GBO)HW
        auto out = at::convolution_symint(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
        // N(GBO)HW -> NG(BO)HW
        out = reshape_dim_outof_symint(1, groups, out);
        // NG(BO)HW -> NGBOHW
        out = reshape_dim_outof_symint(2, rhs.size(*rhs_bdim), out);
        // NGBOHW -> NB(GO)HW
        out = reshape_dim_into(1, 2, out);
        result = std::make_tuple(out, 1);
      } else {
        // conv with groups is normally N(GI)HW, (GO)IHW -> N(GO)HW
        // With RHS batched, we do the following:
        // N(GI)HW, B(GO)IHW -> N(GI)HW, (GBO)IHW -> N(GBO)HW -> BN(GO)HW
        // NB: the following isn't written using rhs_spec
        // (PyTorch convs have a fixed dimension order)

        // B(GO)IHW -> BGOIHW
        auto new_w = reshape_dim_outof_symint(0 + (*rhs_bdim == 0), groups, rhs);
        // BGOIHW -> G(BO)IHW
        new_w = reshape_dim_into(*rhs_bdim + (*rhs_bdim > 0), 1, new_w);
        // G(BO)IHW -> (GBO)IHW
        new_w = reshape_dim_into(0, 0, new_w);
        // N(GI)HW, (GBO)IHW -> N(GBO)HW
        auto out = at::convolution_symint(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
        // N(GBO)HW -> NG(BO)HW
        out = reshape_dim_outof_symint(1, groups, out);
        // NG(BO)HW -> NGBOHW
        out = reshape_dim_outof_symint(2, rhs.size(*rhs_bdim), out);
        // NGBOHW -> NB(GO)HW
        out = reshape_dim_into(1, 2, out);
        result = std::make_tuple(out, 1);
      }
    }
  } else if (lhs_bdim && rhs_bdim) {
    auto new_x = reshape_dim_into(*lhs_bdim, lhs_spec[1], lhs);
    groups *= lhs.sizes()[*lhs_bdim];
    auto dim_with_groups = transposed ? 1 : 0;
    auto new_w = reshape_dim_into(*rhs_bdim, rhs_spec[dim_with_groups], rhs);
    auto out = at::convolution_symint(new_x, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
    out = reshape_dim_outof_symint(out_spec[1], lhs.sizes()[*lhs_bdim], out);
    result = std::make_tuple(out, out_spec[1]);
  } else {
    result = std::make_tuple(at::convolution_symint(lhs, rhs, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups), std::nullopt);
  }
  if (separate_bias) {
    auto& [A, A_batch_dim] = result;
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

static Tensor _convolution_decomp(
    const Tensor& input_r, const Tensor& weight_r, const std::optional<Tensor>& bias_r_opt,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  // Ignore everything. If the user called this in the normal way,
  // then they should be fine.
  (void) benchmark;
  (void) deterministic;
  (void) cudnn_enabled;
  (void) allow_tf32;
  return at::convolution(
      input_r, weight_r, bias_r_opt, stride_, padding_, dilation_, transposed_, output_padding_, groups_);
}

static Tensor compute_grad_bias(
    const Tensor& grad_output_, std::array<bool, 3> output_mask) {
  if (!output_mask[2]) {
    return Tensor();
  }
  DimVector reduce_dims;
  reduce_dims.resize(grad_output_.dim() - 1);
  reduce_dims[0] = 0;
  std::iota(reduce_dims.begin() + 1, reduce_dims.end(), 2);
  return grad_output_.sum(reduce_dims);
}

// reshapes the batch_size into dim
static Tensor make_dummy(
    const Tensor& tensor, std::optional<int64_t> tensor_bdim,
    int64_t dim, int64_t batch_size) {
  auto tensor_ = tensor_bdim ? tensor.select(*tensor_bdim, 0) : tensor;
  auto orig_size = tensor_.size(dim);
  tensor_ = tensor_.slice(dim, 0, 1);

  DimVector expand_shape(tensor_.sizes().begin(), tensor_.sizes().end());
  expand_shape[dim] = batch_size * orig_size;

  return tensor_.new_empty({}).expand(expand_shape);
}

static std::tuple<Tensor, std::optional<int64_t>>
convolution_backward_input_batch_rule(
    const Tensor& grad_output, std::optional<int64_t> grad_output_bdim,
    const Tensor& input, std::optional<int64_t> input_bdim,
    const Tensor& weight, std::optional<int64_t> weight_bdim,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, const c10::SymInt& groups) {
  const std::array<bool, 3> mask = {true, false, false};
  if (grad_output_bdim && weight_bdim) {
    // regular: BNO, BOI -> N(BO), (BO)I -> N(BI)
    // transposed: BNO, BIO -> N(BO), (BI)O -> N(BI)
    const auto batch_size = weight.size(*weight_bdim);
    const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 1, grad_output);
    const auto weight_ = reshape_dim_into(*weight_bdim, 0, weight);
    auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
    const auto result = at::convolution_backward_symint(
        grad_output_, dummy_input, weight_, std::nullopt, stride, padding,
        dilation, transposed, output_padding, groups * batch_size, mask);
    auto grad_input = reshape_dim_outof(1, batch_size, std::get<0>(result));
    return std::make_tuple(std::move(grad_input), 1);
  } else if (grad_output_bdim && !weight_bdim) {
    // BNO, OI -> (BN)O, OI -> (BN)I
    // transposed is the same.
    const auto batch_size = grad_output.size(*grad_output_bdim);
    const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
    auto dummy_input = make_dummy(input, input_bdim, 0, batch_size);
    const auto result = at::convolution_backward_symint(
        grad_output_, dummy_input, weight, std::nullopt, stride, padding,
        dilation, transposed, output_padding, groups, mask);
    auto grad_input = reshape_dim_outof(0, batch_size, std::get<0>(result));
    return std::make_tuple(std::move(grad_input), 0);
  } else if (!grad_output_bdim && weight_bdim) {
    const auto batch_size = weight.size(*weight_bdim);
    if (groups == 1) {
      // regular: NO, BOI -> NO, O(BI) -> N(BI)
      // transposed: NO, BIO -> NO, (BI)O -> N(BI)
      const auto in_ch_dim = transposed ? 0 : 1;
      const auto weight_ = reshape_dim_into(*weight_bdim, in_ch_dim, weight);
      auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
      const auto result = at::convolution_backward_symint(
          grad_output, dummy_input, weight_, std::nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      auto grad_input = reshape_dim_outof(1, batch_size, std::get<0>(result));
      return std::make_tuple(std::move(grad_input), 1);
    }
    Tensor grad_input;
    if (!transposed) {
      // N(GO), B(GO)I -> N(GO), (GO)(BI) -> N(GBI)
      const auto weight_ = reshape_dim_into(*weight_bdim, 1, weight);
      auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
      grad_input = std::get<0>(at::convolution_backward_symint(
          grad_output, dummy_input, weight_, std::nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask)); // N(GBI)
    } else {
      // N(GO), B(GI)O -> N(GO), (GBI)O -> N(GBI)
      auto weight_ = moveBatchDimToFront(weight, weight_bdim); // B(GI)O
      weight_ = reshape_dim_outof_symint(1, groups, weight_);         // BGIO
      weight_ = weight_.transpose(0, 1);                       // GBIO
      weight_ = weight_.flatten(0, 2);                         // (GBI)O
      const auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
      grad_input = std::get<0>(at::convolution_backward_symint(
          grad_output, dummy_input, weight_, std::nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask)); // N(GBI)
    }
    // N(GBI) -> NG(BI) -> NGBI -> NBGI -> NB(GI)
    grad_input = reshape_dim_outof_symint(1, groups, grad_input);
    grad_input = reshape_dim_outof_symint(2, batch_size, grad_input);
    grad_input = grad_input.transpose(1, 2);
    grad_input = reshape_dim_into(2, 2, grad_input);
    return std::make_tuple(std::move(grad_input), 1);
  } else {
    TORCH_INTERNAL_ASSERT(input_bdim);
    const auto dummy_input = make_dummy(input, input_bdim, 0, 1);
    auto result = at::convolution_backward_symint(
        grad_output, dummy_input, weight, std::nullopt, stride, padding,
        dilation, transposed, output_padding, groups, mask);
    return std::make_tuple(std::move(std::get<0>(result)), std::nullopt);
  }
}
static std::tuple<Tensor, std::optional<int64_t>>
convolution_backward_weight_batch_rule(
    const Tensor& grad_output, std::optional<int64_t> grad_output_bdim,
    const Tensor& input, std::optional<int64_t> input_bdim,
    const Tensor& weight, std::optional<int64_t> weight_bdim,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, const c10::SymInt& groups) {
  const std::array<bool, 3> mask = {false, true, false};
  if (grad_output_bdim && input_bdim) {
    // BNO, BNI -> N(BO), N(BI) -> (BO)I (regular) (BI)O (transposed)
    const auto batch_size = input.size(*input_bdim);
    const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 1, grad_output);
    const auto input_ = reshape_dim_into(*input_bdim, 1, input);
    const auto dummy_weight = make_dummy(weight, weight_bdim, 0, batch_size);
    auto result = at::convolution_backward_symint(
        grad_output_, input_, dummy_weight, std::nullopt, stride, padding,
        dilation, transposed, output_padding, groups * batch_size, mask);
    auto& grad_weight = std::get<1>(result);
    grad_weight = reshape_dim_outof_symint(0, batch_size, grad_weight);
    return std::make_tuple(std::move(grad_weight), 0);
  } else if (grad_output_bdim && !input_bdim) {
    const auto batch_size = grad_output.size(*grad_output_bdim);
    if (groups == 1) {
      // regular: BNO, NI -> N(BO), NI -> (BO)I
      // transposed: BNO, NI -> N(BO), NI -> I(BO)
      const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 1, grad_output);
      const auto out_ch_dim = transposed ? 1 : 0;
      const auto dummy_weight = make_dummy(weight, weight_bdim, out_ch_dim, batch_size);
      auto result = at::convolution_backward_symint(
          grad_output_, input, dummy_weight, std::nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      auto& grad_weight = std::get<1>(result);
      grad_weight = reshape_dim_outof_symint(out_ch_dim, batch_size, grad_weight);
      return std::make_tuple(std::move(grad_weight), out_ch_dim);
    } else {
      auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim); // BN(GO)
      grad_output_ = reshape_dim_outof_symint(2, groups, grad_output_);              // BNGO
      grad_output_ = grad_output_.movedim(0, 2);                              // NGBO
      grad_output_ = grad_output_.flatten(1, 3);                              // N(GBO)
      if (!transposed) {
        // BN(GO), N(GI) -> N(GBO), N(GI) -> (GBO)I
        const auto dummy_weight = make_dummy(weight, weight_bdim, 0, batch_size);
        auto result = at::convolution_backward_symint(
            grad_output_, input, dummy_weight, std::nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask);
        auto& grad_weight = std::get<1>(result);
        grad_weight = grad_weight.unflatten_symint(0, { groups, batch_size, -1 }); // GBOI
        grad_weight = grad_weight.transpose(0, 1);                          // BGOI
        grad_weight = grad_weight.flatten(1, 2);                            // B(GO)I
        return std::make_tuple(std::move(grad_weight), 0);
      } else {
        // BN(GO), N(GI) -> N(GBO), N(GI) -> (GI)(BO)
        const auto dummy_weight = make_dummy(weight, weight_bdim, 1, batch_size);
        auto result = at::convolution_backward_symint(
            grad_output_, input, dummy_weight, std::nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask);
        auto& grad_weight = std::get<1>(result);
        grad_weight = reshape_dim_outof_symint(1, batch_size, grad_weight);
        return std::make_tuple(std::move(grad_weight), 1);
      }
    }
  } else if (!grad_output_bdim && input_bdim) {
    const auto batch_size = input.size(*input_bdim);
    if (groups == 1) {
      // regular: NO, BNI -> NO, N(BI) -> O(BI)
      // transposed: NO, BNI -> NO, N(BI) -> (BI)O
      const auto input_ = reshape_dim_into(*input_bdim, 1, input);
      const auto in_ch_dim = transposed ? 0 : 1;
      const auto dummy_weight = make_dummy(weight, weight_bdim, in_ch_dim, batch_size);
      auto result = at::convolution_backward_symint(
          grad_output, input_, dummy_weight, std::nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      auto& grad_weight = std::get<1>(result);
      grad_weight = reshape_dim_outof_symint(in_ch_dim, batch_size, grad_weight);
      return std::make_tuple(std::move(grad_weight), in_ch_dim);
    } else {
      auto input_ = moveBatchDimToFront(input, input_bdim); // BN(GI)
      input_ = reshape_dim_outof_symint(2, groups, input_);        // BNGI
      input_ = input_.movedim(0, 2);                        // NGBI
      input_ = input_.flatten(1, 3);                        // N(GBI)
      if (!transposed) {
        // regular: N(GO), BN(GI) -> N(GO), N(GBI) -> (GO)(BI)
        const auto dummy_weight = make_dummy(weight, weight_bdim, 1, batch_size);
        auto result = at::convolution_backward_symint(
            grad_output, input_, dummy_weight, std::nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask);
        auto& grad_weight = std::get<1>(result);
        grad_weight = reshape_dim_outof_symint(1, batch_size, grad_weight);
        return std::make_tuple(grad_weight, 1);
      } else {
        // transposed: N(GO), BN(GI) -> N(GO), N(GBI) -> (GBI)O
        const auto dummy_weight = make_dummy(weight, weight_bdim, 0, batch_size);
        auto result = at::convolution_backward_symint(
            grad_output, input_, dummy_weight, std::nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask);
        auto& grad_weight = std::get<1>(result);
        grad_weight = grad_weight.unflatten_symint(0, { groups, batch_size, -1 }); // GBIO
        grad_weight = grad_weight.transpose(0, 1);                          // BGIO
        grad_weight = grad_weight.flatten(1, 2);                            // B(GI)O
        return std::make_tuple(std::move(grad_weight), 0);
      }
    }
  } else {
    TORCH_INTERNAL_ASSERT(weight_bdim);
    const auto dummy_weight = make_dummy(weight, weight_bdim, 0, 1);
    auto result = at::convolution_backward_symint(
        grad_output, input, dummy_weight, std::nullopt, stride, padding,
        dilation, transposed, output_padding, groups, mask);
    return std::make_tuple(std::move(std::get<1>(result)), std::nullopt);

  }
}

static std::tuple<Tensor,Tensor,Tensor> convolution_backward_plumbing(
    const Tensor& grad_output_, const Tensor& input_, const Tensor& weight_,
    const c10::OptionalArrayRef<SymInt> bias_sizes_opt,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups, std::array<bool, 3> output_mask) {
  const auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "convolution_backward_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  int64_t cur_level = maybe_layer->layerId();

  if (!areAnyBatchedAtLevel({grad_output_, input_, weight_}, cur_level)){
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::convolution_backward_symint(
        grad_output_, input_, weight_, bias_sizes_opt, stride, padding,
        dilation, transposed, output_padding, std::move(groups), output_mask);
  }

  auto [grad_output, grad_output_bdim] = unwrapTensorAtLevel(grad_output_, cur_level);
  auto [input, input_bdim] = unwrapTensorAtLevel(input_, cur_level);
  auto [weight, weight_bdim] = unwrapTensorAtLevel(weight_, cur_level);

  auto grad_bias = compute_grad_bias(grad_output_, output_mask);
  output_mask[2] = false;

  // TODO: A little bird says that unfold + matmul is actually faster than
  // group convolution in many cases. We should benchmark some of
  // the common cases and replace things with unfold + matmul as necessary.

  // Notation:
  // B - a batch dimension
  // G - groups (sometimes omitted because it doesn't matter)
  // NO - grad_output
  // NI - input
  // OI - weight
  // "(BO)I" - we don't actually care about the values of this Tensor,
  //           we just need to create a tensor on the same device with the
  //           correct shape and pray that the implementation is smart enough
  //           to not do anything with it.

  // BNO, BNI, BOI
  // AKA one of the model ensembling case
  if (grad_output_bdim && input_bdim && weight_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    grad_output = reshape_dim_into(*grad_output_bdim, 1, grad_output);

    // BNO, BNI, BOI -> N(BO), N(BI), (BO)I
    const auto batch_size = weight.size(*weight_bdim);
    input = reshape_dim_into(*input_bdim, 1, input);
    weight = reshape_dim_into(*weight_bdim, 0, weight);
    const auto result = at::convolution_backward_symint(
        grad_output, input, weight, std::nullopt, stride, padding, dilation,
        transposed, output_padding, batch_size * groups, output_mask);
    // N(BI), (BO)I -> NBI, BOI
    auto grad_input = output_mask[0] ?
      reshape_dim_outof(1, batch_size, std::get<0>(result)) : Tensor();
    auto grad_weight = output_mask[1] ?
      reshape_dim_outof(0, batch_size, std::get<1>(result)) : Tensor();
    return std::make_tuple(
        output_mask[0] ? makeBatched(std::move(grad_input), 1, cur_level) : std::move(grad_input),
        output_mask[1] ? makeBatched(std::move(grad_weight), 0, cur_level) : std::move(grad_weight),
        std::move(grad_bias));
  }

  Tensor grad_input;
  if (output_mask[0]) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto [tensor, bdim] = convolution_backward_input_batch_rule(
        grad_output, grad_output_bdim,
        input, input_bdim,
        weight, weight_bdim,
        stride, padding, dilation, transposed, output_padding, groups);
    grad_input = makeBatched(std::move(tensor), bdim, cur_level);
  }

  Tensor grad_weight;
  if (output_mask[1]) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto [tensor, bdim] = convolution_backward_weight_batch_rule(
        grad_output, grad_output_bdim,
        input, input_bdim,
        weight, weight_bdim,
        stride, padding, dilation, transposed, output_padding, groups);
    grad_weight = makeBatched(std::move(tensor), bdim, cur_level);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));

  // Someone's definitely going to find a problem with this batching rule so
  // I'm leaving the following fallback if we need it back.
  // static auto op = c10::Dispatcher::singleton()
  //   .findSchemaOrThrow("aten::convolution_backward", "");
  // auto result = slow_fallback<Tensor,Tensor,Tensor>(op, {
  //   grad_output_, input_, weight_, bias_sizes_opt,
  //   stride, padding, dilation, transposed, output_padding, groups, output_mask
  // });
  // return std::make_tuple(grad_input, std::get<1>(result), grad_bias);
}


TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(convolution, convolution_batch_rule);
  m.impl("_convolution", _convolution_decomp);
  m.impl("convolution_backward", convolution_backward_plumbing);
}

} // namespace at;:functorch
