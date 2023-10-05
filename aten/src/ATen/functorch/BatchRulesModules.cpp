// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <utility>

namespace at { namespace functorch {

static Tensor getStepTensor(const Tensor& indices, const c10::SymInt& bdim_size, const c10::SymInt& num_embeddings) {
  // [batch_size, 1, 1, 1, ..., 1]
  c10::SymDimVector view_shape(indices.dim(), 1);
  view_shape[0] = bdim_size;
  auto range = at::arange(0, bdim_size * num_embeddings, num_embeddings, indices.options());
  return range.view_symint(view_shape);
}

static std::tuple<Tensor,optional<int64_t>> embedding_batch_rule(
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& indices, optional<int64_t> indices_bdim,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (!weight_bdim && indices_bdim) {
    // B*, ED -> B*D
    auto result = at::embedding_symint(weight, indices, std::move(padding_idx), scale_grad_by_freq, sparse);
    return std::make_tuple(std::move(result), indices_bdim);
  } else if (weight_bdim && !indices_bdim) {
    // *, BED -> *, E(BD) -> *(BD) -> *BD
    const auto batch_size = weight.size(*weight_bdim);
    const auto weight_ = reshape_dim_into(*weight_bdim, /*embedding_dim*/1, weight);
    auto result = at::embedding_symint(weight_, indices, std::move(padding_idx), scale_grad_by_freq, sparse);
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

  const auto range = getStepTensor(indices, batch_size, num_embeddings);
  indices_ = indices_ + range;
  auto result = at::embedding_symint(weight_, indices_, std::move(padding_idx), scale_grad_by_freq, sparse);
  return std::make_tuple(std::move(result), 0);
}

static std::tuple<Tensor,optional<int64_t>>
embedding_dense_backward_batch_rule(
    const Tensor& grad_, optional<int64_t> grad_bdim,
    const Tensor& indices_, optional<int64_t> indices_bdim,
    c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq) {
  Tensor grad = grad_;
  Tensor indices = indices_;
  if (!indices_bdim && grad_bdim) {
    const auto bdim_size = grad.sym_size(*grad_bdim);
    grad = reshape_dim_into(*grad_bdim, -1, grad);
    auto result = at::embedding_dense_backward_symint(
        grad, indices, std::move(num_weights), std::move(padding_idx), scale_grad_by_freq);
    result = reshape_dim_outof_symint(1, bdim_size, result);
    return std::make_tuple(std::move(result), 1);
  }
  const auto bdim_size = indices.size(*indices_bdim);
  indices = moveBatchDimToFront(indices, indices_bdim);
  grad = moveBatchDimToFront(grad, grad_bdim);
  grad = ensure_has_bdim(grad, grad_bdim.has_value(), bdim_size);
  const auto range = getStepTensor(indices, bdim_size, num_weights);
  auto result = at::embedding_dense_backward_symint(
      grad, indices + range, num_weights * bdim_size, -1, scale_grad_by_freq);
  result = reshape_dim_outof(0, bdim_size, result);
  // Fill in the padding. We can't do it in the embedding_dense_backward call
  // because we need to fill in multiple rows!
  if (padding_idx >= 0) {
    result.select_symint(1, std::move(padding_idx)).fill_(0);
  }
  return std::make_tuple(std::move(result), 0);
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
    result = std::make_tuple(std::move(out), 1);
  } else if (!input_bdim && grid_bdim) {
    // grid of N(BH)W2 -> NC(BH)W or grid of N(BD)HBW3 -> NC(BD)HW
    auto new_grid = reshape_dim_into(*grid_bdim, 1, grid);
    auto out = Func(input, new_grid, std::forward<ExtraArgs>(extra_args)...);
    out = reshape_dim_outof(2, grid.sizes()[*grid_bdim], out);
    result = std::make_tuple(std::move(out), 2);
  } else if (input_bdim && grid_bdim) {
    auto new_input = reshape_dim_into(*input_bdim, 0, input);
    auto new_grid = reshape_dim_into(*grid_bdim, 0, grid);
    auto out = Func(new_input, new_grid, std::forward<ExtraArgs>(extra_args)...);
    out = reshape_dim_outof(0, input.sizes()[*grid_bdim], out);
    result = std::make_tuple(std::move(out), 0);
  } else {
    result = std::make_tuple(Func(input, grid, std::forward<ExtraArgs>(extra_args)...), nullopt);
  }
  return result;
}

static std::tuple<Tensor, Tensor, Tensor, int64_t>
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

  return std::make_tuple(std::move(grad_output_), std::move(input_), std::move(grid_), batch_size);
}

static std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
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

// TODO: replace with targetable functionalization
static Tensor one_hot_decomposition_hack(const Tensor &self, int64_t num_classes) {
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
      c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size,
      T... extra_args) {
    auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
    TORCH_INTERNAL_ASSERT(!input_size.empty());

    // input_size is wrong so we correct it
    c10::SymDimVector physical_input_size(input_size.begin(), input_size.end());
    physical_input_size[0] = grad_output_.sym_sizes()[0];

    auto out = Func(
        grad_output_,
        output_size,
        physical_input_size,
        std::forward<T>(extra_args)...);
    return std::make_tuple(reshape_dim_outof_symint(0, grad_output.sym_sizes()[*grad_output_bdim], out), 0);
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

#define UPSAMPLE_BACKWARD(op) VMAP_SUPPORT(op, SINGLE_ARG(\
    UpsampleBackwardBatchRuleHelper<\
      decltype(&ATEN_FN(op)),\
      &ATEN_FN(op),\
      c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

#define UPSAMPLE_BATCH(op) \
  EXISTING_BDIM2(op, vec); \
  EXISTING_BDIM(op);


TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  EXISTING_BDIM(im2col);
  EXISTING_BDIM(col2im);

  VMAP_SUPPORT(embedding, embedding_batch_rule);
  VMAP_SUPPORT(embedding_dense_backward, embedding_dense_backward_batch_rule);

  VMAP_SUPPORT(grid_sampler_2d, GRID_SAMPLE_BATCH_RULE(grid_sampler));
  VMAP_SUPPORT(grid_sampler_2d_backward, GRID_SAMPLE_BW_BATCH_RULE(grid_sampler_2d_backward));

  VMAP_SUPPORT(grid_sampler_3d, GRID_SAMPLE_BATCH_RULE(grid_sampler));
  VMAP_SUPPORT(grid_sampler_3d_backward, GRID_SAMPLE_BW_BATCH_RULE(grid_sampler_3d_backward));
  VMAP_SUPPORT(cudnn_grid_sampler_backward, CUDNN_GRID_SAMPLE_BW_BATCH_RULE(cudnn_grid_sampler_backward));

  VMAP_SUPPORT(cudnn_grid_sampler, GRID_SAMPLE_BATCH_RULE(cudnn_grid_sampler));

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
  UPSAMPLE_BATCH(_upsample_bilinear2d_aa);
  UPSAMPLE_BATCH(_upsample_bicubic2d_aa);

  UPSAMPLE_BACKWARD(upsample_bicubic2d_backward);
  UPSAMPLE_BACKWARD(upsample_bilinear2d_backward);
  UPSAMPLE_BACKWARD(upsample_linear1d_backward);
  UPSAMPLE_BACKWARD(upsample_nearest1d_backward);
  UPSAMPLE_BACKWARD(upsample_nearest2d_backward);
  UPSAMPLE_BACKWARD(upsample_nearest3d_backward);
  UPSAMPLE_BACKWARD(upsample_trilinear3d_backward);
  UPSAMPLE_BACKWARD(_upsample_bilinear2d_aa_backward);
  UPSAMPLE_BACKWARD(_upsample_bicubic2d_aa_backward);

  m.impl("one_hot", one_hot_decomposition_hack);
}
}}
