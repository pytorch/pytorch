#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>
#include <limits>
#include <tuple>
#include <vector>
#include <optional>
#include <algorithm>
#include <string>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/max.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/where.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/div.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/linear_cross_entropy_backward_native.h>
#include <ATen/ops/linear_cross_entropy_native.h>
#endif

namespace at::native {

// Strategy selection for optimal chunking approach
enum class ChunkingStrategy {
    NAIVE,           // No chunking - standard approach
    VOCAB_CHUNKING,  // Chunk vocabulary dimension (existing)
    BATCH_CHUNKING   // Chunk batch dimension (new)
};

constexpr int64_t kDefaultVocabChunkSize = 4096;
constexpr int64_t kDefaultBatchChunkSize = 1024;

struct LinearCrossEntropySavedForBackward {
  Tensor logsumexp;        // [N]
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  ChunkingStrategy strategy{ChunkingStrategy::NAIVE};
  int64_t chunk_size{0};
};

struct LinearCrossEntropyForwardResult {
  Tensor loss;
  std::optional<LinearCrossEntropySavedForBackward> saved;
};

inline ChunkingStrategy select_chunking_strategy(c10::string_view strategy) {
  if (strategy == "none") {
    return ChunkingStrategy::NAIVE;
  } else if (strategy == "vocab") {
    return ChunkingStrategy::VOCAB_CHUNKING;
  } else if (strategy == "batch") {
    return ChunkingStrategy::BATCH_CHUNKING;
  }
  TORCH_CHECK(false,
              "Unknown chunking strategy: ",
              strategy,
              ". Valid options: 'vocab', 'batch', 'none'");
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_naive(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    bool save_for_backward);

LinearCrossEntropyForwardResult linear_cross_entropy_forward_vocab_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward);

LinearCrossEntropyForwardResult linear_cross_entropy_forward_batch_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward);

std::tuple<Tensor, Tensor> _linear_cross_entropy_vocab_chunking_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size);

std::tuple<Tensor, Tensor, std::optional<Tensor>> _linear_cross_entropy_vocab_chunking_backward_cpu(
    const Tensor& grad_output,
    const Tensor& saved_logsumexp,
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> _linear_cross_entropy_batch_chunking_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size);

std::tuple<Tensor, Tensor, std::optional<Tensor>> _linear_cross_entropy_batch_chunking_backward_cpu(
    const Tensor& grad_output,
    const Tensor& saved_grad_input,
    const Tensor& saved_grad_weight,
    const Tensor& saved_grad_bias,
    const Tensor& grad_weight_valid,
    const Tensor& grad_bias_valid,
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size);

inline const Tensor& contiguous_if_needed(const Tensor& tensor, Tensor& buffer) {
  if (tensor.is_contiguous()) {
    return tensor;
  }
  buffer = tensor.contiguous();
  return buffer;
}

static LinearCrossEntropyForwardResult linear_cross_entropy_forward_impl(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    ChunkingStrategy strategy,
    int64_t vocab_chunk_size,
    int64_t batch_chunk_size,
    bool save_for_backward) {

  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(linear_weight.dim() == 2, "Expected linear_weight to be 2-dimensional, got ", linear_weight.dim());
  TORCH_CHECK(input.size(-1) == linear_weight.size(1),
              "Expected input.size(-1) to match linear_weight.size(1), got ",
              input.size(-1), " and ", linear_weight.size(1));

  switch (strategy) {
    case ChunkingStrategy::VOCAB_CHUNKING:
      TORCH_CHECK(
          vocab_chunk_size > 0,
          "linear_cross_entropy: vocab_chunk_size must be positive, got ",
          vocab_chunk_size);
      return linear_cross_entropy_forward_vocab_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          reduction,
          ignore_index,
          label_smoothing,
          vocab_chunk_size,
          save_for_backward);
    case ChunkingStrategy::BATCH_CHUNKING:
      TORCH_CHECK(
          batch_chunk_size > 0,
          "linear_cross_entropy: batch_chunk_size must be positive, got ",
          batch_chunk_size);
      return linear_cross_entropy_forward_batch_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          reduction,
          ignore_index,
          label_smoothing,
          batch_chunk_size,
          save_for_backward);
    default:
      return linear_cross_entropy_forward_naive(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          reduction,
          ignore_index,
          label_smoothing,
          save_for_backward);
  }
}

Tensor linear_cross_entropy_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy,
    std::optional<int64_t> vocab_chunk_size_opt,
    std::optional<int64_t> batch_chunk_size_opt) {

  const int64_t vocab_chunk_size = vocab_chunk_size_opt.value_or(kDefaultVocabChunkSize);
  const int64_t batch_chunk_size = batch_chunk_size_opt.value_or(kDefaultBatchChunkSize);
  const ChunkingStrategy strategy = select_chunking_strategy(chunking_strategy);

  auto result = linear_cross_entropy_forward_impl(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      reduction,
      ignore_index,
      label_smoothing,
      strategy,
      vocab_chunk_size,
      batch_chunk_size,
      /*save_for_backward=*/false);

  return std::move(result.loss);
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_naive(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    bool save_for_backward) {

  Tensor input_buffer;
  const Tensor& input_ref = contiguous_if_needed(input, input_buffer);
  Tensor linear_weight_buffer;
  const Tensor& linear_weight_ref = contiguous_if_needed(linear_weight, linear_weight_buffer);
  Tensor target_buffer;
  const Tensor& target_ref = contiguous_if_needed(target, target_buffer);
  Tensor linear_bias_tensor;
  if (linear_bias_opt.has_value()) {
    linear_bias_tensor = linear_bias_opt.value();
  }
  Tensor linear_bias_buffer;
  const Tensor& linear_bias_ref = linear_bias_tensor.defined()
      ? contiguous_if_needed(linear_bias_tensor, linear_bias_buffer)
      : linear_bias_tensor;
  std::optional<Tensor> linear_bias_use;
  if (linear_bias_ref.defined()) {
    linear_bias_use = linear_bias_ref;
  }

  auto logits = at::linear(input_ref, linear_weight_ref, linear_bias_use);
  auto logits_flat = logits.reshape({-1, logits.size(-1)});
  auto target_flat = target_ref.reshape({-1});
  auto options = logits_flat.options();
  const int64_t vocab_size = linear_weight_ref.size(0);

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  Tensor logsumexp = at::logsumexp(logits_flat, {1}, false);
  Tensor target_logits = at::zeros(logsumexp.sizes(), options);

  auto valid_indices = valid_mask.nonzero().reshape({-1});
  if (valid_indices.numel() > 0) {
    auto gathered_targets = at::index_select(target_flat, 0, valid_indices);
    auto gathered_logits = at::index_select(logits_flat, 0, valid_indices);
    auto gathered = at::gather(gathered_logits, 1, gathered_targets.unsqueeze(1)).squeeze(1);
    target_logits.scatter_(0, valid_indices, gathered);
  }

  Tensor losses = logsumexp;
  if (label_smoothing > 0.0) {
    const double smoothing = label_smoothing;
    const double uniform = smoothing / static_cast<double>(vocab_size);
    auto sum_logits = logits_flat.sum(-1);
    losses = losses - target_logits.mul(1.0 - smoothing) - sum_logits.mul(uniform);
  } else {
    losses = losses - target_logits;
  }

  losses.masked_fill_(at::logical_not(valid_mask), 0);

  Tensor loss_result;
  if (reduction == Reduction::None) {
    loss_result = losses.reshape(target.sizes());
  } else {
    auto total_loss = losses.sum();
    if (reduction == Reduction::Sum) {
      loss_result = total_loss;
    } else {
      auto denom_long = valid_mask.sum();
      auto denom = denom_long.to(total_loss.scalar_type());
      if (denom_long.item<int64_t>() == 0) {
        loss_result = denom.div(denom);
      } else {
        loss_result = total_loss.div(denom);
      }
    }
  }

  LinearCrossEntropyForwardResult result;
  result.loss = std::move(loss_result);

  if (save_for_backward) {
    LinearCrossEntropySavedForBackward saved;
    saved.logsumexp = std::move(logsumexp);
    saved.strategy = ChunkingStrategy::NAIVE;
    saved.chunk_size = logits_flat.size(0);
    result.saved = std::move(saved);
  }

  return result;
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_vocab_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward) {

  Tensor input_buffer;
  const Tensor& input_ref = contiguous_if_needed(input, input_buffer);
  Tensor linear_weight_buffer;
  const Tensor& linear_weight_ref = contiguous_if_needed(linear_weight, linear_weight_buffer);
  Tensor target_buffer;
  const Tensor& target_ref = contiguous_if_needed(target, target_buffer);
  Tensor linear_bias_tensor;
  if (linear_bias_opt.has_value()) {
    linear_bias_tensor = linear_bias_opt.value();
  }
  Tensor linear_bias_buffer;
  const Tensor& linear_bias_ref = linear_bias_tensor.defined()
      ? contiguous_if_needed(linear_bias_tensor, linear_bias_buffer)
      : linear_bias_tensor;
  std::optional<Tensor> linear_bias_use;
  if (linear_bias_ref.defined()) {
    linear_bias_use = linear_bias_ref;
  }

  auto input_flat = input_ref.reshape({-1, input_ref.size(-1)});
  auto target_flat = target_ref.reshape({-1});
  const int64_t vocab_size = linear_weight_ref.size(0);
  const auto options = input_flat.options();
  auto long_options = options.dtype(at::kLong);
  const double neg_inf = -std::numeric_limits<double>::infinity();

  Tensor running_max = at::full({input_flat.size(0)}, neg_inf, options);
  Tensor exp_sums = at::zeros({input_flat.size(0)}, options);
  Tensor target_logits = at::zeros({input_flat.size(0)}, options);
  Tensor target_found = at::zeros({input_flat.size(0)}, long_options);
  Tensor sum_logits;
  if (label_smoothing > 0.0) {
    sum_logits = at::zeros({input_flat.size(0)}, options);
  }

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);

    auto weight_chunk = linear_weight_ref.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (linear_bias_ref.defined()) {
      bias_chunk = linear_bias_ref.slice(0, start_idx, end_idx);
    }

    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);

    if (label_smoothing > 0.0) {
      sum_logits = sum_logits + logits_chunk.sum(-1);
    }

    auto chunk_max = std::get<0>(logits_chunk.max(-1));
    auto new_max = at::maximum(running_max, chunk_max);

    auto exp_scale_old = at::exp(running_max.sub(new_max));
    auto shifted_logits = logits_chunk.sub(new_max.unsqueeze(-1));
    auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
    exp_sums = exp_sums.mul(exp_scale_old).add_(exp_chunk);
    running_max = new_max;

    auto lower_bound = at::ge(target_flat, start_idx);
    auto upper_bound = at::lt(target_flat, end_idx);
    auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
    target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);

    auto indices = target_chunk_mask.nonzero().reshape({-1});
    if (indices.numel() > 0) {
      auto selected_targets = at::index_select(target_flat, 0, indices);
      auto local_targets = at::sub(selected_targets, start_idx);
      auto selected_logits = at::index_select(logits_chunk, 0, indices);
    auto gathered = at::gather(selected_logits, 1, local_targets.unsqueeze(1)).squeeze(1);
    target_logits = target_logits.scatter(0, indices, gathered);
      target_found.index_put_({indices}, at::ones(indices.sizes(), long_options));
    }
  }

  auto target_found_mask = target_found.gt(0);
  auto coverage_mask = at::logical_or(target_found_mask, at::logical_not(valid_mask));
  TORCH_CHECK(
      coverage_mask.all().item<bool>(),
      "linear_cross_entropy: target index not found in vocabulary chunks");

  auto logsumexp = running_max.add(exp_sums.log());

  Tensor losses;
  if (label_smoothing > 0.0) {
    const double smoothing = label_smoothing;
    const double uniform = smoothing / static_cast<double>(vocab_size);
    auto main_term = target_logits.mul(1.0 - smoothing);
    auto uniform_term = sum_logits.mul(uniform);
    losses = logsumexp - main_term - uniform_term;
  } else {
    losses = logsumexp - target_logits;
  }

  losses.masked_fill_(at::logical_not(valid_mask), 0);

  Tensor loss_result;
  if (reduction == Reduction::None) {
    loss_result = losses.reshape(target.sizes());
  } else {
    auto total_loss = losses.sum();
    if (reduction == Reduction::Sum) {
      loss_result = total_loss;
    } else {
      auto denom_long = valid_mask.sum();
      auto denom = denom_long.to(total_loss.scalar_type());
      if (denom_long.item<int64_t>() == 0) {
        loss_result = denom.div(denom);
      } else {
        loss_result = total_loss.div(denom);
      }
    }
  }

  LinearCrossEntropyForwardResult result;
  result.loss = std::move(loss_result);

  if (save_for_backward) {
    LinearCrossEntropySavedForBackward saved;
    saved.logsumexp = std::move(logsumexp);
    saved.strategy = ChunkingStrategy::VOCAB_CHUNKING;
    saved.chunk_size = chunk_size;
    result.saved = std::move(saved);
  }

  return result;
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_batch_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward) {

  Tensor input_buffer;
  const Tensor& input_ref = contiguous_if_needed(input, input_buffer);
  Tensor linear_weight_buffer;
  const Tensor& linear_weight_ref = contiguous_if_needed(linear_weight, linear_weight_buffer);
  Tensor target_buffer;
  const Tensor& target_ref = contiguous_if_needed(target, target_buffer);
  Tensor linear_bias_tensor;
  if (linear_bias_opt.has_value()) {
    linear_bias_tensor = linear_bias_opt.value();
  }
  Tensor linear_bias_buffer;
  const Tensor& linear_bias_ref = linear_bias_tensor.defined()
      ? contiguous_if_needed(linear_bias_tensor, linear_bias_buffer)
      : linear_bias_tensor;
  std::optional<Tensor> linear_bias_use;
  if (linear_bias_ref.defined()) {
    linear_bias_use = linear_bias_ref;
  }

  auto input_flat = input_ref.reshape({-1, input_ref.size(-1)});
  auto target_flat = target_ref.reshape({-1});
  const int64_t batch_size = input_flat.size(0);

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  Tensor target_logits = at::zeros({batch_size}, input_ref.options());
  Tensor sum_logits;
  if (label_smoothing > 0.0) {
    sum_logits = at::empty({batch_size}, input_ref.options());
  }

  Tensor losses_buffer;
  if (reduction == Reduction::None) {
    losses_buffer = at::empty({batch_size}, input_ref.options());
  }

  Tensor total_loss = at::zeros({}, input_ref.options());

  Tensor grad_input_saved;
  Tensor grad_weight_saved;
  Tensor grad_bias_saved;
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (save_for_backward) {
    grad_input_saved = at::zeros_like(input_flat);
    if (reduction != Reduction::None) {
      grad_weight_saved = at::zeros_like(linear_weight_ref);
      if (linear_bias_ref.defined()) {
        grad_bias_saved = at::zeros_like(linear_bias_ref);
      }
    }
  }

  const double uniform_component = label_smoothing > 0.0
      ? label_smoothing / static_cast<double>(linear_weight_ref.size(0))
      : 0.0;

  TORCH_CHECK(
      chunk_size > 0,
      "linear_cross_entropy: batch_chunk_size must be positive, got ",
      chunk_size);

  const int64_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t slice = std::min<int64_t>(chunk_size, batch_size - start_idx);
    if (slice <= 0) {
      continue;
    }

    auto input_chunk = input_flat.narrow(0, start_idx, slice);
    auto target_chunk = target_flat.narrow(0, start_idx, slice);
    auto valid_mask_chunk = valid_mask.narrow(0, start_idx, slice);

    auto logits_chunk = at::linear(input_chunk, linear_weight_ref, linear_bias_use);
    auto logsumexp_chunk = at::logsumexp(logits_chunk, {1}, false);

    auto valid_indices = valid_mask_chunk.nonzero().reshape({-1});
    if (valid_indices.numel() > 0) {
      auto local_targets = at::index_select(target_chunk, 0, valid_indices);
      auto selected_logits = at::index_select(logits_chunk, 0, valid_indices);
      auto gathered = at::gather(selected_logits, 1, local_targets.unsqueeze(1)).squeeze(1);
      auto shifted_indices = valid_indices.add(start_idx);
      target_logits.scatter_(0, shifted_indices, gathered);
    }

    if (label_smoothing > 0.0) {
      auto chunk_sum = logits_chunk.sum(-1);
      sum_logits.narrow(0, start_idx, slice).copy_(chunk_sum);
    }

    const auto ce_reduction = (reduction == Reduction::None) ? Reduction::None : Reduction::Sum;
    auto chunk_loss = at::cross_entropy_loss(
        logits_chunk,
        target_chunk,
        std::nullopt,
        ce_reduction,
        ignore_index,
        label_smoothing);

    if (save_for_backward) {
      auto grad_chunk = at::exp(logits_chunk.sub(logsumexp_chunk.unsqueeze(-1)));
      if (label_smoothing > 0.0) {
        grad_chunk = grad_chunk.add(-uniform_component);
      }
      auto mask = valid_mask_chunk.to(grad_chunk.scalar_type()).unsqueeze(1);
      grad_chunk = grad_chunk.mul(mask);
      auto rows = valid_mask_chunk.nonzero().squeeze(-1);
      if (rows.numel() > 0) {
        auto targets_slice = at::index_select(target_chunk, 0, rows).to(at::kLong);
        auto gather = grad_chunk.index({rows, targets_slice}).add(-(1.0 - label_smoothing));
        grad_chunk.index_put_({rows, targets_slice}, gather);
      }
      if (reduction == Reduction::Mean) {
        if (valid_count == 0) {
          grad_chunk.zero_();
        } else {
          grad_chunk.div_(static_cast<double>(valid_count));
        }
      }
      grad_chunk = grad_chunk.mul(mask);
      auto grad_input_chunk = grad_chunk.matmul(linear_weight_ref);
      grad_input_saved.narrow(0, start_idx, slice).copy_(grad_input_chunk);
      if (grad_weight_saved.defined()) {
        grad_weight_saved.add_(grad_chunk.transpose(0, 1).matmul(input_chunk));
      }
      if (grad_bias_saved.defined()) {
        grad_bias_saved.add_(grad_chunk.sum(0));
      }
    }

    if (reduction == Reduction::None) {
      auto dest = losses_buffer.narrow(0, start_idx, slice);
      dest.copy_(chunk_loss);
      dest.masked_fill_(at::logical_not(valid_mask_chunk), 0);
    } else {
      total_loss = total_loss.add(chunk_loss);
    }
  }

  Tensor loss_result;
  if (reduction == Reduction::None) {
    loss_result = losses_buffer.reshape(target.sizes());
  } else if (reduction == Reduction::Sum) {
    loss_result = total_loss;
  } else {
    auto denom_long = valid_mask.sum();
    auto denom = denom_long.to(total_loss.scalar_type());
    if (denom_long.item<int64_t>() == 0) {
      loss_result = denom.div(denom);
    } else {
      loss_result = total_loss.div(denom);
    }
  }

  LinearCrossEntropyForwardResult result;
  result.loss = std::move(loss_result);

  if (save_for_backward) {
    LinearCrossEntropySavedForBackward saved;
    saved.strategy = ChunkingStrategy::BATCH_CHUNKING;
    saved.chunk_size = chunk_size;
    saved.grad_input = grad_input_saved.reshape_as(input);
    if (grad_weight_saved.defined()) {
      saved.grad_weight = std::move(grad_weight_saved);
    }
    if (grad_bias_saved.defined()) {
      saved.grad_bias = std::move(grad_bias_saved);
    }
    result.saved = std::move(saved);
  }

  return result;
}

// The backward implementation mirrors the forward chunking strategy so we never
// materialize a full [N, vocab] tensor while reconstructing gradients.  The
// helpers below break the work into vocabulary-chunked and batch-chunked paths
// and rely exclusively on ATen operators so that the code remains device agnostic
// and benefits from existing BLAS/cuBLAS bindings.
namespace {

inline Tensor zeros_like_tensor(const Tensor& src) {
  return at::_ops::zeros_like::call(src, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

inline bool has_defined_tensor(const std::optional<Tensor>& opt) {
  return opt.has_value() && opt->defined();
}

inline Tensor zeros_like_or_undef(const std::optional<Tensor>& opt) {
  if (has_defined_tensor(opt)) {
    return zeros_like_tensor(opt.value());
  }
  return Tensor();
}

inline Tensor cast_grad_output(const Tensor& grad_output, const Tensor& input) {
  return grad_output.to(input.scalar_type());
}

inline Tensor mask_invalid_rows(const Tensor& tensor, const Tensor& valid_mask) {
  auto mask = valid_mask.to(tensor.scalar_type()).unsqueeze(1);
  return at::mul(tensor, mask);
}

inline void apply_target_updates(
    Tensor& grad_chunk,
    const Tensor& target_flat,
    const Tensor& rows,
    int64_t offset,
    double label_smoothing) {
  if (rows.numel() == 0) {
    return;
  }
  auto selected_targets = at::index_select(target_flat, 0, rows);
  auto local_targets = selected_targets.add(-offset).to(at::kLong);
  auto gather = grad_chunk.index({rows, local_targets}).add(-(1.0 - label_smoothing));
  grad_chunk.index_put_({rows, local_targets}, gather);
}

inline void scale_grad_chunk(
    Tensor& grad_chunk,
    const Tensor& grad_output_tensor,
    const Tensor& grad_output_flat,
    int64_t reduction,
    int64_t valid_count) {
  if (reduction == Reduction::None) {
    grad_chunk.mul_(grad_output_flat.unsqueeze(1));
    return;
  }
  if (reduction == Reduction::Sum) {
    grad_chunk.mul_(grad_output_tensor);
    return;
  }
  TORCH_CHECK(valid_count >= 0, "Valid element count must be non-negative");
  if (valid_count == 0) {
    grad_chunk.zero_();
    return;
  }
  auto scale = grad_output_tensor.div(static_cast<double>(valid_count));
  grad_chunk.mul_(scale);
}

// Computes gradients when the forward pass chose vocabulary chunking.  We run a
// first pass to rebuild the per-sample logsumexp using the same streaming scheme
// as the forward kernel, then revisit each chunk to accumulate gradients for the
// input, weight and (optional) bias tensors.
inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_vocabulary_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    const Tensor& saved_logsumexp,
    bool has_saved) {
  Tensor input_buffer;
  const Tensor& input_ref = contiguous_if_needed(input, input_buffer);
  Tensor linear_weight_buffer;
  const Tensor& linear_weight_ref = contiguous_if_needed(linear_weight, linear_weight_buffer);
  Tensor target_buffer;
  const Tensor& target_ref = contiguous_if_needed(target, target_buffer);
  Tensor grad_output_buffer;
  const Tensor& grad_output_ref = contiguous_if_needed(grad_output, grad_output_buffer);

  Tensor linear_bias_tensor;
  if (linear_bias_opt.has_value()) {
    linear_bias_tensor = linear_bias_opt.value();
  }
  Tensor linear_bias_buffer;
  const Tensor& linear_bias_ref = linear_bias_tensor.defined()
      ? contiguous_if_needed(linear_bias_tensor, linear_bias_buffer)
      : linear_bias_tensor;
  std::optional<Tensor> linear_bias_use;
  if (linear_bias_ref.defined()) {
    linear_bias_use = linear_bias_ref;
  }

  const auto input_flat = input_ref.reshape({-1, input_ref.size(-1)});
  const auto target_flat = target_ref.reshape({-1});
  const auto dtype = input_ref.scalar_type();
  const auto options = input_ref.options();
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor(input);
    Tensor grad_weight = zeros_like_tensor(linear_weight_ref);
    Tensor grad_bias = zeros_like_or_undef(linear_bias_use);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
  }

  TORCH_CHECK(chunk_size > 0, "linear_cross_entropy: vocab_chunk_size must be positive, got ", chunk_size);
  const int64_t vocab_size = linear_weight.size(0);
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  Tensor logsumexp;
  if (has_saved) {
    logsumexp = saved_logsumexp;
  } else {
    Tensor running_max = at::full({input_flat.size(0)}, -std::numeric_limits<double>::infinity(), options).to(dtype);
    Tensor exp_sums = at::zeros({input_flat.size(0)}, options).to(dtype);

    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      const int64_t start_idx = chunk_idx * chunk_size;
      const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
      auto weight_chunk = linear_weight_ref.slice(0, start_idx, end_idx);
      std::optional<Tensor> bias_chunk;
      if (linear_bias_ref.defined()) {
        bias_chunk = linear_bias_ref.slice(0, start_idx, end_idx);
      }
      auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
      auto chunk_max = std::get<0>(logits_chunk.max(-1));
      auto new_max = at::maximum(running_max, chunk_max);
      auto exp_scale_old = at::exp(running_max.sub(new_max));
      auto shifted_logits = logits_chunk.sub(new_max.unsqueeze(-1));
      auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
      exp_sums = exp_sums.mul(exp_scale_old).add_(exp_chunk);
      running_max = new_max;
    }

    logsumexp = running_max.add(exp_sums.log());
  }

  Tensor grad_input = zeros_like_tensor(input_flat);
  Tensor grad_weight = zeros_like_tensor(linear_weight_ref);
  Tensor grad_bias = zeros_like_or_undef(linear_bias_use);

  const double uniform_component = label_smoothing > 0.0
      ? label_smoothing / static_cast<double>(vocab_size)
      : 0.0;
  Tensor grad_output_tensor = cast_grad_output(grad_output_ref, input_ref);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    auto weight_chunk = linear_weight_ref.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (linear_bias_ref.defined()) {
      bias_chunk = linear_bias_ref.slice(0, start_idx, end_idx);
    }
    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
    auto grad_chunk = at::exp(logits_chunk.sub(logsumexp.unsqueeze(-1)));
    if (label_smoothing > 0.0) {
      grad_chunk = grad_chunk.add(-uniform_component);
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask);
    auto lower_bound = target_flat.ge(start_idx);
    auto upper_bound = target_flat.lt(end_idx);
    auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
    target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);
    auto rows = target_chunk_mask.nonzero().squeeze(-1);
    apply_target_updates(grad_chunk, target_flat, rows, start_idx, label_smoothing);
    scale_grad_chunk(grad_chunk, grad_output_tensor, grad_output_flat, reduction, valid_count);
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask);
    grad_input.add_(grad_chunk.matmul(weight_chunk));
    grad_weight.slice(0, start_idx, end_idx).add_(grad_chunk.transpose(0, 1).matmul(input_flat));
    if (grad_bias.defined()) {
      grad_bias.slice(0, start_idx, end_idx).add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.reshape_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
}

inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_batch_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    const Tensor& saved_logsumexp,
    bool has_saved) {
  Tensor input_buffer;
  const Tensor& input_ref = contiguous_if_needed(input, input_buffer);
  Tensor linear_weight_buffer;
  const Tensor& linear_weight_ref = contiguous_if_needed(linear_weight, linear_weight_buffer);
  Tensor target_buffer;
  const Tensor& target_ref = contiguous_if_needed(target, target_buffer);
  Tensor grad_output_buffer;
  const Tensor& grad_output_ref = contiguous_if_needed(grad_output, grad_output_buffer);

  Tensor linear_bias_tensor;
  if (linear_bias_opt.has_value()) {
    linear_bias_tensor = linear_bias_opt.value();
  }
  Tensor linear_bias_buffer;
  const Tensor& linear_bias_ref = linear_bias_tensor.defined()
      ? contiguous_if_needed(linear_bias_tensor, linear_bias_buffer)
      : linear_bias_tensor;
  std::optional<Tensor> linear_bias_use;
  if (linear_bias_ref.defined()) {
    linear_bias_use = linear_bias_ref;
  }

  const auto input_flat = input_ref.reshape({-1, input_ref.size(-1)});
  const auto target_flat = target_ref.reshape({-1});
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor(input);
    Tensor grad_weight = zeros_like_tensor(linear_weight_ref);
    Tensor grad_bias = zeros_like_or_undef(linear_bias_use);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
  }

  const auto options = input_ref.options();

  TORCH_CHECK(chunk_size > 0, "linear_cross_entropy: batch_chunk_size must be positive, got ", chunk_size);
  const int64_t total = input_flat.size(0);
  const int64_t num_chunks = (total + chunk_size - 1) / chunk_size;

  Tensor logsumexp;
  if (has_saved) {
    logsumexp = saved_logsumexp;
  } else {
    logsumexp = at::empty({total}, options);

    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      const int64_t start_idx = chunk_idx * chunk_size;
      const int64_t slice = std::min<int64_t>(chunk_size, total - start_idx);
      if (slice <= 0) {
        continue;
      }
      auto input_chunk = input_flat.narrow(0, start_idx, slice);
      auto logits_chunk = at::linear(input_chunk, linear_weight_ref, linear_bias_use);
      auto logsumexp_chunk = at::logsumexp(logits_chunk, {1}, false);
      logsumexp.narrow(0, start_idx, slice).copy_(logsumexp_chunk);
    }
  }

  Tensor grad_input = zeros_like_tensor(input_flat);
  Tensor grad_weight = zeros_like_tensor(linear_weight_ref);
  Tensor grad_bias = zeros_like_or_undef(linear_bias_use);

  Tensor grad_output_tensor = cast_grad_output(grad_output_ref, input_ref);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  const double uniform_component = label_smoothing > 0.0
      ? label_smoothing / static_cast<double>(linear_weight_ref.size(0))
      : 0.0;

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t slice = std::min<int64_t>(chunk_size, total - start_idx);
    if (slice <= 0) {
      continue;
    }
    auto input_chunk = input_flat.narrow(0, start_idx, slice);
    auto target_chunk = target_flat.narrow(0, start_idx, slice);
    auto valid_mask_chunk = valid_mask.narrow(0, start_idx, slice);
    auto logits_chunk = at::linear(input_chunk, linear_weight_ref, linear_bias_use);
    auto logsumexp_chunk = logsumexp.narrow(0, start_idx, slice);
    auto grad_chunk = at::exp(logits_chunk.sub(logsumexp_chunk.unsqueeze(-1)));
    if (label_smoothing > 0.0) {
      grad_chunk = grad_chunk.add(-uniform_component);
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask_chunk);
    auto rows = valid_mask_chunk.nonzero().squeeze(-1);
    if (rows.numel() > 0) {
      auto targets_slice = at::index_select(target_chunk, 0, rows).to(at::kLong);
      auto gather = grad_chunk.index({rows, targets_slice}).add(-(1.0 - label_smoothing));
      grad_chunk.index_put_({rows, targets_slice}, gather);
    }
    if (reduction == Reduction::None) {
      grad_chunk.mul_(grad_output_flat.narrow(0, start_idx, slice).unsqueeze(1));
    } else if (reduction == Reduction::Sum) {
      grad_chunk.mul_(grad_output_tensor);
    } else {
      if (valid_count == 0) {
        continue;
      }
      grad_chunk.mul_(grad_output_tensor.div(static_cast<double>(valid_count)));
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask_chunk);
    grad_input.narrow(0, start_idx, slice).add_(grad_chunk.matmul(linear_weight_ref));
    grad_weight.add_(grad_chunk.transpose(0, 1).matmul(input_chunk));
    if (grad_bias.defined()) {
      grad_bias.add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.reshape_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
}

} // anonymous namespace

std::tuple<Tensor, Tensor, std::optional<Tensor>> linear_cross_entropy_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy,
    std::optional<int64_t> vocab_chunk_size_opt,
    std::optional<int64_t> batch_chunk_size_opt) {

  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(linear_weight.dim() == 2, "Expected linear_weight to be 2-dimensional, got ", linear_weight.dim());
  TORCH_CHECK(input.size(-1) == linear_weight.size(1),
      "Expected input.size(-1) to match linear_weight.size(1), got ",
      input.size(-1), " and ", linear_weight.size(1));
  TORCH_CHECK(target.device() == input.device(), "Target must be on the same device as input");

  const int64_t flattened_batch = input.numel() / input.size(-1);
  ChunkingStrategy resolved_strategy = select_chunking_strategy(chunking_strategy);

  if (resolved_strategy == ChunkingStrategy::VOCAB_CHUNKING) {
    return backward_vocabulary_chunking(
        input,
        linear_weight,
        target,
        linear_bias_opt,
        grad_output,
        reduction,
        ignore_index,
        label_smoothing,
        vocab_chunk_size_opt.value_or(kDefaultVocabChunkSize),
        Tensor(),
        /*has_saved=*/false);
  }

  const int64_t default_chunk = resolved_strategy == ChunkingStrategy::BATCH_CHUNKING
      ? batch_chunk_size_opt.value_or(kDefaultBatchChunkSize)
      : std::max<int64_t>(flattened_batch, 1);
  return backward_batch_chunking(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      grad_output,
      reduction,
      ignore_index,
      label_smoothing,
      default_chunk,
      Tensor(),
      /*has_saved=*/false);
}

std::tuple<Tensor, Tensor> _linear_cross_entropy_vocab_chunking_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size) {
  auto forward = linear_cross_entropy_forward_vocab_chunking(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      reduction,
      ignore_index,
      label_smoothing,
      chunk_size,
      /*save_for_backward=*/true);
  TORCH_INTERNAL_ASSERT(forward.saved.has_value(), "linear_cross_entropy: missing saved tensors for vocab chunking");
  const auto& saved = forward.saved.value();
  TORCH_INTERNAL_ASSERT(saved.logsumexp.defined(), "linear_cross_entropy: missing logsumexp for vocab chunking");
  return std::make_tuple(std::move(forward.loss), saved.logsumexp);
}

std::tuple<Tensor, Tensor, std::optional<Tensor>> _linear_cross_entropy_vocab_chunking_backward_cpu(
    const Tensor& grad_output,
    const Tensor& saved_logsumexp,
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size) {
  TORCH_CHECK(saved_logsumexp.defined(), "linear_cross_entropy: expected logsumexp for vocab chunking backward");
  return backward_vocabulary_chunking(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      grad_output,
      reduction,
      ignore_index,
      label_smoothing,
      chunk_size,
      saved_logsumexp,
      /*has_saved=*/true);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> _linear_cross_entropy_batch_chunking_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size) {
  auto forward = linear_cross_entropy_forward_batch_chunking(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      reduction,
      ignore_index,
      label_smoothing,
      chunk_size,
      /*save_for_backward=*/true);
  TORCH_INTERNAL_ASSERT(forward.saved.has_value(), "linear_cross_entropy: missing saved tensors for batch chunking");
  const auto& saved = forward.saved.value();
  TORCH_INTERNAL_ASSERT(saved.grad_input.defined(), "linear_cross_entropy: missing grad_input template for batch chunking");

  const bool grad_weight_valid = saved.grad_weight.defined();
  const bool grad_bias_valid = saved.grad_bias.defined();
  auto bool_options = input.options().dtype(at::kBool);
  Tensor grad_weight_flag = at::full({}, grad_weight_valid, bool_options);
  Tensor grad_bias_flag = at::full({}, grad_bias_valid, bool_options);
  return std::make_tuple(
      std::move(forward.loss),
      saved.grad_input,
      saved.grad_weight,
      saved.grad_bias,
      std::move(grad_weight_flag),
      std::move(grad_bias_flag));
}

std::tuple<Tensor, Tensor, std::optional<Tensor>> _linear_cross_entropy_batch_chunking_backward_cpu(
    const Tensor& grad_output,
    const Tensor& saved_grad_input,
    const Tensor& saved_grad_weight,
    const Tensor& saved_grad_bias,
    const Tensor& grad_weight_valid,
    const Tensor& grad_bias_valid,
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size) {
  TORCH_CHECK(saved_grad_input.defined(), "linear_cross_entropy: expected grad_input template for batch chunking backward");

  const bool grad_weight_flag = grad_weight_valid.item<bool>();
  const bool grad_bias_flag = grad_bias_valid.item<bool>();

  if (reduction == Reduction::None || !grad_weight_flag) {
    return backward_batch_chunking(
        input,
        linear_weight,
        target,
        linear_bias_opt,
        grad_output,
        reduction,
        ignore_index,
        label_smoothing,
        chunk_size,
        Tensor(),
        /*has_saved=*/false);
  }

  auto grad_output_tensor = cast_grad_output(grad_output, input);
  Tensor grad_input = saved_grad_input.reshape_as(input).mul(grad_output_tensor);
  Tensor grad_weight = saved_grad_weight.mul(grad_output_tensor);
  std::optional<Tensor> grad_bias_opt;
  if (has_defined_tensor(linear_bias_opt)) {
    if (!grad_bias_flag) {
      auto fallback = backward_batch_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          grad_output,
          reduction,
          ignore_index,
          label_smoothing,
          chunk_size,
          Tensor(),
          /*has_saved=*/false);
      return fallback;
    }
    grad_bias_opt = saved_grad_bias.mul(grad_output_tensor);
  }

  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
}

} // namespace at::native
