#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>
#include <limits>
#include <tuple>
#include <vector>
#include <optional>

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

// Determine optimal chunking strategy based on input dimensions and user preference
// Based on memory reduction analysis and empirical validation
inline ChunkingStrategy select_chunking_strategy(
    int64_t vocab_size, 
    int64_t batch_size, 
    int64_t seq_len, 
    c10::string_view strategy) {
    
    if (strategy == "none") {
        return ChunkingStrategy::NAIVE;
    } else if (strategy == "vocab") {
        return ChunkingStrategy::VOCAB_CHUNKING;
    } else if (strategy == "batch") {
        return ChunkingStrategy::BATCH_CHUNKING;
    } else if (strategy == "auto") {
        // Empirically validated chunk sizes for optimal memory/compute balance
        const int64_t vocab_chunk_size = 4096;   // Same as existing implementation
        const int64_t batch_chunk_size = 1024;   // Optimized for batch processing
        
        const int64_t total_batch_size = batch_size * seq_len;
        
        // Determine which dimensions benefit from chunking
        bool vocab_large = vocab_size > vocab_chunk_size;
        bool batch_large = total_batch_size > batch_chunk_size;
        
        if (!vocab_large && !batch_large) {
            return ChunkingStrategy::NAIVE;
        } else if (vocab_large && !batch_large) {
            return ChunkingStrategy::VOCAB_CHUNKING;
        } else if (!vocab_large && batch_large) {
            return ChunkingStrategy::BATCH_CHUNKING;
        } else {
            // Both dimensions are large - choose strategy with better memory reduction
            // Memory reduction = 1 - (chunk_size / total_size)
            double vocab_reduction = 1.0 - static_cast<double>(vocab_chunk_size) / vocab_size;
            double batch_reduction = 1.0 - static_cast<double>(batch_chunk_size) / total_batch_size;
            
            return (vocab_reduction >= batch_reduction) ? 
                   ChunkingStrategy::VOCAB_CHUNKING : ChunkingStrategy::BATCH_CHUNKING;
        }
    } else {
        TORCH_CHECK(false, "Unknown chunking strategy: ", strategy, 
                   ". Valid options: 'auto', 'vocab', 'batch', 'none'");
    }
}

// Apply final reduction based on reduction mode
// Handles mean/sum reduction consistently across all chunking strategies
// Batch chunking implementation for CPU
// Inspired by Liger Kernel approach: processes input in batch chunks to reduce memory usage
// Memory reduction: [N, V] -> [chunk_size, V] where N = batch_size * seq_len
Tensor batch_chunking_cpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
    
  // Flatten multi-dimensional inputs for processing (standard PyTorch pattern)
  // This allows handling both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
  auto input_flat = input.view({-1, input.size(-1)});  // [N, H] where N = batch * seq_len
  auto target_flat = target.view({-1});                // [N] flattened targets
    
    const int64_t batch_size = input_flat.size(0);
    const int64_t chunk_size = 1024;  // Empirically optimized for batch dimension chunking
    
    // Get bias tensor if provided
    const Tensor& bias = bias_opt.value_or(Tensor());
    
    // Early exit if batch is too small for chunking
    if (batch_size <= chunk_size) {
        auto logits = at::linear(input_flat, weight, bias);
        return at::cross_entropy_loss(logits, target_flat, std::nullopt, reduction, ignore_index, label_smoothing);
    }
    
    const int64_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;
    
    Tensor losses_buffer;
    if (reduction == Reduction::None) {
        losses_buffer = at::zeros({batch_size}, input.options());
    }

    Tensor total_loss = at::zeros({}, input.options());
    int64_t valid_count = 0;
    
    // Process input in batch chunks to avoid materializing large logit tensors
    // Each chunk computes: [chunk_size, hidden] @ [hidden, vocab] -> [chunk_size, vocab]
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int64_t start_idx = chunk_idx * chunk_size;
        int64_t end_idx = std::min(start_idx + chunk_size, batch_size);
        
        if (start_idx >= end_idx) {
            continue;
        }
        
        auto input_chunk = input_flat.slice(0, start_idx, end_idx);   // [actual_chunk_size, H]
        auto target_chunk = target_flat.slice(0, start_idx, end_idx); // [actual_chunk_size]
        auto logits_chunk = at::linear(input_chunk, weight, bias);    // [actual_chunk_size, vocab_size]

        auto valid_mask_chunk = at::ne(target_chunk, ignore_index);
        valid_count += valid_mask_chunk.sum().item<int64_t>();

        const auto ce_reduction = (reduction == Reduction::None) ? Reduction::None : Reduction::Sum;
        auto chunk_loss = at::cross_entropy_loss(
            logits_chunk,
            target_chunk,
            std::nullopt,
            ce_reduction,
            ignore_index,
            label_smoothing);

        if (reduction == Reduction::None) {
            auto dest = losses_buffer.slice(0, start_idx, end_idx);
            dest.copy_(chunk_loss);
            dest.masked_fill_(at::logical_not(valid_mask_chunk), 0);
        } else {
            total_loss = at::add(total_loss, chunk_loss);
        }
    }
    
    if (reduction == Reduction::None) {
        return losses_buffer.view(target.sizes());
    }

    if (reduction == Reduction::Sum || valid_count == 0) {
        return total_loss;
    }
    return at::div(total_loss, valid_count);
}

Tensor linear_cross_entropy_cpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy) {

  // Validate inputs
  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(weight.dim() == 2, "Expected weight to be 2-dimensional, got ", weight.dim());
  TORCH_CHECK(input.size(-1) == weight.size(1), 
              "Expected input.size(-1) to match weight.size(1), got ", 
              input.size(-1), " and ", weight.size(1));
  
  // Get bias tensor if provided
  const Tensor& bias = bias_opt.value_or(Tensor());
  
  // Pick a chunking strategy that mirrors the Python wrapper so we only
  // materialise large logit tensors when it is worthwhile.  Vocabulary chunking
  // slices the weight matrix (large vocabularies), batch chunking slices the
  // flattened batch (very large batches), and the naive path keeps the original
  // computation for small problems.
  
  // Calculate input dimensions for strategy selection
  const int64_t vocab_size = weight.size(0);
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.dim() == 3 ? input.size(1) : 1;
  
  // Select optimal chunking strategy based on input characteristics and user preference
  ChunkingStrategy selected_strategy = select_chunking_strategy(vocab_size, batch_size, seq_len, chunking_strategy);
  
  auto input_flat = input.view({-1, input.size(-1)});  // [N, H]
  auto target_flat = target.view({-1});                // [N]
  auto valid_mask = at::ne(target_flat, ignore_index);

  // Execute selected chunking strategy
  if (selected_strategy == ChunkingStrategy::VOCAB_CHUNKING) {
    const int64_t chunk_size = 4096;  // Empirically validated chunk size for optimal memory/compute balance
    const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

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

    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      const int64_t start_idx = chunk_idx * chunk_size;
      const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);

      auto weight_chunk = weight.slice(0, start_idx, end_idx);  // [chunk, hidden]

      std::optional<Tensor> bias_chunk;
      if (bias.defined()) {
        bias_chunk = bias.slice(0, start_idx, end_idx);
      }

      auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);  // [N, chunk]

      if (label_smoothing > 0.0) {
        sum_logits = at::add(sum_logits, at::sum(logits_chunk, {-1}));
      }

      auto chunk_max = std::get<0>(logits_chunk.max(-1));
      auto new_max = at::maximum(running_max, chunk_max);

      auto exp_scale_old = at::exp(at::sub(running_max, new_max));
      auto shifted_logits = at::sub(logits_chunk, new_max.unsqueeze(-1));
      auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
      exp_sums = at::add(at::mul(exp_sums, exp_scale_old), exp_chunk);
      running_max = new_max;

      auto lower_bound = at::ge(target_flat, start_idx);
      auto upper_bound = at::lt(target_flat, end_idx);
      auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
      target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);

      auto indices = target_chunk_mask.nonzero().view({-1});
      if (indices.numel() > 0) {
        auto selected_targets = at::index_select(target_flat, 0, indices);
        auto local_targets = at::sub(selected_targets, start_idx);
        auto selected_logits = at::index_select(logits_chunk, 0, indices);
        auto gathered = at::gather(selected_logits, 1, local_targets.unsqueeze(1)).squeeze(1);
        target_logits.index_put_({indices}, gathered);
        auto ones_update = at::ones(indices.sizes(), long_options);
        target_found.index_put_({indices}, ones_update);
      }
    }

    auto target_found_mask = target_found.gt(0);
    auto coverage_mask = at::logical_or(target_found_mask, at::logical_not(valid_mask));
    TORCH_CHECK(coverage_mask.all().item<bool>(),
        "linear_cross_entropy: target index not found in vocabulary chunks");

    auto logsumexp = at::add(running_max, at::log(exp_sums));
    Tensor losses;
    if (label_smoothing > 0.0) {
      const double smoothing = label_smoothing;
      const double uniform = smoothing / static_cast<double>(vocab_size);
      auto main_term = at::mul(target_logits, 1.0 - smoothing);
      auto uniform_term = at::mul(sum_logits, uniform);
      losses = at::sub(logsumexp, main_term);
      losses = at::sub(losses, uniform_term);
    } else {
      losses = at::sub(logsumexp, target_logits);
    }

    auto invalid_mask = at::logical_not(valid_mask);
    losses.masked_fill_(invalid_mask, 0);

    if (reduction == Reduction::None) {
      return losses.view(target.sizes());
    }

    auto total_loss = losses.sum();
    if (reduction == Reduction::Sum) {
      return total_loss;
    }

    const int64_t valid_count = valid_mask.sum().item<int64_t>();
    if (valid_count == 0) {
      return total_loss; // Match cross_entropy behaviour when all targets ignored
    }
    return at::div(total_loss, valid_count);

  } else if (selected_strategy == ChunkingStrategy::BATCH_CHUNKING) {
    // Batch chunking implementation - call dedicated function
    return batch_chunking_cpu(input, weight, target, bias_opt, reduction, ignore_index, label_smoothing);
    
  } else { // ChunkingStrategy::NAIVE
    // Naive implementation for small models or when chunking not beneficial
    auto logits = at::linear(input, weight, bias);
    auto logits_flat = logits.view({-1, logits.size(-1)});
    return at::cross_entropy_loss(logits_flat, target_flat, std::nullopt, reduction, ignore_index, label_smoothing);
  }
}

// The backward implementation mirrors the forward chunking strategy so we never
// materialize a full [N, vocab] tensor while reconstructing gradients.  The
// helpers below break the work into vocabulary-chunked and batch-chunked paths
// and rely exclusively on ATen operators so that the code remains device agnostic
// and benefits from existing BLAS/cuBLAS bindings.
namespace {

inline const char* reduction_to_string(int64_t reduction) {
  if (reduction == Reduction::Mean) {
    return "mean";
  }
  if (reduction == Reduction::Sum) {
    return "sum";
  }
  TORCH_CHECK(reduction == Reduction::None, "Unsupported reduction value: ", reduction);
  return "none";
}

inline bool requires_grad_scaling_none(int64_t reduction) {
  return reduction == Reduction::None;
}

inline bool requires_grad_scaling_mean(int64_t reduction) {
  return reduction == Reduction::Mean;
}

inline Tensor zeros_like_tensor(const Tensor& src) {
  return at::_ops::zeros_like::call(src, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

inline Tensor zeros_like_or_undef(const std::optional<Tensor>& opt) {
  if (opt.has_value()) {
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
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    ChunkingStrategy resolved_strategy) {
  const auto input_flat = input.view({-1, input.size(-1)});
  const auto target_flat = target.view({-1});
  const auto dtype = input.scalar_type();
  const auto options = input.options();
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor(input);
    Tensor grad_weight = zeros_like_tensor(weight);
    Tensor grad_bias = zeros_like_or_undef(bias_opt);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
  }

  const int64_t vocab_size = weight.size(0);
  const int64_t chunk_size = 4096;
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  Tensor running_max = at::full({input_flat.size(0)}, -std::numeric_limits<double>::infinity(), options).to(dtype);
  Tensor exp_sums = at::zeros({input_flat.size(0)}, options).to(dtype);

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (bias_opt.has_value()) {
      bias_chunk = bias_opt->slice(0, start_idx, end_idx);
    }
    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
    auto chunk_max = std::get<0>(logits_chunk.max(-1));
    auto new_max = at::maximum(running_max, chunk_max);
    auto exp_scale_old = at::exp(running_max.sub(new_max));
    auto shifted_logits = logits_chunk.sub(new_max.unsqueeze(-1));
    auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
    exp_sums = at::add(at::mul(exp_sums, exp_scale_old), exp_chunk);
    running_max = new_max;
  }

  Tensor logsumexp = running_max.add(exp_sums.log());
  Tensor grad_input = zeros_like_tensor(input_flat);
  Tensor grad_weight = zeros_like_tensor(weight);
  Tensor grad_bias = zeros_like_or_undef(bias_opt);

  const double uniform_component = label_smoothing > 0.0 ? label_smoothing / static_cast<double>(vocab_size) : 0.0;
  Tensor grad_output_tensor = cast_grad_output(grad_output, input);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    auto weight_chunk = weight.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (bias_opt.has_value()) {
      bias_chunk = bias_opt->slice(0, start_idx, end_idx);
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

  grad_input = grad_input.view_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
}

// Computes gradients when we chunked the batch dimension (or not at all).  The
// loop keeps the working set bounded by `chunk_size` rows so that we never
// allocate a full [N, vocab] buffer even when the batch is very large.
inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_batch_chunking(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size) {
  const auto input_flat = input.view({-1, input.size(-1)});
  const auto target_flat = target.view({-1});
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor(input);
    Tensor grad_weight = zeros_like_tensor(weight);
    Tensor grad_bias = zeros_like_or_undef(bias_opt);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
  }

  Tensor grad_input = zeros_like_tensor(input_flat);
  Tensor grad_weight = zeros_like_tensor(weight);
  Tensor grad_bias = zeros_like_or_undef(bias_opt);

  Tensor grad_output_tensor = cast_grad_output(grad_output, input);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  const double uniform_component = label_smoothing > 0.0 ? label_smoothing / static_cast<double>(weight.size(0)) : 0.0;
  const int64_t total = input_flat.size(0);

  for (int64_t start_idx = 0; start_idx < total; start_idx += chunk_size) {
    const int64_t slice = std::min<int64_t>(chunk_size, total - start_idx);
    auto input_chunk = input_flat.narrow(0, start_idx, slice);
    auto target_chunk = target_flat.narrow(0, start_idx, slice);
    auto valid_mask_chunk = valid_mask.narrow(0, start_idx, slice);
    auto logits_chunk = at::linear(input_chunk, weight, bias_opt);
    auto logsumexp_chunk = at::_ops::logsumexp::call(logits_chunk, std::vector<int64_t>{1}, false);
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
    Tensor grad_scale = grad_output_tensor;
    if (reduction == Reduction::None) {
      grad_chunk.mul_(grad_output_flat.narrow(0, start_idx, slice).unsqueeze(1));
    } else if (reduction == Reduction::Sum) {
      grad_chunk.mul_(grad_scale);
    } else {
      if (valid_count == 0) {
        continue;
      }
      grad_chunk.mul_(grad_scale.div(static_cast<double>(valid_count)));
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask_chunk);
    grad_input.narrow(0, start_idx, slice).add_(grad_chunk.matmul(weight));
    grad_weight.add_(grad_chunk.transpose(0, 1).matmul(input_chunk));
    if (grad_bias.defined()) {
      grad_bias.add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.view_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(grad_input, grad_weight, std::move(grad_bias_opt));
}

} // anonymous namespace

std::tuple<Tensor, Tensor, std::optional<Tensor>> linear_cross_entropy_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& target,
    const std::optional<Tensor>& bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy) {

  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(weight.dim() == 2, "Expected weight to be 2-dimensional, got ", weight.dim());
  TORCH_CHECK(input.size(-1) == weight.size(1),
      "Expected input.size(-1) to match weight.size(1), got ",
      input.size(-1), " and ", weight.size(1));
  TORCH_CHECK(target.device() == input.device(), "Target must be on the same device as input");

  const int64_t vocab_size = weight.size(0);
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.dim() == 3 ? input.size(1) : 1;
  ChunkingStrategy resolved_strategy = select_chunking_strategy(vocab_size, batch_size, seq_len, chunking_strategy);

  if (resolved_strategy == ChunkingStrategy::VOCAB_CHUNKING) {
    return backward_vocabulary_chunking(
        input,
        weight,
        target,
        bias_opt,
        grad_output,
        reduction,
        ignore_index,
        label_smoothing,
        resolved_strategy);
  }

  const int64_t default_chunk = resolved_strategy == ChunkingStrategy::BATCH_CHUNKING ? 1024 : input.view({-1, input.size(-1)}).size(0);
  return backward_batch_chunking(
      input,
      weight,
      target,
      bias_opt,
      grad_output,
      reduction,
      ignore_index,
      label_smoothing,
      default_chunk);
}

} // namespace at::native
