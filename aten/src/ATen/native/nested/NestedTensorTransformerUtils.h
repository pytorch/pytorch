#include <ATen/ATen.h>

namespace at::native::preprocessing {

/**
 * This function will take nested query, key, and value
 * and will preprocess it in order to run with either
 * the flash-attention or efficient-attention kernels.
 * @return A tuple containing all the necessary data for running the fused
 * kernels
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, Tensor>
sdpa_nested_preprocessing(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value);

/**
 * This function will take nested query, key, and value, grad_out, and out
 * and will preprocess it in order to run with either
 * the flash-attention or efficient-attention kernels backwards.
 * We use both functions to avoid having to do the same preprocessing
 * for cumulative_sequence_length_q and cumulative_sequence_length_kv
 * @return A tuple containing all the necessary data for running the fused
 * kernels
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
sdpa_nested_preprocessing_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_kv,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_kv);

} // namespace at::native::preprocessing
