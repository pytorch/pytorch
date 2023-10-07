#include <ATen/ATen.h>


namespace at {
namespace native {
namespace preprocessing {

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

} // namespace preprocessing
} // namespace native
} // namespace at
