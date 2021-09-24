#include <ATen/ATen.h>

namespace at {
namespace native {

void check_arguments(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const c10::optional<Tensor>& per_sample_weights,
    bool include_last_offset);

void make_bag_size_out(
    Tensor& bag_size_out,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad);

void make_max_indices_out(
    Tensor& max_indices_out,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset);

void make_offset2bag_out(
    Tensor& offset2bag,
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const c10::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx = -1);

void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
    Tensor& bag_size, Tensor& max_indices,
    const Tensor &weight, const Tensor &indices,
    const Tensor &offsets, const int64_t mode = 0,
    const c10::optional<Tensor>& per_sample_weights = c10::nullopt,
    bool include_last_offset = false,
    int64_t padding_idx = -1);
} // native
} // at
