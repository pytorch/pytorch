#include <ATen/ATen.h>
#include <cstdint>

#ifdef USE_FBGEMM
#include <fbgemm/FbgemmEmbedding.h>
#endif

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

#ifdef USE_FBGEMM
struct _EmbeddingBagKernelCache {

    template<bool has_weight, typename TIndex, typename TData>
    typename fbgemm::EmbeddingSpMDMKernelSignature<TData, TIndex, TIndex, TData>::Type getCallback(int64_t block_size) {
        return fbgemm::GenerateEmbeddingSpMDM<TData, TIndex, TIndex, TData>(
        block_size,
        has_weight,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
    }
};
#else
struct _EmbeddingBagKernelCache {};
#endif

void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
    Tensor& bag_size, Tensor* max_indices,
    const Tensor &weight, const Tensor &indices,
    const Tensor &offsets, const int64_t mode = 0,
    const c10::optional<Tensor>& per_sample_weights = c10::nullopt,
    bool include_last_offset = false,
    int64_t padding_idx = -1,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache = nullptr);

void _embedding_bag_cpu_out(
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor* p_max_indices,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    const bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights,
    const bool include_last_offset,
    const c10::optional<int64_t>& padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache = nullptr);

} // native
} // at
