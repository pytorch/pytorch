#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
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

template<bool has_weight, typename TIndex, typename TData>
struct _CallbackAndBlockSize {
    using TCallback = typename fbgemm::EmbeddingSpMDMKernelSignature<TData, TIndex, TIndex, TData>::Type;

    int64_t blockSize = -1;
    TCallback callback = nullptr;

    static TCallback generateCallback(int64_t block_size) {
        return fbgemm::GenerateEmbeddingSpMDM<TData, TIndex, TIndex, TData>(
                block_size,
                has_weight,
                /* normalize_by_lengths */false,
                /* prefetch */16,
                /* is_weight_positional */false,
                /* use_offsets */true);
    }

    _CallbackAndBlockSize() = default;

    explicit _CallbackAndBlockSize(c10::optional<int64_t> maybe_block_size)
      : blockSize(maybe_block_size.value_or(-1))
      , callback(maybe_block_size.has_value() ? generateCallback(maybe_block_size.value()) : nullptr)
    {}
};

template<typename... StorageMixins>
struct _EmbeddingBagKernelCacheImpl : private StorageMixins... {

    _EmbeddingBagKernelCacheImpl() = default;
    // use each of the mixins to store corresponding kernel and block size
    explicit _EmbeddingBagKernelCacheImpl(c10::optional<int64_t> maybe_block_size)
      : StorageMixins(maybe_block_size)...
    {}

    // this method is thread safe (call sites may call from different threads)
    template<bool has_weight, typename TIndex, typename TData>
    typename _CallbackAndBlockSize<has_weight, TIndex, TData>::TCallback
    getCallback(int64_t block_size) const {
        // if the cache doesn't store the kernel for the incoming block size
        // (so it is different from the one stored in corresponding mixin)
        // regenerate the kernel (not writing it into the cache so we avoid locks)
        if (block_size != _CallbackAndBlockSize<has_weight, TIndex, TData>::blockSize) {
            return _CallbackAndBlockSize<has_weight, TIndex, TData>::generateCallback(block_size);
        }
        // else retrieve the cached kernel from the corresponding mixin
        return _CallbackAndBlockSize<has_weight, TIndex, TData>::callback;
    }
};

// instantiate the cache with the list of storage mixins
// for each of the 8 _EmbeddingBagKernelCache* usages in the EmbeddingBag.cpp impl file
using _EmbeddingBagKernelCache = _EmbeddingBagKernelCacheImpl<
    _CallbackAndBlockSize<true, int32_t, float>,
    _CallbackAndBlockSize<false, int32_t, float>,
    _CallbackAndBlockSize<true, int64_t, float>,
    _CallbackAndBlockSize<false, int64_t, float>,
    _CallbackAndBlockSize<true, int32_t, unsigned short>,
    _CallbackAndBlockSize<false, int32_t, unsigned short>,
    _CallbackAndBlockSize<true, int64_t, unsigned short>,
    _CallbackAndBlockSize<false, int64_t, unsigned short>>;
#else
struct _EmbeddingBagKernelCache {
    explicit _EmbeddingBagKernelCache(c10::optional<int64_t> /* maybe_block_size */) {}
};
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
