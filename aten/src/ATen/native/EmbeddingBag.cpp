#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>

#include <ATen/native/CPUBlas.h>
#include <ATen/native/NonSymbolicBC.h>

#include <c10/util/irange.h>
#include <c10/util/Half.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#else
#include <caffe2/perfkernels/embedding_lookup_idx.h>
#endif

#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_embedding_bag_backward_native.h>
#include <ATen/ops/_embedding_bag_dense_backward.h>
#include <ATen/ops/_embedding_bag_dense_backward_native.h>
#include <ATen/ops/_embedding_bag_forward_only.h>
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>
#include <ATen/ops/_embedding_bag_sparse_backward.h>
#include <ATen/ops/_embedding_bag_sparse_backward_native.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_bag_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

namespace at {
namespace native {

template<typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

static void make_offset2bag(const Tensor &offsets, Tensor& offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0, offset2bag.scalar_type());     // offset2bag = [0 0 1 1 2]
}

namespace {

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

// Determines if we can use a fast implementation for index_select_add, which
// is only applicable if special conditions are met
template<typename index_t>
bool is_fast_path_index_select(const Tensor& src, Tensor& output, index_t padding_idx) {
  auto fast_path = src.strides()[1] == 1 && output.strides()[1] == 1 &&
      padding_idx < static_cast<index_t>(0);
  // check dtype supported
  auto dtype_supported =
      src.scalar_type() == kFloat || src.scalar_type() == kHalf;
#if !defined(USE_FBGEMM) // Caffe2 additionally support bf16
  dtype_supported = dtype_supported || src.scalar_type() == kBFloat16;
#endif
  return fast_path && dtype_supported;
}

// Determines if we can use a fast implementation for index_select_scale_add,
// which is only applicable if special conditions are met
template<typename index_t>
bool is_fast_path_index_select_scale(const Tensor& src, const Tensor& scale, Tensor& output, index_t padding_idx) {
  auto fast_path = src.strides()[1] == 1 && output.strides()[1] == 1 &&
      scale.strides()[0] == 1 && padding_idx < static_cast<index_t>(0);
  // check dtype supported
  auto dtype_supported =
      src.scalar_type() == kFloat || src.scalar_type() == kHalf;
#if !defined(USE_FBGEMM) // Caffe2 additionally support bf16
  dtype_supported = dtype_supported || src.scalar_type() == kBFloat16;
#endif
  return fast_path && dtype_supported;
}

template<typename index_t>
bool is_fast_path(const Tensor& src, const c10::optional<Tensor>& scale, Tensor& output, index_t padding_idx) {
  return (scale.has_value() && scale.value().defined()) ?
         is_fast_path_index_select_scale(src, scale.value(), output, padding_idx) :
         is_fast_path_index_select(src, output, padding_idx);
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template <typename data_t, typename index_t>
static typename std::enable_if<std::is_same<data_t, double>::value, void>::type
index_select_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& src,
    Tensor& output,
    const Tensor& /*offsets*/,
    bool /*include_last_offset*/,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* /* fbgemm_kernel_cache */) {
  TORCH_CHECK(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.data_ptr<index_t>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* src_data = src.data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto vocab_size = src.size(0);
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];

  for (const auto i : c10::irange(numel)) {
    // We can skip indices equal to padding_idx so they are not included in
    // the reduction
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    if (idx != padding_idx) {
      at::native::cpublas::axpy<data_t>(ddim, 1,
              src_data + src_stride0 * idx, src_stride1,
              output_data + output_stride0 * add_indices_data[i], output_stride1);
    } else if (bag_size.defined()) {
      // Decrement bag_size to reflect that the index is padded
      // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
      bag_size_data[add_indices_data[i]]--;
    }
  }
}

namespace {
template <typename index_t>
void fbgemm_spmdm_report_error_(
    int64_t output_size,
    int index_size,
    int64_t N,
    const index_t* offsets,
    const index_t* indices) {
  for (const auto m : c10::irange(output_size)) {
    for (index_t i = offsets[m]; i < offsets[m + 1]; ++i) {
      TORCH_CHECK(i < index_size);
      index_t idx = indices[i];
      TORCH_CHECK(
          0 <= idx && idx < N,
          "Index ",
          i,
          " is out of bounds: ",
          idx,
          ", range 0 to ",
          N);
    }
  }
  TORCH_CHECK(
      offsets[output_size] == index_size,
      "Yout input seems to be incorrect: the last offset value should be "
      "the size of the indices tensor, but it appears not.");
}
} // namespace

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, at::Half>::value ||
        std::is_same<data_t, at::BFloat16>::value,
    void>::type
index_select_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& src,
    Tensor& output,
    const Tensor& offsets,
    bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<data_t>();

  if (is_fast_path_index_select(src, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.data_ptr<data_t>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      if (offsets.numel() > 0) {
        std::memcpy(
            offsets_include_last.data(),
            offsets.data_ptr<index_t>(),
            sizeof(index_t) * offsets.numel());
      }
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }
#if defined(USE_FBGEMM)
    if (std::is_same<data_t, at::Half>::value) {
      using float16 = uint16_t;
      auto kernel_fp16_index_t = fbgemm_kernel_cache
          ? fbgemm_kernel_cache
                ->getCallback</* has_weight */ false, index_t, float16>(ddim)
          : fbgemm::GenerateEmbeddingSpMDM<float16, index_t, index_t, float16>(
                /* block_size */ ddim,
                /* has_weight */ false,
                /* normalize_by_lengths */ false,
                /* prefetch */ 16,
                /* is_weight_positional */ false,
                /* use_offsets */ true);
      at::parallel_for(
          0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
            bool success = kernel_fp16_index_t(
                /* output_size */ end_idx - start_idx,
                /* index_size */ offsets_data[end_idx] -
                    offsets_data[start_idx],
                /* data_size */ src.size(0),
                /* input */ reinterpret_cast<const float16*>(src_data),
                /* indices */ select_indices_data + offsets_data[start_idx],
                /* offsets_or_lengths */ offsets_data + start_idx,
                /* weights */ nullptr,
                /* output */
                reinterpret_cast<float16*>(output_data + start_idx * ddim));
            if (!success) {
              fbgemm_spmdm_report_error_(
                  end_idx - start_idx,
                  offsets_data[end_idx] - offsets_data[start_idx],
                  src.size(0),
                  offsets_data + start_idx,
                  select_indices_data + offsets_data[start_idx]);
            }
          });
      return;
    }

#else
    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 = at::zeros({output_size, ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();
    using bVec = vec::Vectorized<BFloat16>;
    using fVec = vec::Vectorized<float>;
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data_fp32 + start_idx * ddim);
          for (int64_t i = start_idx; i < end_idx; i++) {
            // Convert FP32 intermediate buffer result back to FP16/BF16 for
            // output dtype
            if (std::is_same<data_t, at::Half>::value) {
              // FP16
              for (const auto d : c10::irange(ddim)) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            } else {
              // BF16
              int64_t d = 0;
              for (; d < ddim - (ddim % bVec::size()); d += bVec::size()) {
                fVec temp_fp32_0 = fVec::loadu(output_data_fp32 + ddim * i + d);
                fVec temp_fp32_1 =
                    fVec::loadu(output_data_fp32 + ddim * i + d + fVec::size());
                convert_float_bfloat16(temp_fp32_0, temp_fp32_1)
                    .store(output_data + i * ddim + d);
              }
              for (; d < ddim; d++) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            }
          }
        });
    return;
#endif /* BUILD_CAFFE2 */
  }
  TORCH_CHECK(select_indices.numel() == add_indices.numel());
  auto* src_data = src.data_ptr<data_t>();
  auto* add_indices_data = add_indices.data_ptr<index_t>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  auto vocab_size = src.size(0);
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];
  auto numel = add_indices.numel();

  Tensor src_fp32 = at::empty({ddim}, src.options().dtype(at::kFloat));
  auto* src_data_fp32 = src_fp32.data_ptr<float>();

  // Initialize the intermediate output buffer to be 0.
  Tensor output_fp32 =
      at::zeros({output.size(0), ddim}, output.options().dtype(at::kFloat));
  auto* output_data_fp32 = output_fp32.data_ptr<float>();

  for (const auto i : c10::irange(numel)) {
    // We can skip indices equal to padding_idx so they are not included in
    // the reduction
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    if (idx != padding_idx) {
      // Copy src_data + src_stride0 * idx to src_data_fp32
      for (const auto d : c10::irange(ddim)) {
        src_data_fp32[d] =
            static_cast<float>((src_data + src_stride0 * idx)[d * src_stride1]);
      }
      at::native::cpublas::axpy<float>(
          ddim,
          1,
          src_data_fp32,
          1,
          output_data_fp32 + ddim * add_indices_data[i],
          1);

    } else if (bag_size.defined()) {
      // Decrement bag_size to reflect that the index is padded
      // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
      bag_size_data[add_indices_data[i]]--;
    }
  }
  for (const auto i : c10::irange(output.size(0))) {
    // Convert FP32 intermediate buffer result back to FP16/BF16 for output
    // dtype
    for (const auto d : c10::irange(ddim)) {
      (output_data + output_stride0 * i)[d * output_stride1] =
          static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
    }
  }
}

template<typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, float>::value, void>::type
index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& offsets,
                             bool include_last_offset,
                             Tensor &bag_size,
                             index_t padding_idx,
                             _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  if (is_fast_path_index_select(src, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.data_ptr<float>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      if (offsets.numel() > 0) {
        std::memcpy(
            offsets_include_last.data(),
            offsets.data_ptr<index_t>(),
            sizeof(index_t) * offsets.numel());
      }
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

#ifdef USE_FBGEMM
    auto kernel_fp32_index_t =
      fbgemm_kernel_cache ?
      fbgemm_kernel_cache->getCallback</* has_weight */ false, index_t, float>(ddim) :
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */false,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          bool success = kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */nullptr,
            /* output */output_data + start_idx * ddim);
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.data_ptr<float>();
    auto* add_indices_data = add_indices.data_ptr<index_t>();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto numel = add_indices.numel();
    for (const auto i : c10::irange(numel)) {
      // We can skip indices equal to padding_idx so they are not included in
      // the reduction
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        at::native::cpublas::axpy<float>(
            ddim,
            1,
            src_data + src_stride0 * idx,
            src_stride1,
            output_data + output_stride0 * add_indices_data[i],
            output_stride1);
      } else if (bag_size.defined()) {
        // Decrement bag_size to reflect that the index is padded
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        bag_size_data[add_indices_data[i]]--;
      }
    }
  }
}

// This function fuses the following three fns:
// index_select (using select_indices as the index)
// mul (scaling by per_sample_weights)
// index_add (using add_indices as the index)
template <typename data_t, typename index_t>
static typename std::enable_if<std::is_same<data_t, double>::value, void>::type
index_select_scale_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& scale,
    const Tensor& src,
    Tensor& output,
    const Tensor& /*offsets*/,
    bool /*include_last_offset*/,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* /* fbgemm_kernel_cache */) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.data_ptr<index_t>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* src_data = src.data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto vocab_size = src.size(0);
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];

  auto* scale_data = scale.data_ptr<data_t>();
  auto scale_stride = scale.strides()[0];

  for (const auto i : c10::irange(numel)) {
    // We can skip indices equal to padding_idx so they are not included in
    // the reduction
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    if (idx != padding_idx) {
      auto* src_base = src_data + src_stride0 * idx;
      auto* output_base = output_data + output_stride0 * add_indices_data[i];
      auto scale = scale_data[i * scale_stride];
      for (const auto j : c10::irange(ddim)) {
        output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
      }
    } else if (bag_size.defined()) {
      // Decrement bag_size to reflect that the index is padded
      // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
      bag_size_data[add_indices_data[i]]--;
    }
  }
}

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, at::Half>::value ||
        std::is_same<data_t, at::BFloat16>::value,
    void>::type
index_select_scale_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& scale,
    const Tensor& src,
    Tensor& output,
    const Tensor& offsets,
    bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.data_ptr<data_t>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<data_t>();

  if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.data_ptr<data_t>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

    Tensor scale_fp32 = at::empty(scale.sizes(), scale.options().dtype(at::kFloat));
    auto* scale_data_fp32 = scale_fp32.data_ptr<float>();

#if defined(USE_FBGEMM)
    if (std::is_same<data_t, at::Half>::value) {
      using float16 = uint16_t;
      fbgemm::Float16ToFloat_simd(
          reinterpret_cast<const float16*>(scale_data),
          scale_data_fp32,
          scale_fp32.numel());
      auto kernel_fp16_index_t = fbgemm_kernel_cache
          ? fbgemm_kernel_cache
                ->getCallback</* has_weight */ true, index_t, float16>(ddim)
          : fbgemm::GenerateEmbeddingSpMDM<float16, index_t, index_t, float16>(
                /* block_size */ ddim,
                /* has_weight */ true,
                /* normalize_by_lengths */ false,
                /* prefetch */ 16,
                /* is_weight_positional */ false,
                /* use_offsets */ true);
      at::parallel_for(
          0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
            bool success = kernel_fp16_index_t(
                /* output_size */ end_idx - start_idx,
                /* index_size */ offsets_data[end_idx] -
                    offsets_data[start_idx],
                /* data_size */ src.size(0),
                /* input */ reinterpret_cast<const float16*>(src_data),
                /* indices */ select_indices_data + offsets_data[start_idx],
                /* offsets_or_lengths */ offsets_data + start_idx,
                /* weights */ scale_data_fp32 + offsets_data[start_idx],
                /* output */
                reinterpret_cast<float16*>(output_data + start_idx * ddim));
            if (!success) {
              fbgemm_spmdm_report_error_(
                  end_idx - start_idx,
                  offsets_data[end_idx] - offsets_data[start_idx],
                  src.size(0),
                  offsets_data + start_idx,
                  select_indices_data + offsets_data[start_idx]);
            }
          });
      return;
    }
#else
    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 =
        at::zeros({output_size, ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();
    for (const auto i : c10::irange(scale.numel())) {
      scale_data_fp32[i] = static_cast<float>(scale_data[i]);
    }
    using bVec = vec::Vectorized<BFloat16>;
    using fVec = vec::Vectorized<float>;
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data_fp32 + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data_fp32 + start_idx * ddim);
          for (int64_t i = start_idx; i < end_idx; i++) {
            // Convert FP32 intermediate buffer result back to FP16/BF16 for
            // output dtype
            if (std::is_same<data_t, at::Half>::value) {
              // FP16
              for (const auto d : c10::irange(ddim)) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            } else {
              // BF16
              int64_t d = 0;
              for (; d < ddim - (ddim % bVec::size()); d += bVec::size()) {
                fVec temp_fp32_0 = fVec::loadu(output_data_fp32 + ddim * i + d);
                fVec temp_fp32_1 =
                    fVec::loadu(output_data_fp32 + ddim * i + d + fVec::size());
                convert_float_bfloat16(temp_fp32_0, temp_fp32_1)
                    .store(output_data + i * ddim + d);
              }
              for (; d < ddim; d++) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            }
          }
        });
    return;
#endif /* BUILD_CAFFE2 */
  }
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* src_data = src.data_ptr<data_t>();
  auto* add_indices_data = add_indices.data_ptr<index_t>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  auto vocab_size = src.size(0);
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];
  auto scale_stride = scale.strides()[0];
  auto numel = add_indices.numel();

  // Initialize the intermediate output buffer to be 0.
  Tensor output_fp32 =
      at::zeros({output.size(0), ddim}, output.options().dtype(at::kFloat));
  auto* output_data_fp32 = output_fp32.data_ptr<float>();

  for (const auto i : c10::irange(numel)) {
    // We can skip indices equal to padding_idx so they are not included in
    // the reduction
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    if (idx != padding_idx) {
      auto* src_base = src_data + src_stride0 * idx;
      auto* output_base_fp32 = output_data_fp32 + ddim * add_indices_data[i];
      auto scale = scale_data[i * scale_stride];
      for (const auto j : c10::irange(ddim)) {
        output_base_fp32[j] += static_cast<float>(src_base[j * src_stride1]) *
            static_cast<float>(scale);
      }
    } else if (bag_size.defined()) {
      // Decrement bag_size to reflect that the index is padded
      // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
      bag_size_data[add_indices_data[i]]--;
    }
  }
  for (const auto i : c10::irange(output.size(0))) {
    // Convert FP32 intermediate buffer result back to FP16/BF16 for output
    // dtype
    for (const auto d : c10::irange(ddim)) {
      (output_data + output_stride0 * i)[d * output_stride1] =
          static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
    }
  }
}

template<typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, float>::value, void>::type
index_select_scale_add(const Tensor &select_indices,
                                          const Tensor &add_indices,
                                          const Tensor &scale,
                                          const Tensor &src,
                                          Tensor &output,
                                          const Tensor& offsets,
                                          bool include_last_offset,
                                          Tensor &bag_size,
                                          index_t padding_idx,
                                          _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.data_ptr<float>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.data_ptr<float>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

#ifdef USE_FBGEMM
    auto kernel_fp32_index_t =
      fbgemm_kernel_cache ?
      fbgemm_kernel_cache->getCallback</* has_weight */ true, index_t, float>(ddim) :
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */true,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          bool success = kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */scale_data + offsets_data[start_idx],
            /* output */output_data + start_idx * ddim);
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.data_ptr<float>();
    auto* add_indices_data = add_indices.data_ptr<index_t>();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto scale_stride = scale.strides()[0];
    auto numel = add_indices.numel();


    for (const auto i : c10::irange(numel)) {
      // We can skip indices equal to padding_idx so they are not included in
      // the reduction
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        auto* src_base = src_data + src_stride0 * idx;
        auto* output_base = output_data + output_stride0 * add_indices_data[i];
        auto scale = scale_data[i * scale_stride];
        for (const auto j : c10::irange(ddim)) {
          output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
        }
      } else if (bag_size.defined()) {
        // Decrement bag_size to reflect that the index is padded
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        bag_size_data[add_indices_data[i]]--;
      }
    }
  }
}

}  // namespace

void check_arguments(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const c10::optional<Tensor>& per_sample_weights,
    bool include_last_offset) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes(
      "embedding_bag", weight_arg, {kHalf, kBFloat16, kFloat, kDouble});

  AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_embedding_bag_cpu_impl", [&]() {
    if (offsets.size(0) > 0) {
      index_t offset_0 = offsets.data_ptr<index_t>()[0];
      index_t offset_n = offsets.data_ptr<index_t>()[offsets.size(0)-1];
      TORCH_CHECK(offset_0 == 0, "offsets[0] has to be 0, i.e., the first sequence "
                                "in the mini-batch has to start from position 0. "
                                "However, got ", offsets[0]);
      TORCH_CHECK(offset_n <= indices.size(0), "offsets[-1] can not "
                  "be greater than input's length ", indices.size(0), " but got offsets[-1] of ",
                  offset_n);
    }
  });

  if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
    TORCH_CHECK(mode == MODE_SUM,
        "embedding_bag: per_sample_weights only supported with mode='sum'");
    auto per_input_weights_arg = TensorArg(
        per_sample_weights.value(),"per_sample_weights", 1);
    checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
    TORCH_CHECK(per_sample_weights.value().dim() == 1);
    TORCH_CHECK(per_sample_weights.value().numel() == indices.numel());
  }

  if (include_last_offset) {
    TORCH_CHECK(
        offsets.size(0) >= 1,
        "include_last_offset: number of offset should be at least 1");
  }
}

void make_bag_size_out(
    Tensor& bag_size_out,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad) {
  if (requires_grad || mode == MODE_MEAN || mode == MODE_MAX) {
    auto num_bags = offsets.size(0) - (include_last_offset ? 1 : 0);
    at::native::resize_(bag_size_out, {num_bags}, c10::nullopt);
    // Compute this for MODE_MEAN and MODE_MAX (latter needed for backwards)
    if (num_bags != 1) {
      bag_size_out.slice(0, 0, bag_size_out.size(0) - 1, 1) =
          offsets.slice(0, 1, num_bags, 1) -
          offsets.slice(0, 0, num_bags - 1, 1);
    }
    if (num_bags > 0) {
      bag_size_out[-1] = indices.size(0) - offsets[num_bags - 1];
    }
  } else {
    at::native::resize_(bag_size_out, offsets.sizes(), c10::nullopt);
  }
}

void make_max_indices_out(
    Tensor& max_indices_out,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset) {
  int64_t numBags = offsets.size(0);
  if (mode == MODE_MAX) {
    if (include_last_offset) {
      TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
      numBags -= 1;
    }
    at::native::resize_(max_indices_out, {numBags, weight.sizes()[1]}, c10::nullopt);
    at::native::zero_(max_indices_out);
  } else {
      at::native::resize_(max_indices_out, bag_size.sizes(), c10::nullopt);
  }
}

void make_offset2bag_out(
    Tensor& offset2bag,
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const c10::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx) {
  // To save compute, if we are going to go down the fast path case for the 'sum'
  // mode, we skip calculating offset2bag, since it is not going to be used.
  bool fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx);

  if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum) {
    at::native::resize_(offset2bag, {indices.size(0) + 1}, c10::nullopt);
    at::native::zero_(offset2bag);

    make_offset2bag(offsets, offset2bag);
    at::native::resize_(offset2bag, {indices.size(0)}, c10::nullopt);
    // only initialize output in slow path
    at::native::zero_(output);
  }
}

static Tensor make_bag_size(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad) {
  Tensor bag_size = at::empty(offsets.sizes(), offsets.options());
  make_bag_size_out(bag_size, offsets, indices, mode, include_last_offset, requires_grad);
  return bag_size;
}

static Tensor make_max_indices(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset) {
  Tensor max_indices = at::empty(bag_size.sizes(), offsets.options());
  make_max_indices_out(max_indices, weight, indices, offsets, bag_size, mode, include_last_offset);
  return max_indices;
}

static Tensor make_offset2bag(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const c10::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx) {
  Tensor offset2bag = at::empty({0}, offsets.options());
  make_offset2bag_out(offset2bag, output, weight, indices, offsets, mode, per_sample_weights, padding_idx);
  return offset2bag;
}

static Tensor apply_bag_size(
    const int64_t mode,
    Tensor &output,
    const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                         .to(output.options())
                         .unsqueeze(1)
                         .expand_as(output);
    output /= bag_size_;
  }
  return output;
}

static Tensor apply_bag_size_backward(
    const int64_t mode,
    Tensor &output,
    const Tensor &offset2bag,
    const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                           .unsqueeze(1)
                           .index_select(0, offset2bag);
    output *= inv_bag_size_;
  }
  return output;
}

template <typename scalar_t>
void embedding_bag_cpu_max_out(
    Tensor* max_indices,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    bool include_last_offset,
    Tensor& bag_size,
    int64_t padding_idx) {
  int64_t numIndices = indices.numel();
  int64_t featureSize = weight.size(1);
  int64_t vocab_size = weight.size(0);
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max_out", [&] {
    auto* indices_data = indices.data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    index_t* max_indices_data = nullptr;
    int64_t max_indices_stride = 0;
    if (max_indices) {
      max_indices_data = max_indices->data_ptr<index_t>();
      max_indices_stride = max_indices->strides()[0];
    }

    auto* weight_data = weight.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    auto* bag_size_data = bag_size.data_ptr<index_t>();
    auto weight_stride0 = weight.strides()[0];
    auto weight_stride1 = weight.strides()[1];
    auto output_stride = output.strides()[0];
    int64_t numBags = bag_size.size(0);
    std::vector<bool> bag_empty(numBags, true);

    for (const auto i : c10::irange(numIndices)) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];
      TORCH_CHECK(
          word_idx >= 0 && word_idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          word_idx);
      if (word_idx != static_cast<index_t>(padding_idx)) {
        bool is_first_for_bag = bag_empty[bag];
        for (const auto dim : c10::irange(featureSize)) {
          auto& current_item = output_data[output_stride * bag + dim];
          auto weight_item =
              weight_data[weight_stride0 * word_idx + dim * weight_stride1];

          if (is_first_for_bag || (weight_item > current_item)) {
            current_item = weight_item;
            if (max_indices_data) {
              max_indices_data[max_indices_stride * bag + dim] = word_idx;
            }
          }
        }
        if (is_first_for_bag) {
          bag_empty[bag] = false;
        }
      } else {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[bag]--;
      }
    }
  });
}

void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
                            Tensor& bag_size, Tensor* max_indices,
                            const Tensor &weight, const Tensor &indices,
                            const Tensor &offsets, const int64_t mode,
                            const c10::optional<Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx, _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  if (mode == MODE_MEAN || mode == MODE_SUM) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "embedding_bag_no_grad_cpu_out",
      [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx, &fbgemm_kernel_cache]() {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_no_grad_cpu_out",
        [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx, &fbgemm_kernel_cache]() {
        if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
          TORCH_INTERNAL_ASSERT(mode == MODE_SUM);
          index_select_scale_add<scalar_t, index_t>(
            indices, offset2bag, per_sample_weights.value(), weight, output, offsets, include_last_offset, bag_size, padding_idx, fbgemm_kernel_cache);
        } else {
          index_select_add<scalar_t, index_t>(indices, offset2bag, weight, output, offsets, include_last_offset, bag_size, padding_idx, fbgemm_kernel_cache);
        }
      });
    });
    apply_bag_size(mode, output, bag_size);
    if (mode == MODE_SUM) {
      // make bag_size output deterministic
      at::native::zero_(bag_size);
    }
    if (max_indices) {
      max_indices->copy_(bag_size);
    }
  } else { // MODE_MAX
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        weight.scalar_type(),
        "embedding_bag_cpu_max_out",
        [&]() {
          embedding_bag_cpu_max_out<scalar_t>(
              max_indices,
              weight,
              indices,
              offset2bag,
              output,
              include_last_offset,
              bag_size,
              padding_idx);
        });
  }
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_cpu_impl(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const int64_t mode,
    const Tensor& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    bool requires_grad) {
  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  check_arguments(weight, indices, offsets, mode, per_sample_weights, include_last_offset);

  Tensor output = at::empty(
      {include_last_offset ? offsets.size(0) - 1 : offsets.size(0),
       weight.sizes()[1]},
      weight.options());

  Tensor offset2bag = make_offset2bag(output, weight, indices, offsets, mode, per_sample_weights, padding_idx);

  Tensor bag_size = make_bag_size(offsets, indices, mode, include_last_offset, requires_grad);

  Tensor max_indices = make_max_indices(weight, indices, offsets, bag_size, mode, include_last_offset);

  _embedding_bag_cpu_impl_out(output, offset2bag,
                          bag_size, &max_indices,
                          weight, indices, offsets,
                          mode, per_sample_weights,
                          include_last_offset, padding_idx);

  return std::make_tuple(std::move(output), std::move(offset2bag), std::move(bag_size), std::move(max_indices));
}

// embedding_bag wrapper to enforce contiguity in tensors other than `weight`.
// This is created to save extra `.contiguous()` call in backward.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt,
              bool include_last_offset, c10::optional<int64_t> padding_idx_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  int64_t padding_idx = -1;

  if (padding_idx_opt.has_value()) {
    auto num_embeddings = weight.size(0);
    padding_idx = padding_idx_opt.value();
    TORCH_CHECK(
      (padding_idx >= -num_embeddings) && (padding_idx < num_embeddings),
      "padding_idx must be within the number of embeddings, -", num_embeddings,
      " through ", num_embeddings - 1, ", but got ", padding_idx);
    padding_idx = maybe_wrap_dim(padding_idx, weight.size(0));
  }
  std::tuple<Tensor, Tensor, Tensor, Tensor> out;
  if (!weight.requires_grad() && !weight._fw_grad(/*level=*/0).defined()) {
    out = at::_embedding_bag_forward_only(
      weight, indices.contiguous(), offsets.contiguous(), scale_grad_by_freq,
      mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  } else {
    out = at::_embedding_bag(
      weight, indices.contiguous(), offsets.contiguous(), scale_grad_by_freq,
      mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  }
  return out;
};

std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt,
              bool include_last_offset) {
  return at::native::embedding_bag(weight, indices, offsets, scale_grad_by_freq,
      mode, sparse, per_sample_weights_opt, include_last_offset, c10::nullopt);
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_forward_only_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt, bool include_last_offset,
                  int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;
  return _embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      padding_idx,
      /*requires_grad=*/false);
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt, bool include_last_offset,
                  int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;
  return _embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      padding_idx,
      /*requires_grad=*/true);
}

void _embedding_bag_cpu_out(
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor* p_max_indices,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const bool /* scale_grad_by_freq */,
    const int64_t mode,
    const bool /* sparse */,
    const c10::optional<at::Tensor>& per_sample_weights,
    const bool include_last_offset,
    const c10::optional<int64_t>& padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  at::native::check_arguments(
      weight, indices, offsets, mode, per_sample_weights, include_last_offset);

  at::native::make_offset2bag_out(
      offset2bag,
      output,
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      padding_idx.value_or(-1));

  at::native::make_bag_size_out(
      bag_size, offsets, indices, mode, include_last_offset, false);

  if (p_max_indices) {
    at::native::make_max_indices_out(
        *p_max_indices,
        weight,
        indices,
        offsets,
        bag_size,
        mode,
        include_last_offset);
  }

  at::native::_embedding_bag_cpu_impl_out(
      output,
      offset2bag,
      bag_size,
      p_max_indices,
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      padding_idx.value_or(-1),
      fbgemm_kernel_cache);
}

Tensor _embedding_bag_backward(const Tensor &grad, const Tensor &indices_,
                              const Tensor &offsets_,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse, const c10::optional<Tensor>& per_sample_weights_opt,
                              int64_t padding_idx) {
    return at::native::_embedding_bag_backward_symint(
        grad, indices_, offsets_, offset2bag, bag_size_, max_indices_, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights_opt, padding_idx);
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
Tensor _embedding_bag_backward_symint(const Tensor &grad, const Tensor &indices_,
                              const Tensor &offsets_,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              c10::SymInt num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse, const c10::optional<Tensor>& per_sample_weights_opt,
                              int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  checkContiguous("embedding_bag", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  checkContiguous("embedding_bag", offsets_arg);

  Tensor offset2bag_;
  if (indices.sym_numel() != 0 && offset2bag.sym_numel() == 0) {
    offset2bag_ = offsets.new_zeros(
      {indices.size(0) + 1}, offsets.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag_);
    // For Composite Compliance, if `offset2bag_` is CCT
    // then we can't call `resize_`. Instead we call `narrow`
    // to slice the tensor.
    if (isTensorSubclassLike(offset2bag_)) {
      offset2bag_ = offset2bag_.narrow(0, 0, indices.size(0));
    } else {
      offset2bag_.resize_({indices.size(0)});
    }
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  if (sparse) {
    return at::_embedding_bag_sparse_backward_symint(
        grad, indices, offsets, offset2bag_, bag_size_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  } else {
    return at::_embedding_bag_dense_backward_symint(
        grad, indices, offset2bag_, bag_size_, max_indices_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  }
}

static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  AT_ASSERT(max_indices.defined());
  auto index_grad_weight =
      at::zeros({num_weights, grad.sizes()[1]}, grad.options());
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  for (const auto dim : c10::irange(grad.sizes()[1])) {
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  return index_grad_weight;
}

template<typename index_t>
static std::vector<index_t> compute_counts(
    int64_t num_weights,
    index_t* indices_data,
    int64_t indices_length) {
  std::vector<index_t> counts(num_weights, 0);
  for (const auto i : c10::irange(indices_length)) {
    counts[indices_data[i]]++;
  }
  return counts;
}

// counts_uniq stores the index of the NEXT unique element
// of the (sorted) indices vector.
//
// For example:
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [3, 4, 6, 7]
//
// The unique indices can be found at index 0, 3, 4, 6.
template<typename index_t>
static std::vector<index_t> compute_counts_uniq(
    int64_t num_weights,
    index_t* indices_data,
    int64_t indices_length,
    const std::vector<index_t>& counts) {
  std::vector<index_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  for (int64_t i = 0; i < indices_length; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }
  return counts_uniq;
}

template <typename scalar_t>
void _embedding_bag_dense_backward_cpu_sum_mean(
    const Tensor& grad,
    const Tensor& indices_,
    const Tensor& offset2bag__,
    const Tensor& bag_size_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_,
    Tensor& index_grad_weight,
    int64_t padding_idx) {

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  optional<Tensor> per_sample_weights;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  scalar_t* per_sample_weights_data;
  optional<int64_t> per_sample_weights_stride;
  if (per_sample_weights_.defined()) {
    per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
    per_sample_weights_data = per_sample_weights->data_ptr<scalar_t>();
    per_sample_weights_stride = per_sample_weights->strides()[0];
  }

  int64_t numel = indices.numel();

  // explicitly capture all required variables to work around windows build
  // TODO: fix this when windows can correctly capture variables in nested lambda
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_dense_backward_cpu_sum_mean",
    [&indices, &offset2bag, &bag_size_, &num_weights, &numel, &per_sample_weights,
      &per_sample_weights_data, &per_sample_weights_stride, &mode, &scale_grad_by_freq,
      &grad, &index_grad_weight, &padding_idx] {
    auto* indices_data = indices.data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();
    auto* bag_size_data = bag_size_.data_ptr<index_t>();

    auto counts = compute_counts(num_weights, indices_data, numel);
    auto next_unique_index_idx =
        compute_counts_uniq(num_weights, indices_data, numel, counts);

    auto loop =
      [&next_unique_index_idx, &indices_data, &offset2bag_data, &bag_size_data, &per_sample_weights,
        &mode, &per_sample_weights_data, &per_sample_weights_stride, &scale_grad_by_freq,
        &counts, &grad, &index_grad_weight, &padding_idx
      ](index_t start, index_t end) {
      for (index_t i = start; i < end; i++) {
        index_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
        index_t index = indices_data[start];

        if (index != static_cast<index_t>(padding_idx)) {
          for (index_t j = start; j < next_unique_index_idx[i]; j++) {
            index_t source = offset2bag_data[j];
            double scale = 1.0;
            if (per_sample_weights) {
              AT_ASSERT(mode == MODE_SUM);
              scale = per_sample_weights_data[*per_sample_weights_stride * j];
            }
            if (scale_grad_by_freq) {
              scale /= counts[indices_data[i]];
            }
            if (mode == MODE_MEAN) {
              auto bag_size = bag_size_data[source];
              if (bag_size != 0) {
                scale /= bag_size;
              }
            }
            int64_t ddim = grad.size(1);
            auto igwd = index_grad_weight.data_ptr<scalar_t>();
            auto gd = grad.data_ptr<scalar_t>();
            at::native::cpublas::axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                        igwd + ddim * index, 1);
          }
        }
      }
    };

    if (numel > 1000) {
      at::parallel_for(0, (int64_t)next_unique_index_idx.size(), 0, loop);
    } else {
      loop(0, (int64_t)next_unique_index_idx.size());
    }
  });
}

Tensor _embedding_bag_dense_backward_cpu(const Tensor &grad_, const Tensor &indices_,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor>& per_sample_weights__opt,
                                  int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights__maybe_owned = at::borrow_from_optional_tensor(per_sample_weights__opt);
  const Tensor& per_sample_weights_ = *per_sample_weights__maybe_owned;

  // indices_, offsets_ and offset2bag__ are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.
  auto grad = grad_.contiguous();
  auto grad_arg = TensorArg(grad, "grad_", 1);
  checkScalarTypes(
      "embedding_bag", grad_arg, {kHalf, kBFloat16, kFloat, kDouble});

  if (mode == MODE_MAX) {
    return _embedding_bag_dense_backward_cpu_max(
        grad_, bag_size_, max_indices_, num_weights);
  }
  AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

  auto index_grad_weight =
      at::zeros({num_weights, grad.sizes()[1]}, grad.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_backward",
      [&] {
        _embedding_bag_dense_backward_cpu_sum_mean<scalar_t>(
            grad,
            indices_,
            offset2bag__,
            bag_size_,
            num_weights,
            scale_grad_by_freq,
            mode,
            per_sample_weights_,
            index_grad_weight,
            padding_idx);
      });
  return index_grad_weight;
}

template<typename scalar_t>
Tensor _embedding_bag_per_sample_weights_backward_cpu_template(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.sizes()[1];

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.sizes()[1] == embedding_features);

  auto output = at::zeros({num_samples}, grad.options());

  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  checkContiguous("embedding_bag", indices_arg);

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.size(0) + 1}, offset2bag.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag_);

    at::native::resize_(offset2bag_, {indices.size(0)}, c10::nullopt);
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  auto* grad_data = grad.data_ptr<scalar_t>();
  auto grad_stride0 = grad.strides()[0];
  auto grad_stride1 = grad.strides()[1];

  auto* weight_data = weight.data_ptr<scalar_t>();
  auto weight_stride0 = weight.strides()[0];
  auto weight_stride1 = weight.strides()[1];

  // explicitly capture all required variables to work around windows build
  // TODO: fix this when windows can correctly capture variables in nested lambda
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu_template",
    [&indices, &output, &offset2bag_, &num_samples, &embedding_features,
      &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0, &weight_stride1,
      &padding_idx] () {
    auto* indices_data = indices.data_ptr<index_t>();

    // The following are contiguous
    auto* output_data = output.data_ptr<scalar_t>();
    auto* offset2bag_data = offset2bag_.data_ptr<index_t>();

    // XXX: 64 was arbitrarily chosen. There is probably a sweet spot for this number.
    parallel_for(0, num_samples, 64,
      [&embedding_features, &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0,
        &weight_stride1, &offset2bag_data, &indices_data, &output_data, &padding_idx](index_t begin, index_t end) {
      for (index_t sample_idx = begin; sample_idx < end; sample_idx++) {
        auto bag_idx = offset2bag_data[sample_idx];
        auto embedding_idx = indices_data[sample_idx];

        if (embedding_idx != static_cast<index_t>(padding_idx)) {
          output_data[sample_idx] = dot_impl<scalar_t>(
              embedding_features,
              grad_data + grad_stride0 * bag_idx, grad_stride1,
              weight_data + weight_stride0 * embedding_idx, weight_stride1);
        }
      }
    });
  });
  return output;
}

Tensor _embedding_bag_per_sample_weights_backward_cpu(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "_embedding_bag_per_sample_weights_backward_cpu",
      [&]() {
        return _embedding_bag_per_sample_weights_backward_cpu_template<
            scalar_t>(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx);
      });
}

Tensor _embedding_bag_sparse_backward(
    const Tensor &grad_, const Tensor &indices, const Tensor &offsets,
    const Tensor &offset2bag, const Tensor &bag_size_, SymInt num_weights,
    bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
    return at::native::_embedding_bag_sparse_backward_symint(grad_, indices, offsets, offset2bag, bag_size_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights_opt, padding_idx);
}

Tensor _embedding_bag_sparse_backward_symint(
    const Tensor &grad_, const Tensor &indices, const Tensor &offsets,
    const Tensor &offset2bag, const Tensor &bag_size_, SymInt num_weights,
    bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Tensor grad = grad_;
  Tensor index_grad = grad_.index_select(0, offset2bag);

  index_grad = apply_bag_size_backward(mode, index_grad, offset2bag, bag_size_);

  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == MODE_SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }
  return native::embedding_backward_symint(index_grad, indices, num_weights, padding_idx,
                                    scale_grad_by_freq, true);
}
}
} // namespace at::native
