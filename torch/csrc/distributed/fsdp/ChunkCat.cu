#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/fsdp/ChunkCat.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.h>
#include <torch/library.h>
#include <ATen/native/cuda/ChunkCat.cuh>

#include <algorithm>
#include <cmath>

namespace torch::distributed::fsdp {
namespace {

using at::native::detail::BLOCK_SIZE;
using at::native::detail::BYTES_PER_BLOCK;
using at::native::detail::chunk_cat_cuda_kernel;
using at::native::detail::div_up;
using at::native::detail::pack_vecs;
using at::native::detail::split_with_sizes_copy_out_contiguous_no_cast_kernel;

template <typename dst_t, typename src_t>
void launch_chunk_cat(
    const std::vector<int64_t*>& ptrs,
    int64_t* block_idx_to_tensor_idx,
    int64_t num_blocks,
    int64_t num_chunks,
    int64_t chunk_size,
    at::Tensor& out) {
  chunk_cat_cuda_kernel<dst_t, src_t>
      <<<dim3(num_blocks, num_chunks, 1),
         dim3(BLOCK_SIZE, 1, 1),
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<src_t**>(ptrs[0]),
          static_cast<dst_t*>(out.mutable_data_ptr()),
          block_idx_to_tensor_idx,
          ptrs[2],
          ptrs[3],
          ptrs[4],
          ptrs[5],
          ptrs[6],
          /*slice_size=*/num_chunks * chunk_size,
          chunk_size,
          /*dst_to_src_ratio=*/sizeof(dst_t) / sizeof(src_t));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Inputs already in out's dtype are copied by one launch, and cast_dtype inputs
// into an fp32 out are cast by a second one. The output layout doesn't depend
// on input dtypes, so both launches share one metadata upload and differ only
// in their block_idx_to_tensor_idx ranges.
// start_block_idx_per_tensor_chunk is relative to each tensor's launch.
// Each index into the first num_leading_dims[i] dims of tensors[i] is a
// separate _chunk_cat input. Empty num_leading_dims means all zeros.
void fused_chunk_cat(
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks,
    at::ScalarType cast_dtype,
    at::Tensor& out) {
  c10::cuda::CUDAGuard device_guard(out.device());
  const auto out_dtype = out.scalar_type();
  const auto num_tensors = tensors.size();
  std::vector<int64_t> srcs;
  std::vector<int64_t> copy_blocks;
  std::vector<int64_t> cast_blocks;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes;
  std::vector<int64_t> start_block_idx_per_tensor_chunk;
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
  srcs.reserve(num_tensors);
  tensor_idx_to_start_tensor_bytes.reserve(num_tensors);
  start_block_idx_per_tensor_chunk.reserve(num_tensors);
  actual_tensor_sizes.reserve(num_tensors);
  pad_tensor_chunk_sizes.reserve(num_tensors);
  num_blocks_per_tensor_chunk.reserve(num_tensors);
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    const at::Tensor& tensor = tensors[i];
    const auto sizes = tensor.sizes();
    const int64_t dim = num_leading_dims.empty() ? 0 : num_leading_dims[i];
    const int64_t row_numel = c10::multiply_integers(sizes.slice(dim + 1));
    const int64_t pad_tensor_chunk_size =
        div_up(sizes[dim], num_chunks) * row_numel * out.element_size();
    const int64_t actual_tensor_size =
        sizes[dim] * row_numel * tensor.element_size();
    const int64_t num_blocks = div_up(pad_tensor_chunk_size, BYTES_PER_BLOCK);
    auto& blocks =
        tensor.scalar_type() == out_dtype ? copy_blocks : cast_blocks;
    // The cast launch below reads cast_dtype and writes fp32
    TORCH_INTERNAL_ASSERT(
        tensor.scalar_type() == out_dtype ||
        (tensor.scalar_type() == cast_dtype && out_dtype == at::kFloat));
    const auto src = reinterpret_cast<int64_t>(tensor.const_data_ptr());
    const int64_t num_slices = c10::multiply_integers(sizes.slice(0, dim));
    for (const auto slice : c10::irange(num_slices)) {
      srcs.push_back(src + slice * actual_tensor_size);
      tensor_idx_to_start_tensor_bytes.push_back(chunk_size);
      start_block_idx_per_tensor_chunk.push_back(
          static_cast<int64_t>(blocks.size()));
      blocks.insert(
          blocks.end(), num_blocks, static_cast<int64_t>(srcs.size()) - 1);
      actual_tensor_sizes.push_back(actual_tensor_size);
      pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
      num_blocks_per_tensor_chunk.push_back(num_blocks);
      chunk_size += pad_tensor_chunk_size;
    }
  }
  const auto num_copy_blocks = static_cast<int64_t>(copy_blocks.size());
  const auto num_cast_blocks = static_cast<int64_t>(cast_blocks.size());
  auto& block_idx_to_tensor_idx = copy_blocks;
  block_idx_to_tensor_idx.insert(
      block_idx_to_tensor_idx.end(), cast_blocks.begin(), cast_blocks.end());
  at::native::resize_output(out, {num_chunks, chunk_size / out.element_size()});
  // `packed` keeps the device metadata alive until the kernels are enqueued.
  // ROCm clang 18 rejects capturing a structured binding in AT_DISPATCH below.
  auto packed = pack_vecs(
      {&srcs,
       &block_idx_to_tensor_idx,
       &tensor_idx_to_start_tensor_bytes,
       &start_block_idx_per_tensor_chunk,
       &actual_tensor_sizes,
       &pad_tensor_chunk_sizes,
       &num_blocks_per_tensor_chunk},
      out.device());
  const auto& ptrs = packed.second;
  if (num_copy_blocks > 0) {
    launch_chunk_cat<char, char>(
        ptrs, ptrs[1], num_copy_blocks, num_chunks, chunk_size, out);
  }
  if (num_cast_blocks > 0) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(cast_dtype, "fused_chunk_cat", [&] {
      launch_chunk_cat<float, scalar_t>(
          ptrs,
          ptrs[1] + num_copy_blocks,
          num_cast_blocks,
          num_chunks,
          chunk_size,
          out);
    });
  }
}

// The fused kernels copy raw bytes of contiguous tensors on one device, so
// other tensors, conj or neg views, and overlapping memory take the composite.
bool use_composite(at::TensorList tensors, const at::Tensor& other) {
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  ranges.reserve(tensors.size() + 1);
  const auto add = [&](const at::Tensor& t) {
    if (!t.is_cuda() || t.device() != other.device() || !t.is_contiguous() ||
        t.is_conj() || t.is_neg()) {
      return false;
    }
    if (t.numel() > 0) {
      const auto begin = reinterpret_cast<uintptr_t>(t.const_data_ptr());
      ranges.emplace_back(begin, begin + t.nbytes());
    }
    return true;
  };
  if (!add(other) || !std::all_of(tensors.begin(), tensors.end(), add)) {
    return true;
  }
  std::sort(ranges.begin(), ranges.end());
  return std::adjacent_find(
             ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
               return b.first < a.second;
             }) != ranges.end();
}

void all_gather_copy_out_cuda(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks) {
  if (use_composite(out, input)) {
    all_gather_copy_out(out, input, split_sizes, outer_sizes, num_chunks);
    return;
  }
  check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks);
  c10::cuda::CUDAGuard device_guard(input.device());
  std::vector<int64_t> srcs;
  std::vector<int64_t> dsts;
  std::vector<int64_t> chunk_sizes;
  auto src = reinterpret_cast<int64_t>(input.const_data_ptr());
  const auto elem_size = input.element_size();
  for (const auto i : c10::irange(out.size())) {
    const auto split_size = split_sizes[i].expect_int();
    if (split_size == 0) {
      continue;
    }
    const auto outer_size = outer_sizes[i].expect_int();
    const int64_t chunk_size = split_size / outer_size * elem_size;
    const auto dst = reinterpret_cast<int64_t>(out[i].mutable_data_ptr());
    for (const auto outer_idx : c10::irange(outer_size)) {
      srcs.push_back(src + outer_idx * chunk_size);
      dsts.push_back(dst + outer_idx * num_chunks * chunk_size);
      chunk_sizes.push_back(chunk_size);
    }
    src += split_size * elem_size;
  }
  if (srcs.empty()) {
    return;
  }
  int64_t num_blocks = 0;
  for (const auto chunk_size : chunk_sizes) {
    num_blocks += div_up(chunk_size, BYTES_PER_BLOCK);
  }
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int64_t max_blocks =
      static_cast<int64_t>(properties->multiProcessorCount) *
      properties->maxThreadsPerMultiProcessor / BLOCK_SIZE * 2;
  const int64_t iter_factor = div_up(num_blocks * num_chunks, max_blocks);
  int64_t chunks_per_block = std::ceil(std::sqrt(iter_factor));
  chunks_per_block = std::min(chunks_per_block, num_chunks);
  const int64_t iters_per_chunk = div_up(iter_factor, chunks_per_block);
  std::vector<int64_t> block_idx_to_split_idx;
  std::vector<int64_t> blocks_cumsums{0};
  block_idx_to_split_idx.reserve(num_blocks);
  for (const auto i : c10::irange(chunk_sizes.size())) {
    const auto blocks =
        div_up(chunk_sizes[i], BYTES_PER_BLOCK * iters_per_chunk);
    block_idx_to_split_idx.insert(
        block_idx_to_split_idx.end(), blocks, static_cast<int64_t>(i));
    blocks_cumsums.push_back(blocks_cumsums.back() + blocks);
  }
  // `packed` keeps the device metadata alive until the kernel is enqueued.
  auto [packed, ptrs] = pack_vecs(
      {&dsts, &srcs, &chunk_sizes, &block_idx_to_split_idx, &blocks_cumsums},
      input.device());
  split_with_sizes_copy_out_contiguous_no_cast_kernel<<<
      dim3(blocks_cumsums.back(), num_chunks / chunks_per_block, 1),
      dim3(BLOCK_SIZE, 1, 1),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<char**>(ptrs[0]),
      reinterpret_cast<char**>(ptrs[1]),
      ptrs[2],
      ptrs[3],
      ptrs[4],
      /*src_stride=*/input.numel() / num_chunks * elem_size,
      num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Uniform groups without leading dims take the composite, which forwards them
// to _chunk_cat. So do dim != 0, which FSDP doesn't use, and dtypes the fused
// kernel can't cast.
void chunk_cat_mixed_dtype_cuda(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef num_leading_dims) {
  const bool leading_dims = has_leading_dims(num_leading_dims);
  const auto out_dtype = out.scalar_type();
  const bool mixed_dtypes =
      std::any_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) {
        return t.scalar_type() != tensors[0].scalar_type();
      });
  // The fused kernel copies inputs of out's dtype and casts at most one other
  // dtype, bf16 or fp16, into an fp32 out.
  std::optional<at::ScalarType> cast_dtype;
  const bool fused_dtypes =
      std::all_of(tensors.begin(), tensors.end(), [&](const at::Tensor& t) {
        const auto dtype = t.scalar_type();
        if (dtype == out_dtype) {
          return true;
        }
        if (out_dtype != at::kFloat ||
            (dtype != at::kBFloat16 && dtype != at::kHalf) ||
            cast_dtype.value_or(dtype) != dtype) {
          return false;
        }
        cast_dtype = dtype;
        return true;
      });
  const bool use_fused_kernel = (mixed_dtypes || leading_dims) &&
      fused_dtypes && dim == 0 && num_chunks >= 1 &&
      std::all_of(tensors.begin(),
                  tensors.end(),
                  [](const at::Tensor& t) {
                    return t.dim() > 0 && t.numel() > 0;
                  }) &&
      !use_composite(tensors, out);
  if (!use_fused_kernel) {
    chunk_cat_mixed_dtype(tensors, dim, num_chunks, out, num_leading_dims);
    return;
  }
  const auto fused_cast_dtype = cast_dtype.value_or(at::kBFloat16);
  if (!leading_dims) {
    fused_chunk_cat(tensors, {}, num_chunks, fused_cast_dtype, out);
    return;
  }
  check_chunk_cat_leading_dims_inputs(
      tensors, *num_leading_dims, num_chunks, out);
  auto chunks = out.view({num_chunks, -1});
  fused_chunk_cat(
      tensors, *num_leading_dims, num_chunks, fused_cast_dtype, chunks);
}

TORCH_LIBRARY_IMPL(fsdp, CUDA, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype_cuda));
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out_cuda));
}

} // namespace
} // namespace torch::distributed::fsdp
