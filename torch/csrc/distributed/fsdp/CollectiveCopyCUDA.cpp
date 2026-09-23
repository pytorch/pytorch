#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.hpp>
#include <torch/csrc/distributed/fsdp/CollectiveCopyCUDA.hpp>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_chunk_cat.h>
#endif

namespace c10d::fsdp {
namespace {

struct ChunkCatMetadata {
  int64_t chunk_size = 0;
  int64_t leading_dim = 1;
  int64_t num_blocks_per_chunk = 0;
  int64_t slice_size = 0;
  std::vector<int64_t> srcs;
  std::vector<int64_t> block_idx_to_tensor_idx;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes;
  std::vector<int64_t> start_block_idx_per_tensor_chunk;
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
};

void all_gather_copy_out_cuda(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks) {
  const bool needs_resize = check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks);
  if (!input.is_contiguous()) {
    all_gather_copy_out(out, input, split_sizes, outer_sizes, num_chunks);
    return;
  }
  if (needs_resize) {
    if (std::all_of(
            outer_sizes.begin(), outer_sizes.end(), [](int64_t outer_size) {
              return outer_size == 1;
            })) {
      all_gather_copy_out(out, input, split_sizes, outer_sizes, num_chunks);
      return;
    }
    std::vector<std::pair<uintptr_t, uintptr_t>> output_ranges;
    output_ranges.reserve(out.size());
    bool use_generic = false;
    for (const auto& tensor : out) {
      use_generic |= tensor.is_conj() || tensor.is_neg();
      if (tensor.numel() != 0) {
        const auto begin = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
        output_ranges.emplace_back(begin, begin + tensor.nbytes());
      }
    }
    std::sort(output_ranges.begin(), output_ranges.end());
    for (const auto i : c10::irange(1, output_ranges.size())) {
      use_generic |= output_ranges[i].first < output_ranges[i - 1].second;
    }
    if (use_generic) {
      all_gather_copy_out(out, input, split_sizes, outer_sizes, num_chunks);
      return;
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto buffers = split_all_gather_output_with_resize(
        out, input, split_sizes, outer_sizes, num_chunks);
    std::vector<detail::AllGatherReassembly> copies;
    copies.reserve(out.size());
    bool use32 = true;
    for (const auto i : c10::irange(out.size())) {
      if (outer_sizes[i] == 1 || split_sizes[i] == 0 || out[i].numel() == 0) {
        continue;
      }
      const auto rank_size = static_cast<int64_t>(out[i].nbytes() / num_chunks);
      copies.push_back(
          {buffers[i].const_data_ptr(),
           out[i].mutable_data_ptr(),
           rank_size,
           outer_sizes[i]});
      use32 &= out[i].nbytes() <= std::numeric_limits<int32_t>::max() &&
          at::native::canUse32BitIndexMath(out[i]) &&
          at::native::canUse32BitIndexMath(buffers[i]);
    }
    detail::launch_all_gather_reassembly(copies, num_chunks, use32);
    for (const auto& tensor : out) {
      if (!tensor.is_inference()) {
        torch::autograd::impl::bump_version(tensor);
      }
    }
    return;
  }
  const c10::cuda::CUDAGuard device_guard(input.device());
  std::vector<int64_t> srcs;
  std::vector<int64_t> dsts;
  std::vector<int64_t> chunk_sizes;
  int64_t num_splits = 0;
  for (const auto i : c10::irange(split_sizes.size())) {
    if (split_sizes[i] > 0) {
      num_splits += outer_sizes[i];
    }
  }
  srcs.reserve(num_splits);
  dsts.reserve(num_splits);
  chunk_sizes.reserve(num_splits);
  auto src = reinterpret_cast<int64_t>(input.const_data_ptr());
  const auto elem_size = input.element_size();
  for (const auto i : c10::irange(out.size())) {
    if (split_sizes[i] == 0) {
      continue;
    }
    const int64_t chunk_size = split_sizes[i] / outer_sizes[i] * elem_size;
    const auto dst = reinterpret_cast<int64_t>(out[i].data_ptr());
    for (const auto outer_idx : c10::irange(outer_sizes[i])) {
      srcs.push_back(src + outer_idx * chunk_size);
      dsts.push_back(dst + outer_idx * num_chunks * chunk_size);
      chunk_sizes.push_back(chunk_size);
    }
    src += split_sizes[i] * elem_size;
  }
  if (!srcs.empty()) {
    const auto bytes_per_block = detail::kChunkCatBytesPerBlock;
    int64_t num_blocks = 0;
    for (const auto chunk_size : chunk_sizes) {
      num_blocks += (chunk_size + bytes_per_block - 1) / bytes_per_block;
    }
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const int64_t max_blocks =
        static_cast<int64_t>(properties->multiProcessorCount) *
        properties->maxThreadsPerMultiProcessor / detail::kCopyThreadsPerBlock *
        2;
    const int64_t iter_factor =
        (num_blocks * num_chunks + max_blocks - 1) / max_blocks;
    int64_t chunks_per_block = std::ceil(std::sqrt(iter_factor));
    chunks_per_block = std::min(chunks_per_block, num_chunks);
    const int64_t iters_per_chunk =
        (iter_factor + chunks_per_block - 1) / chunks_per_block;
    std::vector<int64_t> block_idx_to_split_idx;
    std::vector<int64_t> blocks_cumsums{0};
    block_idx_to_split_idx.reserve(num_blocks);
    const int64_t bytes_per_block_iter = bytes_per_block * iters_per_chunk;
    for (const auto i : c10::irange(chunk_sizes.size())) {
      const auto blocks =
          (chunk_sizes[i] + bytes_per_block_iter - 1) / bytes_per_block_iter;
      block_idx_to_split_idx.insert(
          block_idx_to_split_idx.end(), blocks, static_cast<int64_t>(i));
      blocks_cumsums.push_back(blocks_cumsums.back() + blocks);
    }
    auto packed = detail::pack_vecs(
        {&dsts, &srcs, &chunk_sizes, &block_idx_to_split_idx, &blocks_cumsums},
        input.device());
    detail::launch_split_with_sizes_copy(
        packed.second,
        blocks_cumsums.back(),
        num_chunks / chunks_per_block,
        input.numel() / num_chunks * elem_size,
        num_chunks);
  }
  for (const auto& tensor : out) {
    if (!tensor.is_inference()) {
      torch::autograd::impl::bump_version(tensor);
    }
  }
}

at::Tensor& reduce_scatter_copy_in_cuda(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  check_reduce_scatter_copy_in_inputs(
      out, tensors, num_leading_dims, num_chunks);
  if (std::all_of(
          num_leading_dims.begin(), num_leading_dims.end(), [](int64_t dim) {
            return dim == 0;
          })) {
    return at::_chunk_cat_out(out, tensors, 0, num_chunks);
  }
  const c10::cuda::CUDAGuard device_guard(out.device());
  const auto src_dtype = tensors[0].scalar_type();
  bool fast_path = out.is_contiguous() &&
      (src_dtype == out.scalar_type() ||
       (src_dtype == at::kBFloat16 && out.scalar_type() == at::kFloat));
  for (const auto& tensor : tensors) {
    fast_path &= tensor.is_contiguous();
  }
  if (!fast_path) {
    return reduce_scatter_copy_in(out, tensors, num_leading_dims, num_chunks);
  }

  int64_t num_inputs = 0;
  for (const auto i : c10::irange(tensors.size())) {
    num_inputs += c10::multiply_integers(
        tensors[i].sizes().slice(0, num_leading_dims[i]));
  }
  ChunkCatMetadata metadata;
  metadata.srcs.reserve(num_inputs);
  metadata.pad_tensor_chunk_sizes.reserve(num_inputs);
  metadata.num_blocks_per_tensor_chunk.reserve(num_inputs);
  metadata.start_block_idx_per_tensor_chunk.reserve(num_inputs + 1);
  metadata.actual_tensor_sizes.reserve(num_inputs);
  metadata.tensor_idx_to_start_tensor_bytes.reserve(num_inputs + 1);
  metadata.start_block_idx_per_tensor_chunk.push_back(0);
  metadata.tensor_idx_to_start_tensor_bytes.push_back(0);
  const auto src_elem_size = tensors[0].element_size();
  const auto dst_elem_size = out.element_size();
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto sizes = tensor.sizes();
    const auto dim = num_leading_dims[i];
    const auto outer_size = c10::multiply_integers(sizes.slice(0, dim));
    if (outer_size == 0) {
      continue;
    }
    const auto trailing_numel = c10::multiply_integers(sizes.slice(dim + 1));
    const int64_t chunk_size = (sizes[dim] + num_chunks - 1) / num_chunks *
        trailing_numel * dst_elem_size;
    const auto bytes_per_block = detail::kChunkCatBytesPerBlock;
    const int64_t num_blocks =
        (chunk_size + bytes_per_block - 1) / bytes_per_block;
    const int64_t actual_size = sizes[dim] * trailing_numel * src_elem_size;
    const auto src = reinterpret_cast<int64_t>(tensor.const_data_ptr());
    for (const auto outer_idx : c10::irange(outer_size)) {
      const auto input_idx = static_cast<int64_t>(metadata.srcs.size());
      metadata.srcs.push_back(src + outer_idx * actual_size);
      metadata.pad_tensor_chunk_sizes.push_back(chunk_size);
      metadata.chunk_size += chunk_size;
      metadata.num_blocks_per_tensor_chunk.push_back(num_blocks);
      metadata.start_block_idx_per_tensor_chunk.push_back(
          metadata.start_block_idx_per_tensor_chunk.back() + num_blocks);
      metadata.block_idx_to_tensor_idx.insert(
          metadata.block_idx_to_tensor_idx.end(), num_blocks, input_idx);
      metadata.tensor_idx_to_start_tensor_bytes.push_back(
          metadata.tensor_idx_to_start_tensor_bytes.back() + chunk_size);
      metadata.actual_tensor_sizes.push_back(actual_size);
    }
  }
  metadata.num_blocks_per_chunk =
      metadata.start_block_idx_per_tensor_chunk.back();
  metadata.slice_size = num_chunks * metadata.chunk_size;
  at::native::resize_output(
      out, {num_chunks, metadata.chunk_size / dst_elem_size});
  auto packed = detail::pack_vecs(
      {&metadata.srcs,
       &metadata.block_idx_to_tensor_idx,
       &metadata.tensor_idx_to_start_tensor_bytes,
       &metadata.start_block_idx_per_tensor_chunk,
       &metadata.actual_tensor_sizes,
       &metadata.pad_tensor_chunk_sizes,
       &metadata.num_blocks_per_tensor_chunk},
      out.device());
  detail::launch_chunk_cat(
      out,
      packed.second,
      metadata.num_blocks_per_chunk,
      num_chunks,
      metadata.leading_dim,
      metadata.slice_size,
      metadata.chunk_size,
      src_dtype);
  if (!out.is_inference()) {
    torch::autograd::impl::bump_version(out);
  }
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(fsdp, CUDA, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out_cuda));
  m.impl("_reduce_scatter_copy_in_", TORCH_FN(reduce_scatter_copy_in_cuda));
}

} // namespace c10d::fsdp
