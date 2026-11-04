#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/util/TypeCast.h>
#include <ATen/native/cuda/ChunkCat.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_chunk_cat_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/split_with_sizes_copy_native.h>
#endif

namespace at::native {

namespace detail {

// Calculate the base addr for each split.
static inline std::vector<int64_t> get_split_base_addrs(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto* data_ptr = static_cast<const char*>(tensor.const_data_ptr());
  const auto slice_numel = c10::multiply_integers(tensor.sizes().slice(dim + 1));
  const auto element_sz = tensor.element_size();
  int64_t off = 0;
  std::vector<int64_t> split_base_addrs;
  split_base_addrs.reserve(split_sizes.size());
  for (const auto& split_size : split_sizes) {
    split_base_addrs.push_back(reinterpret_cast<int64_t>(data_ptr + off));
    off += split_size * slice_numel * element_sz;
  }
  return split_base_addrs;
}

static inline std::vector<int64_t> get_dst_addrs(at::TensorList out) {
  std::vector<int64_t> addrs;
  addrs.reserve(out.size());
  for (const auto& tensor : out) {
    addrs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
  }
  return addrs;
}

// Calculate the chunk size for each split in bytes.
static inline std::vector<int64_t> get_split_chunk_sizes(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto slice_numel = c10::multiply_integers(tensor.sizes().slice(dim + 1));
  const auto element_sz = tensor.element_size();
  std::vector<int64_t> split_chunk_sizes;
  split_chunk_sizes.reserve(split_sizes.size());
  for (const auto& split_size : split_sizes) {
    split_chunk_sizes.push_back(split_size * slice_numel * element_sz);
  }
  return split_chunk_sizes;
}

// Calculate the chunk stride in bytes. This is the same for all splits.
static inline int64_t get_chunk_stride(const at::Tensor& tensor, int64_t dim) {
  int64_t stride = 1;
  for (int64_t d = dim; d < tensor.dim(); ++d) {
    stride *= tensor.sizes()[d];
  }
  return stride * tensor.element_size();
}

// Calculate the number of chunks. This is the same for all splits.
static inline int64_t get_num_chunks(const at::Tensor& tensor, int64_t dim) {
  int64_t num_chunks = tensor.numel();
  for (int64_t d = dim; d < tensor.dim(); ++d) {
    num_chunks /= tensor.sizes()[d];
  }
  return num_chunks;
}

static inline std::vector<int64_t> get_chunk_cat_out_sizes(
    IntArrayRef input_tensor_sizes,
    int64_t dim,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t out_element_size) {
  std::vector<int64_t> view_sizes = std::vector<int64_t>(
      input_tensor_sizes.begin(), input_tensor_sizes.begin() + dim);
  view_sizes.insert(
      view_sizes.end(), {num_chunks, chunk_size / out_element_size});
  return view_sizes;
}

bool all_contiguous(TensorList tensors) {
  for (const auto& t : tensors) {
    if (!t.is_contiguous()) {
      return false;
    }
  }
  return true;
}

// Get leading dimensions before `dim`-th dimension.
static inline int64_t get_leading_dim(at::IntArrayRef sizes, int64_t dim) {
  int64_t leading_dim = 1;
  if (dim > 0) {
    leading_dim = c10::multiply_integers(sizes.slice(0, dim));
  }
  return leading_dim;
}

// Get trailing dimensions after `dim`-th dimension and padded size along
// `dim`-th dimension.
static inline std::pair<int64_t, int64_t> get_pad_size(
    at::IntArrayRef sizes,
    int64_t dim,
    int64_t num_chunks) {
  int64_t trailing_numel = 1;
  if (sizes.size() > (uint64_t)dim + 1) {
    trailing_numel =
        c10::multiply_integers(sizes.slice(dim + 1, sizes.size() - dim - 1));
  }
  int64_t pad_size_along_dim =
      detail::div_up(sizes[dim], num_chunks) * num_chunks;
  return std::make_pair(pad_size_along_dim, trailing_numel);
}

// Get the padded chunk size.
static inline int64_t get_chunk_size(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    int64_t elem_size) {
  auto num_tensors = tensors.size();
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    auto [pad_size_along_dim, trailing_numel] =
        get_pad_size(tensors[i].sizes(), dim, num_chunks);
    const int64_t pad_tensor_chunk_size =
        pad_size_along_dim * trailing_numel * elem_size / num_chunks;
    chunk_size += pad_tensor_chunk_size;
  }
  return chunk_size;
}

// Get metadata for chunk_cat.
std::tuple<
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>>
get_chunk_cat_metadata(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    int64_t dst_elem_size,
    int64_t src_elem_size) {
  TORCH_CHECK(
      dst_elem_size % src_elem_size == 0,
      "get_chunk_cat_metadata error: only support dst_elem_size % src_elem_size == 0");
  auto num_tensors = tensors.size();
  int64_t leading_dim = get_leading_dim(tensors[0].sizes(), dim);
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_blocks_per_tensor_chunk;
  std::vector<int64_t> start_block_idx_per_tensor_chunk{0};
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes{0};
  std::vector<int64_t> srcs;
  pad_tensor_chunk_sizes.reserve(num_tensors);
  num_blocks_per_tensor_chunk.reserve(num_tensors);
  start_block_idx_per_tensor_chunk.reserve(num_tensors + 1);
  actual_tensor_sizes.reserve(num_tensors);
  tensor_idx_to_start_tensor_bytes.reserve(num_tensors + 1);
  srcs.reserve(num_tensors);
  // block_idx_to_tensor_idx cannot be reserved since the number of blocks is
  // data dependent
  std::vector<int64_t> block_idx_to_tensor_idx;
  // Inline computing `chunk_size` to avoid redundant computation
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    const at::Tensor& tensor = tensors[i];
    srcs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
    auto sizes = tensor.sizes();
    auto [pad_size_along_dim, trailing_numel] =
        get_pad_size(sizes, dim, num_chunks);
    const int64_t pad_tensor_chunk_size =
        pad_size_along_dim * trailing_numel * dst_elem_size / num_chunks;
    pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
    chunk_size += pad_tensor_chunk_size;
    // Number of blocks required to process this tensor chunk.
    const int64_t num_blocks =
        detail::div_up(pad_tensor_chunk_size, detail::BYTES_PER_BLOCK);
    num_blocks_per_tensor_chunk.push_back(num_blocks);
    start_block_idx_per_tensor_chunk.push_back(
        start_block_idx_per_tensor_chunk.back() + num_blocks);
    block_idx_to_tensor_idx.insert(
        block_idx_to_tensor_idx.end(), num_blocks, i);
    tensor_idx_to_start_tensor_bytes.push_back(
        tensor_idx_to_start_tensor_bytes.back() + pad_tensor_chunk_size);
    actual_tensor_sizes.push_back(sizes[dim] * trailing_numel * src_elem_size);
  }
  const int64_t num_blocks_per_chunk = start_block_idx_per_tensor_chunk.back();
  const int64_t slice_size = num_chunks * chunk_size;
  return std::make_tuple(
      chunk_size,
      leading_dim,
      num_blocks_per_chunk,
      slice_size,
      std::move(srcs),
      std::move(block_idx_to_tensor_idx),
      std::move(tensor_idx_to_start_tensor_bytes),
      std::move(start_block_idx_per_tensor_chunk),
      std::move(actual_tensor_sizes),
      std::move(pad_tensor_chunk_sizes),
      std::move(num_blocks_per_tensor_chunk));
}

// See NOTE [CUDA kernel for chunk_cat] in ChunkCat.cuh
template <typename dst_t, typename src_t>
void _chunk_cat_out_cuda_contiguous(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out,
    int64_t dst_elem_size,
    int64_t src_elem_size) {
  const auto device = tensors[0].device();
  // `get_chunk_cat_metadata` must return vectors and `pack_vecs` cannot be
  // moved into `get_chunk_cat_metadata`. Otherwise `packed` would point to
  // vectors allocated inside `get_chunk_cat_metadata` which become out of local
  // scope.
  auto
      [chunk_size,
       leading_dim,
       num_blocks_per_chunk,
       slice_size,
       srcs,
       block_idx_to_tensor_idx,
       tensor_idx_to_start_tensor_bytes,
       start_block_idx_per_tensor_chunk,
       actual_tensor_sizes,
       pad_tensor_chunk_sizes,
       num_blocks_per_tensor_chunk] =
          get_chunk_cat_metadata(
              tensors, dim, num_chunks, dst_elem_size, src_elem_size);
  auto packed = pack_vecs(
      {&srcs,
       &block_idx_to_tensor_idx,
       &tensor_idx_to_start_tensor_bytes,
       &start_block_idx_per_tensor_chunk,
       &actual_tensor_sizes,
       &pad_tensor_chunk_sizes,
       &num_blocks_per_tensor_chunk},
      device);
  std::vector<int64_t> view_sizes = get_chunk_cat_out_sizes(
      tensors[0].sizes(), dim, num_chunks, chunk_size, dst_elem_size);
  at::native::resize_output(out, view_sizes);
  dim3 blocks(num_blocks_per_chunk, num_chunks, leading_dim);
  dim3 threads(detail::BLOCK_SIZE, 1, 1);
  detail::chunk_cat_cuda_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      /*srcs=*/reinterpret_cast<src_t**>(packed.second[0]),
      reinterpret_cast<dst_t*>(out.data_ptr()),
      /*block_idx_to_tensor_idx=*/packed.second[1],
      /*tensor_idx_to_start_tensor_bytes=*/packed.second[2],
      /*start_block_idx_per_tensor_chunk=*/packed.second[3],
      /*actual_tensor_sizes=*/packed.second[4],
      /*pad_tensor_chunk_sizes=*/packed.second[5],
      /*num_blocks_per_tensor_chunk=*/packed.second[6],
      slice_size,
      chunk_size,
      dst_elem_size / src_elem_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail

// See NOTE [CUDA fast path for split_with_sizes_copy.out] in ChunkCat.cuh.
// Callers must first check has_dense_matching_slices; other layouts produce
// wrong values.
void split_with_sizes_copy_out_cuda_contiguous_no_cast(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  // Every output is empty, and the launch size math below would divide by
  // zero chunks per block.
  if (self.numel() == 0) {
    return;
  }
  const auto device = self.device();
  const auto src_base_addrs =
      detail::get_split_base_addrs(self, split_sizes, dim);
  const auto dst_base_addrs = detail::get_dst_addrs(out);
  const auto src_stride = detail::get_chunk_stride(self, dim);
  const auto split_chunk_sizes =
      detail::get_split_chunk_sizes(self, split_sizes, dim);
  const auto num_chunks = detail::get_num_chunks(self, dim);

  // Calculate the number of blocks required for the first chunk across all
  // splits, assuming each thread only processes BYTES_PER_THREAD bytes.
  int64_t num_blocks = 0;
  for (const auto& split_chunk_size : split_chunk_sizes) {
    num_blocks += detail::div_up(
        split_chunk_size, detail::BLOCK_SIZE * detail::BYTES_PER_THREAD);
  }

  // Calculate the maximum number of blocks to launch. Only consider
  // maxThreadsPerMultiProcessor as a limiting factor as the kernel uses no
  // shared memory and little registers. Over-subscribe the SMs to hide I/O
  // latency.
  const auto num_sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const auto max_threads_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  const int64_t max_blocks =
      num_sms * max_threads_per_sm / detail::BLOCK_SIZE * 2.0;

  // Make each thread process BYTES_PER_THREAD * iter_factor bytes to regulate
  // block size. Spread iter_factor evenly between chunks_per_block and
  // iters_per_chunk.
  int64_t iter_factor = detail::div_up(num_blocks * num_chunks, max_blocks);
  int64_t chunks_per_block = std::ceil(std::sqrt(iter_factor));
  chunks_per_block = std::min(chunks_per_block, num_chunks);
  const int64_t iters_per_chunk = detail::div_up(iter_factor, chunks_per_block);

  // Launch a logically jagged grid of shape
  // (chunk_size*, num_splits, num_chunks / chunks_per_block)
  // backed by a physical grid of shape
  // (sum(chunk_size), num_chunks / chunks_per_block).
  // A block can find its split_idx via block_idx_to_split_idx.
  std::vector<int64_t> block_idx_to_split_idx;
  std::vector<int64_t> blocks_cumsums{0};
  block_idx_to_split_idx.reserve(num_blocks);
  for (size_t split_idx = 0; split_idx < split_sizes.size(); ++split_idx) {
    const auto blocks = detail::div_up(
        split_chunk_sizes[split_idx],
        detail::BLOCK_SIZE * detail::BYTES_PER_THREAD * iters_per_chunk);
    block_idx_to_split_idx.insert(
        block_idx_to_split_idx.end(), blocks, split_idx);
    blocks_cumsums.push_back(blocks_cumsums.back() + blocks);
  }

  dim3 blocks(blocks_cumsums.back(), num_chunks / chunks_per_block, 1);
  dim3 threads(detail::BLOCK_SIZE, 1, 1);

  auto [_, ptrs] = detail::pack_vecs(
      {&dst_base_addrs,
       &src_base_addrs,
       &split_chunk_sizes,
       &block_idx_to_split_idx,
       &blocks_cumsums},
      device);

  detail::split_with_sizes_copy_out_contiguous_no_cast_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      /*dst_base_addrs=*/reinterpret_cast<char**>(ptrs[0]),
      /*src_base_addrs=*/reinterpret_cast<char**>(ptrs[1]),
      /*split_chunk_sizes=*/ptrs[2],
      /*block_idx_to_split_idx=*/ptrs[3],
      /*blocks_cumsums=*/ptrs[4],
      src_stride,
      num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// True if t is dense with its dims in the same memory order as ref_strides;
// size-1 dims can sit anywhere.
static bool has_dense_order(const Tensor& t, IntArrayRef ref_strides) {
  if (!t.is_non_overlapping_and_dense()) {
    return false;
  }
  const auto want = at::infer_dense_strides(t.sizes(), ref_strides);
  for (int64_t d = 0; d < t.dim(); ++d) {
    if (t.size(d) != 1 && t.stride(d) != want[d]) {
      return false;
    }
  }
  return true;
}

// The fast path copies the i-th memory-order chunk of self to the i-th chunk
// of each output, one flat byte range per split, ignoring layout inside the
// range. That is exact when every tensor is dense with the same dim order and
// splits start where the kernel puts them, at multiples of slice_numel. Empty
// outputs never matter.
static bool has_dense_matching_slices(
    const Tensor& self,
    TensorList out,
    int64_t dim) {
  if (self.is_contiguous() && detail::all_contiguous(out)) {
    return true;
  }
  if (self.numel() == 0) {
    return true;
  }
  int64_t non_empty = 0;
  for (const auto& t : out) {
    if (t.numel() == 0) {
      continue;
    }
    if (!has_dense_order(t, self.strides())) {
      return false;
    }
    ++non_empty;
  }
  // One split covering all of self is a flat copy
  if (non_empty == 1) {
    return true;
  }
  // The kernel starts split j at cum_j * slice_numel; it actually starts at
  // cum_j * stride(dim). Several non-empty splits imply size(dim) > 1.
  return self.stride(dim) == c10::multiply_integers(self.sizes().slice(dim + 1));
}

void split_with_sizes_copy_out_cuda(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out) {
  bool dense_no_cast = self.is_non_overlapping_and_dense();
  for (const auto& t : out) {
    dense_no_cast &= (t.dtype() == self.dtype());
  }
  if (dense_no_cast) {
    // Perform equivalent checks performed by the composite impl
    if (dim < 0) {
      dim = at::maybe_wrap_dim(dim, self.dim());
    }
    TORCH_CHECK(
        self.dim() != 0, "split expects at least a 1-dimensional tensor")

    const int64_t dim_size = self.size(dim);
    int64_t split_sizes_sum = 0;
    for (const auto i : c10::irange(split_sizes.size())) {
      TORCH_CHECK(
          split_sizes[i] >= 0,
          "split_with_sizes expects split_sizes have only non-negative ",
          "entries, but got split_sizes=",
          split_sizes[i]);
      split_sizes_sum += split_sizes[i];
    }
    TORCH_CHECK(
        split_sizes_sum == dim_size,
        "split_with_sizes expects split_sizes to sum exactly to ",
        dim_size,
        " (input tensor's size at dimension ",
        dim,
        "), ",
        "but got split_sizes=",
        split_sizes);

    TORCH_CHECK(
        out.size() == split_sizes.size(),
        "split_with_sizes_copy_out() expected an out= argument of size ",
        split_sizes.size(),
        ", got size ",
        out.size());

    auto out_shape = self.sizes().vec();
    for (const auto i : c10::irange(split_sizes.size())) {
      out_shape[dim] = split_sizes[i];
      if (resize_output_check(out[i], out_shape)) {
        out[i].resize_(out_shape);
      }
      TORCH_CHECK(
          out[i].dtype() == self.dtype(),
          "Expected out tensor to have dtype ",
          self.dtype(),
          ", but got ",
          out[i].dtype(),
          " instead");
      TORCH_CHECK(
          out[i].device() == self.device(),
          "Expected out tensor to have device ",
          self.device(),
          ", but got ",
          out[i].device(),
          " instead");
    }
    // Checked after resizing, when every output has its final layout
    if (has_dense_matching_slices(self, out, dim)) {
      split_with_sizes_copy_out_cuda_contiguous_no_cast(
          self, split_sizes, dim, out);
      return;
    }
  }
  at::native::split_with_sizes_copy_out(self, split_sizes, dim, out);
}

Tensor _chunk_cat_cuda(TensorList tensors, int64_t dim, int64_t num_chunks) {
  dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  if (detail::all_contiguous(tensors)) {
    // Return a tensor with the same dtype as input tensors
    int64_t elem_size = tensors[0].element_size();
    int64_t chunk_size =
        detail::get_chunk_size(tensors, dim, num_chunks, elem_size);
    int64_t leading_dim = detail::get_leading_dim(tensors[0].sizes(), dim);
    auto view_sizes = detail::get_chunk_cat_out_sizes(
        tensors[0].sizes(), dim, num_chunks, chunk_size, elem_size);
    Tensor out =
        tensors[0]
            .new_empty(chunk_size * num_chunks * leading_dim / elem_size)
            .view(view_sizes);
    // Type-agnostic copy since out and input tensors have the same type.
    detail::_chunk_cat_out_cuda_contiguous<char, char>(
        tensors, dim, num_chunks, out, elem_size, elem_size);
    return out;
  } else {
    return at::native::_chunk_cat(tensors, dim, num_chunks);
  }
}

Tensor& _chunk_cat_out_cuda(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out) {
  dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  TORCH_CHECK(
      tensors[0].device() == out.device(),
      "_chunk_cat_out_cuda: mismatch between input and out tensor devices");
  bool both_input_output_contiguous =
      detail::all_contiguous(tensors) && out.is_non_overlapping_and_dense();
  if (both_input_output_contiguous &&
      (tensors[0].dtype() == at::ScalarType::BFloat16) &&
      (out.dtype() == at::ScalarType::Float)) {
    // _chunk_cat_out_cuda_contiguous should also support other types, thanks to
    // static_cast_with_inter_type. Here, we dispatch to BFloat16 in and float32
    // out since it is the only known use case.
    detail::_chunk_cat_out_cuda_contiguous<float, BFloat16>(
        tensors,
        dim,
        num_chunks,
        out,
        out.element_size(),
        tensors[0].element_size());
  } else if (
      both_input_output_contiguous && tensors[0].dtype() == out.dtype()) {
    // Type-agnostic copy since out and input tensors have the same type.
    detail::_chunk_cat_out_cuda_contiguous<char, char>(
        tensors,
        dim,
        num_chunks,
        out,
        out.element_size(),
        tensors[0].element_size());
  } else {
    at::native::_chunk_cat_out(tensors, dim, num_chunks, out);
  }
  return out;
}

} // namespace at::native
