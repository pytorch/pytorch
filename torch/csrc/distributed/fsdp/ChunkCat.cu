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

#include <algorithm>
#include <limits>
#include <optional>
#include <type_traits>

namespace torch::distributed::fsdp {
namespace {

// The device code before NOTE [Padded chunk copies] and pack_vecs are copied
// verbatim from aten/src/ATen/native/cuda/TensorShape.cu, where they are not
// exported, apart from namespace qualification.
static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;

static __host__ __device__ inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline void stream_load128(uint4& val, const T* addr) {
  uint64_t low, high;
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  low = reinterpret_cast<const uint64_t*>(addr)[0];
  high = reinterpret_cast<const uint64_t*>(addr)[1];
#else
  asm("ld.global.nc.v2.u64 {%0, %1}, [%2];"
      : "=l"(low), "=l"(high)
      : "l"(addr));
#endif
  reinterpret_cast<uint64_t*>(&val)[0] = low;
  reinterpret_cast<uint64_t*>(&val)[1] = high;
}

template <typename T>
__device__ inline void stream_store128(T* addr, const uint4& val) {
  uint64_t low, high;
  low = reinterpret_cast<const uint64_t*>(&val)[0];
  high = reinterpret_cast<const uint64_t*>(&val)[1];
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  reinterpret_cast<uint64_t*>(addr)[0] = low;
  reinterpret_cast<uint64_t*>(addr)[1] = high;
#else
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

template <typename T>
static __device__ inline bool is_aligned(const void* addr) {
  return reinterpret_cast<uintptr_t>(addr) % sizeof(T) == 0;
}

template <typename T>
static __device__ inline void load128(uint4& val, const char* addr) {
  for (size_t i = 0; i < BYTES_PER_THREAD / sizeof(T); ++i) {
    reinterpret_cast<T*>(&val)[i] = reinterpret_cast<const T*>(addr)[i];
  }
}

template <>
__device__ inline void load128<uint4>(uint4& val, const char* addr) {
  stream_load128(val, addr);
}

static __device__ __inline__ void get_aligned_region(
    char* ptr,
    const int64_t chunk_size,
    const int64_t alignment,
    int64_t& align_off,
    int64_t& aligned_size) {
  const int64_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  align_off = div_up(ptr_val, alignment) * alignment - ptr_val;
  aligned_size = (chunk_size - align_off) / alignment * alignment;
}

// Pack multiple std::vector<int64_t> into a single cuda tensor.
std::pair<at::Tensor, std::vector<int64_t*>> pack_vecs(
    std::vector<const std::vector<int64_t>*> vecs,
    const at::Device& device) {
  int64_t numel = 0;
  for (const auto* vec : vecs) {
    numel += vec->size();
  }

  auto packed = at::empty(
      {numel}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  size_t offset = 0;
  for (const auto* vec : vecs) {
    memcpy(
        packed.data_ptr<int64_t>() + offset,
        vec->data(),
        sizeof(int64_t) * vec->size());
    offset += vec->size();
  }
  packed = packed.to(device, /*non_blocking=*/true);

  std::vector<int64_t*> ptrs;
  ptrs.reserve(vecs.size());
  offset = 0;
  for (const auto* vec : vecs) {
    ptrs.push_back(packed.data_ptr<int64_t>() + offset);
    offset += vec->size();
  }
  return std::make_pair(std::move(packed), std::move(ptrs));
}

// Copy `max_chunk_size` bytes from `src` to `dst` by `num_threads`, and pad
// zero when `src` size (i.e., actual_chunk_size) is less than `max_chunk_size`.
// Assume elements of src and dst have the same data type.
template <typename dst_t, typename src_t>
__device__ __inline__ void copy_chunk_with_pad(
    dst_t* dst_ptr,
    src_t* src_ptr,
    int64_t max_chunk_size,
    int64_t actual_chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  // Supports type cast
  if (!std::is_same_v<dst_t, src_t>) {
    const int64_t max_num_elems = max_chunk_size / sizeof(dst_t);
    const int64_t actual_num_elems = actual_chunk_size / sizeof(src_t);
    int64_t elem_index = thread_idx;
    while (elem_index < actual_num_elems) {
      dst_ptr[elem_index] =
          c10::static_cast_with_inter_type<dst_t, src_t>::apply(
              src_ptr[elem_index]);
      elem_index += num_threads;
    }
    while (elem_index < max_num_elems) {
      dst_ptr[elem_index] =
          c10::static_cast_with_inter_type<dst_t, int>::apply(0);
      elem_index += num_threads;
    }
    return;
  }
  char* dst = reinterpret_cast<char*>(dst_ptr);
  char* src = reinterpret_cast<char*>(src_ptr);
  // Fast path when the number of threads is larger than the number of bytes to
  // be copied (i.e., max_chunk_size). In this case, each thread only copies 1
  // byte. For 0 <= thread_idx < actual_chunk_size, the thread copies data from
  // `src`. For actual_chunk_size <= thread_idx < max_chunk_size, the thread set
  // the val=0 for padding.
  if (max_chunk_size < num_threads) {
    char val = static_cast<char>(0);
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    if (thread_idx < max_chunk_size) {
      dst[thread_idx] = val;
    }
    return;
  }
  // Split dst array into three parts:
  // [dst, dst+align_off), [dst+align_off, dst+align_end), [dst+align_end,
  // dst+max_chunk_size) The second part is aligned with BYTES_PER_THREAD(=16
  // bytes) to enable `stream_store128`.
  int64_t align_off, aligned_size;
  get_aligned_region(
      dst, actual_chunk_size, BYTES_PER_THREAD, align_off, aligned_size);
  int64_t align_end = align_off + aligned_size;
  for (int64_t i = align_off + thread_idx * BYTES_PER_THREAD; i < align_end;
       i += num_threads * BYTES_PER_THREAD) {
    uint4 val;
    if (is_aligned<uint4>(src + i)) {
      stream_load128(val, src + i);
    } else {
      for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
        reinterpret_cast<char*>(&val)[j] = src[i + j];
      }
    }
    stream_store128(&dst[i], val);
  }
  // Copy data for the first part of dst array [dst, dst+align_off).
  // Check `thread_idx<max_chunk_sze` for the edge case that max_chunk_size <
  // align_off.
  if (thread_idx < align_off && thread_idx < max_chunk_size) {
    char val = (char)0;
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    dst[thread_idx] = val;
  }
  // Copy data for the third part of dst array [dst+align_end,
  // dst+max_chunk_size).
  while (align_end + thread_idx < max_chunk_size) {
    char val = (char)0;
    if (align_end + thread_idx < actual_chunk_size) {
      val = src[align_end + thread_idx];
    }
    dst[align_end + thread_idx] = val;
    align_end += num_threads;
  }
}

// NOTE [CUDA kernel for chunk_cat]
// chunk_cat_cuda adopts a "jagged grid" strategy, inspired by NOTE [CUDA fast
// path for split_with_sizes_copy.out]. In addition, chunk_cat_cuda supports
// padding via copy_chunk_with_pad when src chunk size is less than dst chunk
// size.
template <typename dst_t, typename src_t>
static __global__ void chunk_cat_cuda_kernel(
    src_t** src,
    dst_t* dst,
    int64_t* block_idx_to_tensor_idx,
    int64_t* tensor_idx_to_start_tensor_bytes,
    int64_t* start_block_idx_per_tensor_chunk,
    int64_t* actual_tensor_sizes,
    int64_t* pad_tensor_chunk_sizes,
    int64_t* num_blocks_per_tensor_chunk,
    int64_t slice_size,
    int64_t chunk_size,
    int64_t dst_to_src_ratio) {
  const int64_t slice_idx = blockIdx.z;
  const int64_t chunk_idx = blockIdx.y;
  const int64_t tensor_idx = block_idx_to_tensor_idx[blockIdx.x];
  const int64_t tile_idx =
      blockIdx.x - start_block_idx_per_tensor_chunk[tensor_idx];
  // Number of threads for the `tensor_idx`-th tensor chunk.
  const int64_t num_threads =
      num_blocks_per_tensor_chunk[tensor_idx] * BLOCK_SIZE;
  const int64_t thread_idx = tile_idx * BLOCK_SIZE + threadIdx.x;
  char* src_addr = reinterpret_cast<char**>(src)[tensor_idx] +
      slice_idx * actual_tensor_sizes[tensor_idx] +
      chunk_idx * pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio;
  char* dst_addr = reinterpret_cast<char*>(dst) + slice_idx * slice_size +
      chunk_idx * chunk_size + tensor_idx_to_start_tensor_bytes[tensor_idx];
  // Compute the actual number of bytes to copy from src.
  const int64_t actual_copy_size = std::min(
      pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio,
      std::max(
          (int64_t)0,
          actual_tensor_sizes[tensor_idx] -
              chunk_idx * pad_tensor_chunk_sizes[tensor_idx] /
                  dst_to_src_ratio));
  copy_chunk_with_pad<dst_t, src_t>(
      reinterpret_cast<dst_t*>(dst_addr),
      reinterpret_cast<src_t*>(src_addr),
      pad_tensor_chunk_sizes[tensor_idx],
      actual_copy_size,
      thread_idx,
      num_threads);
}

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

// NOTE [Padded chunk copies]
// With leading dims or padded dim sizes, fsdp::chunk_cat_mixed_dtype packs
// tensors into a buffer of num_chunks equal chunks, and
// fsdp::_all_gather_copy_out_ unpacks such a buffer into tensors. Viewed as
// (rows, row_numel), a tensor's rows split into pieces of chunk_numel elements
// as torch.chunk splits a dim: the last piece with data may be shorter, and any
// after it are empty. Piece c of row r is row r of the tensor's slice of chunk
// c, whose rows have chunk_numel elements, so short and empty pieces leave
// padding. Packing zero-fills the padding and unpacking skips it.
//
// One launch copies every tensor, described by a few int64s however many rows
// it has. Its pieces form tiles of about TILE_BYTES: runs of whole pieces if
// they are short, column ranges of one piece if they are long. Each block
// copies a tile, finding its tensor by binary search over the prefix sums of
// the tensors' tile counts.
static constexpr int64_t TILE_BYTES = 8192;
// Shorter pieces are copied by elements, so that a small group of threads can
// copy each of them
static constexpr int64_t MIN_VECTOR_BYTES = 64;
// Elements or vectors that a thread loads before storing them, to keep more
// loads in flight
static constexpr int64_t UNROLL = 4;

// A tensor's descriptor: its address, its slice's address in chunk 0 of the
// padded buffer, rows, row_numel, chunk_numel and its CompactDtype
static constexpr int64_t DESC_FIELDS = 6;

// A tensor's dtype relative to the padded buffer's
enum CompactDtype : int64_t {
  kCompactSameDtype,
  kCompactFloat,
  kCompactBFloat16,
  kCompactHalf
};

struct Tiles {
  int64_t pieces_per_tile;
  int64_t cols_per_tile;
  int64_t col_tiles;
};

static __host__ __device__ inline Tiles get_tiles(
    int64_t chunk_numel,
    int64_t tile_numel) {
  if (chunk_numel >= tile_numel) {
    return {1, tile_numel, div_up(chunk_numel, tile_numel)};
  }
  return {tile_numel / chunk_numel, chunk_numel, 1};
}

// Unpacking skips the empty pieces, which hold only padding
static __host__ __device__ inline int64_t get_num_pieces(
    bool pack,
    int64_t num_chunks,
    int64_t rows,
    int64_t row_numel,
    int64_t chunk_numel) {
  return rows *
      (pack ? num_chunks
            : std::min(num_chunks, div_up(row_numel, chunk_numel)));
}

template <typename dst_t, typename src_t>
static __device__ inline dst_t cast_to(src_t val) {
  if constexpr (std::is_same_v<dst_t, src_t>) {
    return val;
  } else {
    return c10::static_cast_with_inter_type<dst_t, src_t>::apply(val);
  }
}

// Copies num_copy elements and zero-fills the rest up to num_total, each of
// group_size threads taking every group_size-th element.
template <typename dst_t, typename src_t>
static __device__ inline void copy_elems(
    dst_t* dst,
    const src_t* src,
    int64_t num_copy,
    int64_t num_total,
    int64_t lane,
    int64_t group_size) {
  for (int64_t base = lane; base < num_total; base += UNROLL * group_size) {
    src_t vals[UNROLL];
#pragma unroll
    for (int64_t j = 0; j < UNROLL; ++j) {
      const int64_t i = base + j * group_size;
      if (i < num_copy) {
        vals[j] = src[i];
      }
    }
#pragma unroll
    for (int64_t j = 0; j < UNROLL; ++j) {
      const int64_t i = base + j * group_size;
      if (i < num_total) {
        dst[i] = i < num_copy ? cast_to<dst_t>(vals[j]) : dst_t(0);
      }
    }
  }
}

// Elements of each vector of cast_vectors, which takes 16 bytes of each 2-byte
// dtype and 32 of fp32
static constexpr int64_t CAST_VECTOR_NUMEL = 8;

// Like copy_elems, but casts the copied elements in vectors of
// CAST_VECTOR_NUMEL, to which dst and src must be aligned
template <typename dst_t, typename src_t>
static __device__ inline void cast_vectors(
    dst_t* dst,
    const src_t* src,
    int64_t num_copy,
    int64_t num_total,
    int64_t lane,
    int64_t group_size) {
  constexpr int64_t kSrcWords = CAST_VECTOR_NUMEL * sizeof(src_t) / 16;
  constexpr int64_t kDstWords = CAST_VECTOR_NUMEL * sizeof(dst_t) / 16;
  const int64_t num_vectors = num_copy / CAST_VECTOR_NUMEL;
  for (int64_t v = lane; v < num_vectors; v += group_size) {
    uint4 in[kSrcWords];
    uint4 out[kDstWords];
    const auto* src_words =
        reinterpret_cast<const uint4*>(src + v * CAST_VECTOR_NUMEL);
#pragma unroll
    for (int64_t w = 0; w < kSrcWords; ++w) {
      stream_load128(in[w], src_words + w);
    }
#pragma unroll
    for (int64_t i = 0; i < CAST_VECTOR_NUMEL; ++i) {
      reinterpret_cast<dst_t*>(out)[i] =
          cast_to<dst_t>(reinterpret_cast<const src_t*>(in)[i]);
    }
    auto* dst_words = reinterpret_cast<uint4*>(dst + v * CAST_VECTOR_NUMEL);
#pragma unroll
    for (int64_t w = 0; w < kDstWords; ++w) {
      stream_store128(dst_words + w, out[w]);
    }
  }
  const int64_t num_cast = num_vectors * CAST_VECTOR_NUMEL;
  copy_elems(
      dst + num_cast,
      src + num_cast,
      num_copy - num_cast,
      num_total - num_cast,
      lane,
      group_size);
}

// Loads 16 bytes from word_t-aligned memory with the widest aligned loads
template <typename word_t>
static __device__ inline void load_words(uint4& val, const char* addr) {
  if (is_aligned<uint4>(addr)) {
    load128<uint4>(val, addr);
  } else if (is_aligned<uint64_t>(addr)) {
    load128<uint64_t>(val, addr);
  } else if (is_aligned<uint32_t>(addr)) {
    load128<uint32_t>(val, addr);
  } else {
    load128<word_t>(val, addr);
  }
}

// Like copy_elems for one dtype, whose elements are word_t, but with 16-byte
// stores aligned on dst
template <typename word_t>
static __device__ inline void copy_vectors(
    char* dst,
    const char* src,
    int64_t copy_bytes,
    int64_t total_bytes,
    int64_t lane,
    int64_t group_size) {
  constexpr int64_t kWord = sizeof(word_t);
  constexpr int64_t kWordsPerVector = BYTES_PER_THREAD / kWord;
  const auto misalignment =
      static_cast<int64_t>(reinterpret_cast<uintptr_t>(dst) % BYTES_PER_THREAD);
  const int64_t head = std::min(
      total_bytes, (BYTES_PER_THREAD - misalignment) % BYTES_PER_THREAD);
  const int64_t body_end =
      head + (total_bytes - head) / BYTES_PER_THREAD * BYTES_PER_THREAD;
  const auto copy_word = [&](int64_t off) {
    *reinterpret_cast<word_t*>(dst + off) = off < copy_bytes
        ? *reinterpret_cast<const word_t*>(src + off)
        : word_t(0);
  };
  for (int64_t off = lane * kWord; off < head; off += group_size * kWord) {
    copy_word(off);
  }
  for (int64_t off = body_end + lane * kWord; off < total_bytes;
       off += group_size * kWord) {
    copy_word(off);
  }
  const int64_t stride = group_size * BYTES_PER_THREAD;
  for (int64_t base = head + lane * BYTES_PER_THREAD; base < body_end;
       base += UNROLL * stride) {
    uint4 vals[UNROLL];
#pragma unroll
    for (int64_t j = 0; j < UNROLL; ++j) {
      const int64_t off = base + j * stride;
      if (off >= body_end) {
        continue;
      }
      if (off + BYTES_PER_THREAD <= copy_bytes) {
        load_words<word_t>(vals[j], src + off);
      } else if (off >= copy_bytes) {
        vals[j] = make_uint4(0, 0, 0, 0);
      } else {
        for (int64_t w = 0; w < kWordsPerVector; ++w) {
          reinterpret_cast<word_t*>(&vals[j])[w] = off + w * kWord < copy_bytes
              ? reinterpret_cast<const word_t*>(src + off)[w]
              : word_t(0);
        }
      }
    }
#pragma unroll
    for (int64_t j = 0; j < UNROLL; ++j) {
      const int64_t off = base + j * stride;
      if (off < body_end) {
        stream_store128(dst + off, vals[j]);
      }
    }
  }
}

// Copies the pieces [piece_begin, piece_end), columns [col_begin, col_end), of
// a tensor and its slice of the padded buffer. Each piece is copied by a group
// of threads, the smallest power of two that covers its columns (in vectors or
// elements) in UNROLL steps, so short pieces are copied side by side.
template <bool kPack, typename padded_t, typename compact_t>
static __device__ inline void copy_pieces(
    char* compact,
    char* padded,
    int64_t rows,
    int64_t row_numel,
    int64_t chunk_numel,
    int64_t chunk_stride,
    int64_t piece_begin,
    int64_t piece_end,
    int64_t col_begin,
    int64_t col_end) {
  // Tensors in the padded buffer's dtype are copied as words of its size
  constexpr bool kSameType = std::is_same_v<padded_t, compact_t>;
  constexpr int64_t kElemSize = sizeof(padded_t);
  const int64_t cols = col_end - col_begin;
  bool vectors = false;
  int64_t units = div_up(cols, UNROLL);
  if constexpr (kSameType) {
    if (cols * kElemSize >= MIN_VECTOR_BYTES) {
      vectors = true;
      // A misaligned piece spans one more vector
      units = div_up(div_up(cols * kElemSize, BYTES_PER_THREAD) + 1, UNROLL);
    }
  } else {
    // Casts take vectors if every piece of the tile starts aligned to them
    constexpr auto kPaddedAlign = CAST_VECTOR_NUMEL * sizeof(padded_t);
    constexpr auto kCompactAlign = CAST_VECTOR_NUMEL * sizeof(compact_t);
    vectors = reinterpret_cast<uintptr_t>(padded) % kPaddedAlign == 0 &&
        chunk_stride % kPaddedAlign == 0 &&
        reinterpret_cast<uintptr_t>(compact) % kCompactAlign == 0 &&
        (row_numel | chunk_numel | col_begin) % CAST_VECTOR_NUMEL == 0;
    if (vectors) {
      units = div_up(cols, CAST_VECTOR_NUMEL);
    }
  }
  int64_t group_size = 1;
  while (group_size < units && group_size < BLOCK_SIZE) {
    group_size *= 2;
  }
  const int64_t lane = threadIdx.x % group_size;
  for (int64_t piece = piece_begin + threadIdx.x / group_size;
       piece < piece_end;
       piece += BLOCK_SIZE / group_size) {
    const int64_t chunk = rows == 1 ? piece : piece / rows;
    const int64_t row = piece - chunk * rows;
    const int64_t data_numel = std::min(
        chunk_numel, std::max(int64_t(0), row_numel - chunk * chunk_numel));
    const int64_t num_copy =
        std::min(cols, std::max(int64_t(0), data_numel - col_begin));
    auto* padded_piece =
        reinterpret_cast<padded_t*>(padded + chunk * chunk_stride) +
        row * chunk_numel + col_begin;
    auto* compact_piece = reinterpret_cast<compact_t*>(compact) +
        row * row_numel + chunk * chunk_numel + col_begin;
    if constexpr (kPack) {
      if constexpr (kSameType) {
        if (vectors) {
          copy_vectors<padded_t>(
              reinterpret_cast<char*>(padded_piece),
              reinterpret_cast<const char*>(compact_piece),
              num_copy * kElemSize,
              cols * kElemSize,
              lane,
              group_size);
          continue;
        }
      } else {
        if (vectors) {
          cast_vectors(
              padded_piece, compact_piece, num_copy, cols, lane, group_size);
          continue;
        }
      }
      copy_elems(padded_piece, compact_piece, num_copy, cols, lane, group_size);
    } else {
      if constexpr (kSameType) {
        if (vectors) {
          copy_vectors<padded_t>(
              reinterpret_cast<char*>(compact_piece),
              reinterpret_cast<const char*>(padded_piece),
              num_copy * kElemSize,
              num_copy * kElemSize,
              lane,
              group_size);
          continue;
        }
      }
      copy_elems(
          compact_piece, padded_piece, num_copy, num_copy, lane, group_size);
    }
  }
}

// See NOTE [Padded chunk copies]. padded_t is the padded buffer's dtype, or a
// word type of its size if the kernel doesn't cast it, and word_t is the word
// type of its size.
template <bool kPack, typename padded_t, typename word_t>
static __global__ void padded_chunk_copy_kernel(
    const int64_t* __restrict__ descs,
    const int64_t* __restrict__ tile_offsets,
    int64_t num_descs,
    int64_t num_chunks,
    int64_t chunk_stride,
    int64_t tile_numel) {
  const int64_t tile = blockIdx.x;
  int64_t lo = 0;
  int64_t hi = num_descs - 1;
  while (lo < hi) {
    const int64_t mid = (lo + hi + 1) / 2;
    if (tile_offsets[mid] <= tile) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int64_t* desc = descs + lo * DESC_FIELDS;
  auto* compact = reinterpret_cast<char*>(desc[0]);
  auto* padded = reinterpret_cast<char*>(desc[1]);
  const int64_t rows = desc[2];
  const int64_t row_numel = desc[3];
  const int64_t chunk_numel = desc[4];
  const Tiles tiles = get_tiles(chunk_numel, tile_numel);
  const int64_t tile_idx = tile - tile_offsets[lo];
  const int64_t piece_begin =
      tile_idx / tiles.col_tiles * tiles.pieces_per_tile;
  const int64_t piece_end = std::min(
      piece_begin + tiles.pieces_per_tile,
      get_num_pieces(kPack, num_chunks, rows, row_numel, chunk_numel));
  const int64_t col_begin = tile_idx % tiles.col_tiles * tiles.cols_per_tile;
  const int64_t col_end =
      std::min(col_begin + tiles.cols_per_tile, chunk_numel);
#define COPY_PIECES(padded_type, compact_type)   \
  copy_pieces<kPack, padded_type, compact_type>( \
      compact,                                   \
      padded,                                    \
      rows,                                      \
      row_numel,                                 \
      chunk_numel,                               \
      chunk_stride,                              \
      piece_begin,                               \
      piece_end,                                 \
      col_begin,                                 \
      col_end)
  // The host only passes casts into a padded buffer of another floating dtype
  constexpr bool kCasts = kPack &&
      (std::is_same_v<padded_t, float> ||
       std::is_same_v<padded_t, c10::BFloat16> ||
       std::is_same_v<padded_t, c10::Half>);
  switch (desc[5]) {
    case kCompactSameDtype:
      COPY_PIECES(word_t, word_t);
      break;
    case kCompactFloat:
      if constexpr (kCasts && !std::is_same_v<padded_t, float>) {
        COPY_PIECES(padded_t, float);
      }
      break;
    case kCompactBFloat16:
      if constexpr (kCasts && !std::is_same_v<padded_t, c10::BFloat16>) {
        COPY_PIECES(padded_t, c10::BFloat16);
      }
      break;
    case kCompactHalf:
      if constexpr (kCasts && !std::is_same_v<padded_t, c10::Half>) {
        COPY_PIECES(padded_t, c10::Half);
      }
      break;
  }
#undef COPY_PIECES
}

// The CompactDtype for packing a tensor of compact_dtype into a padded buffer
// of padded_dtype, or for unpacking one of the same dtype, or nullopt if
// padded_chunk_copy_kernel can't copy it
std::optional<CompactDtype> get_compact_dtype(
    at::ScalarType compact_dtype,
    at::ScalarType padded_dtype) {
  if (compact_dtype == padded_dtype) {
    return c10::elementSize(padded_dtype) <= 8
        ? std::optional<CompactDtype>(kCompactSameDtype)
        : std::nullopt;
  }
  const auto floating =
      [](at::ScalarType dtype) -> std::optional<CompactDtype> {
    switch (dtype) {
      case at::kFloat:
        return kCompactFloat;
      case at::kBFloat16:
        return kCompactBFloat16;
      case at::kHalf:
        return kCompactHalf;
      default:
        return std::nullopt;
    }
  };
  return floating(padded_dtype).has_value() ? floating(compact_dtype)
                                            : std::nullopt;
}

// Collects the descriptors and tiles of one padded_chunk_copy_kernel launch
class PaddedChunkCopies {
 public:
  PaddedChunkCopies(bool pack, int64_t num_chunks, at::ScalarType padded_dtype)
      : pack_(pack),
        num_chunks_(num_chunks),
        padded_dtype_(padded_dtype),
        tile_numel_(TILE_BYTES / c10::elementSize(padded_dtype)) {}

  void add(
      const at::Tensor& tensor,
      CompactDtype dtype,
      const char* padded,
      int64_t rows,
      int64_t row_numel,
      int64_t chunk_numel) {
    if (num_chunks_ == 1 &&
        (row_numel == chunk_numel || (pack_ && row_numel == 0))) {
      // Rows are back to back on both sides, so copy them as one
      row_numel *= rows;
      chunk_numel *= rows;
      rows = 1;
    }
    if (rows == 0 || chunk_numel == 0) {
      return;
    }
    const auto num_pieces =
        get_num_pieces(pack_, num_chunks_, rows, row_numel, chunk_numel);
    if (num_pieces == 0) {
      return;
    }
    const auto tiles = get_tiles(chunk_numel, tile_numel_);
    const auto* compact =
        pack_ ? tensor.const_data_ptr() : tensor.mutable_data_ptr();
    descs_.insert(
        descs_.end(),
        {reinterpret_cast<int64_t>(compact),
         reinterpret_cast<int64_t>(padded),
         rows,
         row_numel,
         chunk_numel,
         dtype});
    tile_offsets_.push_back(
        tile_offsets_.back() +
        div_up(num_pieces, tiles.pieces_per_tile) * tiles.col_tiles);
  }

  // chunk_stride is the padded buffer's chunk size in bytes
  void launch(int64_t chunk_stride, const at::Device& device) {
    switch (padded_dtype_) {
      case at::kFloat:
        return launch_kernel<float, uint32_t>(chunk_stride, device);
      case at::kBFloat16:
        return launch_kernel<c10::BFloat16, uint16_t>(chunk_stride, device);
      case at::kHalf:
        return launch_kernel<c10::Half, uint16_t>(chunk_stride, device);
      default:
        break;
    }
    switch (c10::elementSize(padded_dtype_)) {
      case 1:
        return launch_kernel<uint8_t, uint8_t>(chunk_stride, device);
      case 2:
        return launch_kernel<uint16_t, uint16_t>(chunk_stride, device);
      case 4:
        return launch_kernel<uint32_t, uint32_t>(chunk_stride, device);
      case 8:
        return launch_kernel<uint64_t, uint64_t>(chunk_stride, device);
      default:
        TORCH_INTERNAL_ASSERT(false, "unsupported dtype ", padded_dtype_);
    }
  }

 private:
  template <typename padded_t, typename word_t>
  void launch_kernel(int64_t chunk_stride, const at::Device& device) {
    const auto num_descs = static_cast<int64_t>(descs_.size()) / DESC_FIELDS;
    if (num_descs == 0) {
      return;
    }
    const auto num_tiles = tile_offsets_.back();
    TORCH_INTERNAL_ASSERT(num_tiles <= std::numeric_limits<int32_t>::max());
    // `packed` keeps the device metadata alive until the kernel is enqueued.
    const auto packed = pack_vecs({&descs_, &tile_offsets_}, device);
    const auto& ptrs = packed.second;
    const auto kernel = pack_
        ? padded_chunk_copy_kernel<true, padded_t, word_t>
        : padded_chunk_copy_kernel<false, padded_t, word_t>;
    kernel<<<
        dim3(num_tiles, 1, 1),
        dim3(BLOCK_SIZE, 1, 1),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        ptrs[0], ptrs[1], num_descs, num_chunks_, chunk_stride, tile_numel_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const bool pack_;
  const int64_t num_chunks_;
  const at::ScalarType padded_dtype_;
  const int64_t tile_numel_;
  std::vector<int64_t> descs_;
  std::vector<int64_t> tile_offsets_{0};
};

// Inputs already in out's dtype are copied by one launch, and cast_dtype inputs
// into an fp32 out are cast by a second one. The output layout doesn't depend
// on input dtypes, so both launches share one metadata upload and differ only
// in their block_idx_to_tensor_idx ranges.
// start_block_idx_per_tensor_chunk is relative to each tensor's launch.
void fused_chunk_cat(
    at::TensorList tensors,
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
    const int64_t row_numel = c10::multiply_integers(sizes.slice(1));
    const int64_t pad_tensor_chunk_size =
        div_up(sizes[0], num_chunks) * row_numel * out.element_size();
    const int64_t actual_tensor_size = tensor.numel() * tensor.element_size();
    const int64_t num_blocks = div_up(pad_tensor_chunk_size, BYTES_PER_BLOCK);
    auto& blocks =
        tensor.scalar_type() == out_dtype ? copy_blocks : cast_blocks;
    // The cast launch below reads cast_dtype and writes fp32
    TORCH_INTERNAL_ASSERT(
        tensor.scalar_type() == out_dtype ||
        (tensor.scalar_type() == cast_dtype && out_dtype == at::kFloat));
    srcs.push_back(reinterpret_cast<int64_t>(tensor.const_data_ptr()));
    tensor_idx_to_start_tensor_bytes.push_back(chunk_size);
    start_block_idx_per_tensor_chunk.push_back(
        static_cast<int64_t>(blocks.size()));
    blocks.insert(blocks.end(), num_blocks, static_cast<int64_t>(i));
    actual_tensor_sizes.push_back(actual_tensor_size);
    pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
    num_blocks_per_tensor_chunk.push_back(num_blocks);
    chunk_size += pad_tensor_chunk_size;
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

// The CompactDtype of each tensor for padded_chunk_copy_kernel, or nullopt if
// the composite must copy them
std::optional<std::vector<CompactDtype>> get_compact_dtypes(
    at::TensorList tensors,
    const at::Tensor& padded) {
  if (use_composite(tensors, padded)) {
    return std::nullopt;
  }
  std::vector<CompactDtype> dtypes;
  dtypes.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    const auto dtype =
        get_compact_dtype(tensor.scalar_type(), padded.scalar_type());
    if (!dtype.has_value()) {
      return std::nullopt;
    }
    dtypes.push_back(*dtype);
  }
  return dtypes;
}

void all_gather_copy_out_cuda(
    at::TensorList out,
    const at::Tensor& input,
    c10::SymIntArrayRef split_sizes,
    c10::SymIntArrayRef outer_sizes,
    int64_t num_chunks,
    at::OptionalSymIntArrayRef output_inner_sizes) {
  const auto dtypes = get_compact_dtypes(out, input);
  if (!dtypes.has_value()) {
    all_gather_copy_out(
        out, input, split_sizes, outer_sizes, num_chunks, output_inner_sizes);
    return;
  }
  check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks, output_inner_sizes);
  c10::cuda::CUDAGuard device_guard(input.device());
  const auto elem_size = input.element_size();
  PaddedChunkCopies copies(/*pack=*/false, num_chunks, input.scalar_type());
  const auto* padded = static_cast<const char*>(input.const_data_ptr());
  for (const auto i : c10::irange(out.size())) {
    const auto split_size = split_sizes[i].expect_int();
    const auto outer_size = outer_sizes[i].expect_int();
    const auto chunk_numel = split_size / outer_size;
    const auto row_numel = output_inner_sizes.has_value()
        ? (*output_inner_sizes)[i].expect_int()
        : chunk_numel * num_chunks;
    copies.add(
        out[i], (*dtypes)[i], padded, outer_size, row_numel, chunk_numel);
    padded += split_size * elem_size;
  }
  copies.launch(input.numel() / num_chunks * elem_size, input.device());
}

void chunk_cat_padded_cuda(
    at::TensorList tensors,
    const std::vector<CompactDtype>& dtypes,
    int64_t num_chunks,
    at::Tensor& out,
    at::IntArrayRef num_leading_dims,
    at::OptionalIntArrayRef padded_dim_sizes) {
  check_chunk_cat_leading_dims_inputs(
      tensors, num_leading_dims, num_chunks, out, padded_dim_sizes);
  c10::cuda::CUDAGuard device_guard(out.device());
  const auto elem_size = out.element_size();
  PaddedChunkCopies copies(/*pack=*/true, num_chunks, out.scalar_type());
  const auto* padded = static_cast<const char*>(out.mutable_data_ptr());
  for (const auto i : c10::irange(tensors.size())) {
    const auto sizes = tensors[i].sizes();
    const auto dim = num_leading_dims[i];
    const auto rows = c10::multiply_integers(sizes.slice(0, dim));
    const auto inner_size = c10::multiply_integers(sizes.slice(dim + 1));
    const auto chunk_rows = padded_dim_sizes.has_value()
        ? (*padded_dim_sizes)[i] / num_chunks
        : div_up(sizes[dim], num_chunks);
    const auto chunk_numel = chunk_rows * inner_size;
    copies.add(
        tensors[i],
        dtypes[i],
        padded,
        rows,
        sizes[dim] * inner_size,
        chunk_numel);
    padded += rows * chunk_numel * elem_size;
  }
  copies.launch(out.numel() / num_chunks * elem_size, out.device());
}

// Uniform groups without leading dims or padded dim sizes take the composite,
// which forwards them to _chunk_cat. So do dim != 0, which FSDP doesn't use,
// and dtypes the fused kernels can't cast.
void chunk_cat_mixed_dtype_cuda(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out,
    at::OptionalIntArrayRef num_leading_dims,
    at::OptionalIntArrayRef padded_dim_sizes) {
  if (has_leading_dims(num_leading_dims) || padded_dim_sizes.has_value()) {
    const auto dtypes = dim == 0 && num_chunks > 0 && !tensors.empty()
        ? get_compact_dtypes(tensors, out)
        : std::nullopt;
    if (!dtypes.has_value()) {
      chunk_cat_mixed_dtype(
          tensors, dim, num_chunks, out, num_leading_dims, padded_dim_sizes);
      return;
    }
    chunk_cat_padded_cuda(
        tensors,
        *dtypes,
        num_chunks,
        out,
        leading_dims_or_zeros(num_leading_dims, tensors.size()),
        padded_dim_sizes);
    return;
  }
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
  const bool use_fused_kernel =
      mixed_dtypes && fused_dtypes && dim == 0 && num_chunks >= 1 &&
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [](const at::Tensor& t) { return t.dim() > 0 && t.numel() > 0; }) &&
      !use_composite(tensors, out);
  if (!use_fused_kernel) {
    chunk_cat_mixed_dtype(
        tensors, dim, num_chunks, out, num_leading_dims, padded_dim_sizes);
    return;
  }
  fused_chunk_cat(tensors, num_chunks, cast_dtype.value_or(at::kBFloat16), out);
}

TORCH_LIBRARY_IMPL(fsdp, CUDA, m) {
  m.impl("chunk_cat_mixed_dtype", TORCH_FN(chunk_cat_mixed_dtype_cuda));
  m.impl("_all_gather_copy_out_", TORCH_FN(all_gather_copy_out_cuda));
}

} // namespace
} // namespace torch::distributed::fsdp
