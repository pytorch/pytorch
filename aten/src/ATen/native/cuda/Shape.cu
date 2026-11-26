#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/Resize.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/TensorShape.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/CallOnce.h>

#include <array>
#include <limits>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/narrow.h>
#endif

namespace at::native {

constexpr int CAT_ARRAY_BATCH_SIZE = 128;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;
constexpr int ALIGNED_VEC_LOAD_BYTES_16 = 16;
constexpr int ALIGNED_VEC_LOAD_BYTES_8 = 8;

namespace {

inline bool is_aligned_vec4(const void* ptr) {
  auto iptr = reinterpret_cast<uintptr_t>(ptr);
  return !(iptr % alignof(int4));
}

#ifdef USE_ROCM
bool is_gfx1250_device(const c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(
      device_index >= 0 && device_index < C10_COMPILE_TIME_MAX_GPUS);
  static std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> once_flags;
  static std::array<bool, C10_COMPILE_TIME_MAX_GPUS> results{};
  c10::call_once(once_flags[device_index], [device_index] {
    results[device_index] =
        at::detail::getCUDAHooks().isGPUArch({"gfx1250"}, device_index);
  });
  return results[device_index];
}
#endif

inline bool getCatGrid(ptrdiff_t nTensors, dim3& grid) {
  const int numSM = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // X dim of grid for cat array cooperates on a single tensor in the cat.
  // Given half of the GPU, full utilization will always occur.

  // This will have cating two tensors fill the entire grid, but prevent
  // many threads from needlessly load meta data if their sizes is small.

  grid = dim3( 2LL * numSM, (long long) nTensors );

  return true;
}

template<typename T>
inline std::tuple<dim3, dim3> getCatGridRocm(unsigned int max_elements_per_tensor,
  ptrdiff_t nTensors) {
  constexpr unsigned int threads_per_block = 256;
  constexpr unsigned int elements_per_thread = 8;
  constexpr unsigned int max_tb_per_sm = 32;

  unsigned int max_threads = ceil_div(max_elements_per_tensor, elements_per_thread);
  unsigned int thread_blocks = ceil_div(max_threads, threads_per_block);

  // Limit the number of thread blocks to prevent too many threads to load the metadata
  // if they operate on very small tensors.

  const unsigned int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  thread_blocks = std::min(num_sm * max_tb_per_sm, thread_blocks);

  dim3 block = dim3(threads_per_block);
  dim3 grid = dim3(thread_blocks, (long long)nTensors);

  return std::make_tuple(grid, block);
}

template<typename T, int aligned_vec_load_bytes>
inline std::tuple<dim3, dim3> getCatGridContig(unsigned int max_elements_per_tensor,
  ptrdiff_t nTensors) {
  constexpr unsigned int threads_per_block = 128;
  constexpr unsigned int min_aligned_vec_per_thread = 1;
  constexpr unsigned int max_tb_per_sm = 32;

  unsigned int elements_per_thread = aligned_vec_load_bytes / sizeof(T) *
    min_aligned_vec_per_thread;
  unsigned int max_threads = ceil_div(max_elements_per_tensor, elements_per_thread);
  unsigned int thread_blocks = ceil_div(max_threads, threads_per_block);

  // Limit the number of thread blocks to prevent too many threads to load the metadata
  // if they operate on very small tensors.

  const unsigned int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  thread_blocks = std::min(num_sm * max_tb_per_sm, thread_blocks);

  dim3 block = dim3(threads_per_block);
  dim3 grid = dim3(thread_blocks, (long long)nTensors);

  return std::make_tuple(grid, block);
}

// Similar to any other IndexToOffset calculation for copying along a given
// dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline __device__ IndexType compute(
      const IndexType tensorSize[Dims],
      const IndexType tensorStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    // linearIndex is not really linear index, but instead the offset in
    // input tensor. If the input tensor is contiguous, then this offset
    // is the linear index, but if the input tensor is channels last, then
    // it is the linear index of the permuted contiguous tensor
    IndexType offset = 0;

    #pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : tensorSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * tensorStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * tensorStride[0];
  }
};

template<typename IndexType, unsigned int MaxDims>
struct TensorSizeStride {
  IndexType tensorSize[MaxDims];
  IndexType tensorStride[MaxDims];
};

/**
  * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a
  * grid-stride loop based off of the blockIdx.x, threadIdx.x for each input to
  * copy each element from each input tensor into the output.
  *
  * output: base pointer to the storage associated with the output tensor
  * inputs: GPU-allocated array of input metadata for each input to concatenate
  *         in the kernel
  * os: the size/stride vectors for the output tensor
  * concatDim: dimension along which we are concatenating
  * dimStride: the stride of the output tensor at the concatDim
  *
  * The most important assumption made is that the input tensors are contiguous.
  */


// pass meta data directly through kernel argument instead of pin memory
// In contiguous case, we will not need stride_size, setting it as 1 as placeholder
// to pass compile.
template <typename T, typename IndexType, int n, int stride_size>
struct CatArrInputTensorMetadata {
  const T* input[n];
  IndexType offset[n];
  IndexType dimSize[n];
  IndexType nElements[n];
  bool isContiguous[n];
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> tensorStride[stride_size];
};

template <typename T, typename IndexType, int Dims, int batch_size, int stride_size>
__global__ void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs.nElements[blockIdx.y];
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> ins = stride_size > 1 ? inputs.tensorStride[blockIdx.y] : inputs.tensorStride[0];
    bool isContig = inputs.isContiguous[blockIdx.y];

    if(tid >= nElements) return;

    const T* data = inputs.input[blockIdx.y];
    IndexType offset = inputs.offset[blockIdx.y];
    IndexType dimSize = inputs.dimSize[blockIdx.y];
    IndexType dataOffset = offset * dimStride;

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    os.tensorSize, os.tensorStride, dimSize, concatDim, tid);
      if (isContig) {
        output[dataOffset + elementOffset] = data[tid];
      } else {
        IndexType inElementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    ins.tensorSize, ins.tensorStride, dimSize, concatDim, tid);
        output[dataOffset + elementOffset] = data[inElementOffset];
      }
    tid += stride;
    }
}

template <typename T, typename IndexType, int Dims, int batch_size, int stride_size>
__global__ void CatArrayBatchedCopy_contig(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs.nElements[blockIdx.y];

    if(tid >= nElements) return;

    const T* data = inputs.input[blockIdx.y];
    IndexType offset = inputs.offset[blockIdx.y];
    IndexType dimSize = inputs.dimSize[blockIdx.y];
    IndexType dataOffset = offset * dimStride;

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    os.tensorSize, os.tensorStride, dimSize, concatDim, tid);
      output[dataOffset + elementOffset] = data[tid];
      tid += stride;
    }
}


template <typename T, typename IndexType, int Dims, int batch_size, int stride_size, int alignment, int elems_per_vec>
__global__ void CatArrayBatchedCopy_vectorized(
    char* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType trailingSize) {
// On ROCm only gfx1250 launches this kernel (vectorizedCatAllowed in
// parallel_cat), so other ROCm targets compile a stub.
#if !defined(USE_ROCM) || defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__)
    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs.nElements[blockIdx.y] / elems_per_vec;

    if(tid >= nElements) return;

    const char * data = (char*)inputs.input[blockIdx.y];
    IndexType offset = inputs.offset[blockIdx.y] * trailingSize / elems_per_vec;
    IndexType dimSize = inputs.dimSize[blockIdx.y] * trailingSize / elems_per_vec;
    int64_t dataOffset = (int64_t)offset  * alignment; // in bytes

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
      int64_t elementOffset = (int64_t)CatArrIndexToOffset<IndexType, Dims>::compute(
                    os.tensorSize, os.tensorStride, dimSize, concatDim, tid) * alignment; // in bytes
      auto vec = at::native::memory::ld_vec<alignment>(data + (int64_t)alignment * tid);
      at::native::memory::st_vec<alignment>(output + dataOffset + elementOffset, vec);
      tid += stride;
    }
#else
  CUDA_KERNEL_ASSERT(false && "CatArrayBatchedCopy_vectorized is gfx1250-only on ROCm");
#endif
}



/*
  Specialized implementation of the CatArrayBatchedCopy written to generate wide memory loads
  to improve memory bandwidth throughput.
*/

template <typename T, typename IndexType, int Dims, int batch_size, int stride_size, int aligned_vec_load_bytes>
__global__ void CatArrayBatchedCopy_alignedK_contig(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    // This kernel tries to use aligned_vec_load_bytes*8 bit loads
    // Special case 2-byte types to use 8-byte vec loads to reduce register pressure
    // The below lambda is to allow cc compiler to pass kILP>0 checks for large types (e.g. ComplexDouble, 16 bytes)
    constexpr int kILP = aligned_vec_load_bytes / sizeof(T) > 0 ? aligned_vec_load_bytes / sizeof(T) : ALIGNED_VEC_LOAD_BYTES_16/sizeof(T);

    IndexType inputOffset = (blockIdx.x * blockDim.x + threadIdx.x) * kILP;
    IndexType inputStride = gridDim.x * blockDim.x * kILP;

    IndexType nElements = inputs.nElements[blockIdx.y];
    if (inputOffset >= nElements) {
      return;
    }

    const T* data = inputs.input[blockIdx.y];
    IndexType offset = inputs.offset[blockIdx.y];
    IndexType dimSize = inputs.dimSize[blockIdx.y];
    IndexType dataOffset = offset * dimStride;

    IndexType v_elementOffset[kILP];
    T reg_data[kILP];

    while (inputOffset + kILP <= nElements) {
      for (int i = 0; i < kILP; ++i) {
        v_elementOffset[i] = CatArrIndexToOffset<IndexType, Dims>::compute(os.tensorSize,
          os.tensorStride, dimSize, concatDim, inputOffset + i);
      }

      using LT = at::native::memory::aligned_vector<T, kILP>;
      ((LT*)reg_data)[0] = ((LT*)(data + inputOffset))[0];

      #pragma unroll
      for (int i = 0; i < kILP; ++i) {
        output[dataOffset + v_elementOffset[i]] = reg_data[i];
      }

      inputOffset += inputStride;
    }

    // Handle remaining tail in case nElements does not divide
    // exactly to kILP

    while (inputOffset < nElements) {
      v_elementOffset[0] = CatArrIndexToOffset<IndexType, Dims>::compute(os.tensorSize,
        os.tensorStride, dimSize, concatDim, inputOffset);
      output[dataOffset + v_elementOffset[0]] = data[inputOffset];
      inputOffset++;
    }
}

#ifdef USE_ROCM
// Returns the 16 bytes that start byte_offset (0..15) bytes into lo:hi, where
// lo is the lower-addressed half: a funnel shift of 64-bit halves.
__device__ __forceinline__ at::native::memory::Vec<16> extract_16_bytes(
    const at::native::memory::Vec<16>& lo,
    const at::native::memory::Vec<16>& hi,
    unsigned int byte_offset) {
  constexpr unsigned int half_bytes = sizeof(uint64_t);
  constexpr unsigned int half_bits = half_bytes * 8;
  const bool skip_half = byte_offset >= half_bytes;
  const uint64_t first = skip_half ? lo.u64[1] : lo.u64[0];
  const uint64_t second = skip_half ? hi.u64[0] : lo.u64[1];
  const uint64_t third = skip_half ? hi.u64[1] : hi.u64[0];
  const unsigned int shift = (byte_offset % half_bytes) * 8;
  at::native::memory::Vec<16> out;
  out.u64[0] = shift ? (first >> shift) | (second << (half_bits - shift)) : first;
  out.u64[1] = shift ? (second >> shift) | (third << (half_bits - shift)) : second;
  return out;
}

/*
  gfx1250 kernel for contiguous cats that CatArrayBatchedCopy_vectorized cannot
  take because a slice is not a multiple of 16 bytes or a pointer is not 16-byte
  aligned, and for the inputs of strided cats whose rows are contiguous. Each
  input is viewed as outerSize rows of rowSize contiguous elements; row r lands
  in output row r at column offset * innerSize. Input rows are back to back
  when stride_size == 1; when stride_size > 1, the host stores input i's row
  stride in tensorStride[i].tensorStride[0]. A thread owns one
  16-byte-aligned chunk of the output that a row overlaps and reads the row
  bytes it covers with two aligned 16-byte input loads: a chunk fully inside the
  row is written with one 16-byte store, and a partial chunk at an end of the
  row with one store per element. A chunk whose aligned loads would reach
  before the input's first byte or past its last byte (only chunks within 32
  bytes of either end of the input can) reads its bytes with per-element loads
  instead, so no byte outside the input is read.
*/
template <typename T, typename IndexType, int batch_size, int stride_size>
__global__ void CatArrayBatchedCopy_unaligned_vectorized(
    char* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    IndexType innerSize,
    IndexType outputRowSize,
    IndexType outerSize) {
#if defined(__gfx1250__) || defined(__gfx1250_strict__) || defined(__SPIRV__)
  constexpr unsigned int vec_bytes = ALIGNED_VEC_LOAD_BYTES_16;
  constexpr IndexType elems_per_vec = vec_bytes / sizeof(T);
  const IndexType rowSize = inputs.dimSize[blockIdx.y] * innerSize;
  if (rowSize == 0) {
    return;
  }
  // A row of rowSize elements overlaps at most this many aligned output chunks.
  const IndexType chunksPerRow = (rowSize + elems_per_vec - 1) / elems_per_vec + 1;
  const IndexType nChunks = outerSize * chunksPerRow;
  const char* input = reinterpret_cast<const char*>(inputs.input[blockIdx.y]);
  // Byte offsets below are relative to output; outputMisalignment places them
  // on the 16-byte grid of addresses.
  const int64_t outputMisalignment = reinterpret_cast<uintptr_t>(output) % vec_bytes;
  const int64_t outputColumn = static_cast<int64_t>(inputs.offset[blockIdx.y]) * innerSize;
  const int64_t rowBytes = static_cast<int64_t>(rowSize) * sizeof(T);
  const int64_t inputRowBytes = stride_size > 1
      ? static_cast<int64_t>(inputs.tensorStride[blockIdx.y].tensorStride[0]) * sizeof(T)
      : rowBytes;
  // Bytes from the input's first element to the end of its last row.
  const int64_t inputBytes = (static_cast<int64_t>(outerSize) - 1) * inputRowBytes + rowBytes;

  for (IndexType chunk = blockIdx.x * blockDim.x + threadIdx.x; chunk < nChunks;
       chunk += gridDim.x * blockDim.x) {
    const IndexType row = chunk / chunksPerRow;
    const IndexType chunkInRow = chunk - row * chunksPerRow;
    const int64_t rowBegin = (static_cast<int64_t>(row) * outputRowSize + outputColumn) * sizeof(T);
    const int64_t rowEnd = rowBegin + rowBytes;
    const int64_t chunkBegin = ((rowBegin + outputMisalignment) & ~static_cast<int64_t>(vec_bytes - 1)) -
        outputMisalignment + static_cast<int64_t>(chunkInRow) * vec_bytes;
    const int64_t chunkEnd = chunkBegin + vec_bytes;
    const int64_t begin = chunkBegin > rowBegin ? chunkBegin : rowBegin;
    const int64_t end = chunkEnd < rowEnd ? chunkEnd : rowEnd;
    if (begin >= end) {
      continue;
    }
    const int64_t srcInInput = static_cast<int64_t>(row) * inputRowBytes + (begin - rowBegin);
    const char* src = input + srcInInput;
    const int64_t nBytes = end - begin;
    const unsigned int srcOffset = reinterpret_cast<uintptr_t>(src) % vec_bytes;
    // The next aligned 16 bytes are loaded only if they hold bytes of
    // [begin, end); otherwise the same 16 bytes are reloaded, so that both loads
    // are always in flight together.
    const bool loadNext = srcOffset + nBytes > vec_bytes;
    const int64_t loadBegin = srcInInput - srcOffset;
    if (loadBegin < 0 || loadBegin + (loadNext ? 2 : 1) * vec_bytes > inputBytes) {
      for (int64_t i = 0; i < nBytes; i += sizeof(T)) {
        *reinterpret_cast<T*>(output + begin + i) = *reinterpret_cast<const T*>(src + i);
      }
      continue;
    }
    const char* srcAligned = src - srcOffset;
    const auto lo = at::native::memory::ld_vec<vec_bytes>(srcAligned);
    const auto hi = at::native::memory::ld_vec<vec_bytes>(srcAligned + (loadNext ? vec_bytes : 0));
    const auto bytes = extract_16_bytes(lo, hi, srcOffset);
    if (nBytes == vec_bytes) {
      at::native::memory::st_vec<vec_bytes>(output + chunkBegin, bytes);
    } else {
      // A partial chunk at an end of the row: its first nBytes bytes, one
      // element store each.
#pragma unroll
      for (IndexType i = 0; i < elems_per_vec; ++i) {
        if (static_cast<int64_t>(i * sizeof(T)) < nBytes) {
          T value;
          __builtin_memcpy(&value, reinterpret_cast<const char*>(&bytes) + i * sizeof(T), sizeof(T));
          *reinterpret_cast<T*>(output + begin + i * sizeof(T)) = value;
        }
      }
    }
  }
#else
  CUDA_KERNEL_ASSERT(false && "CatArrayBatchedCopy_unaligned_vectorized is gfx1250-only");
#endif
}

// True if one batch of a cat can take CatArrayBatchedCopy_unaligned_vectorized:
// Contiguous memory format and output, an output pointer that is a multiple of
// sizeof(T) (the kernel places chunk edges on whole elements of the output),
// and every input's chunk count fits the int range so that the unsigned
// grid-stride index cannot wrap.
template <typename T, int batch_size, int stride_size>
static inline bool can_use_unaligned_vectorized_cat(
    const CatArrInputTensorMetadata<T, unsigned int, batch_size, stride_size>& inputs,
    const Tensor& out,
    int64_t dimension,
    int nInputs,
    c10::MemoryFormat memory_format) {
  if (memory_format != c10::MemoryFormat::Contiguous || !out.is_contiguous() ||
      reinterpret_cast<uintptr_t>(out.const_data_ptr()) % sizeof(T) != 0) {
    return false;
  }
  const int64_t innerSize = c10::multiply_integers(out.sizes().begin() + dimension + 1, out.sizes().end());
  const int64_t outerSize = c10::multiply_integers(out.sizes().begin(), out.sizes().begin() + dimension);
  constexpr int64_t elems_per_vec = ALIGNED_VEC_LOAD_BYTES_16 / sizeof(T);
  for (int i = 0; i < nInputs; ++i) {
    const int64_t rowSize = inputs.dimSize[i] * innerSize;
    if (outerSize * (ceil_div(rowSize, elems_per_vec) + 1) > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  return true;
}

// Requires can_use_unaligned_vectorized_cat(inputs, out, dimension, nInputs, memory_format).
template <typename T, int batch_size, int stride_size>
static inline void launch_unaligned_vectorized_cat_kernel(
    char* output,
    const CatArrInputTensorMetadata<T, unsigned int, batch_size, stride_size>& inputs,
    const Tensor& out,
    int64_t dimension,
    int nInputs,
    unsigned int max_elements_per_tensor,
    cudaStream_t stream) {
  const int64_t innerSize = c10::multiply_integers(out.sizes().begin() + dimension + 1, out.sizes().end());
  const int64_t outputRowSize = out.size(dimension) * innerSize;
  const int64_t outerSize = out.numel() / outputRowSize;
  auto [catGrid, applyBlock] = getCatGridContig<T, ALIGNED_VEC_LOAD_BYTES_16>(max_elements_per_tensor, nInputs);
  CatArrayBatchedCopy_unaligned_vectorized<T, unsigned int, batch_size, stride_size><<<catGrid, applyBlock, 0, stream>>>(
      output,
      inputs,
      static_cast<unsigned int>(innerSize),
      static_cast<unsigned int>(outputRowSize),
      static_cast<unsigned int>(outerSize));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Bytes of row-contiguous inputs from which a strided batch that also has other
// inputs is split into the alignment-tolerant kernel and CatArrayBatchedCopy.
constexpr int64_t kMinRowBytesToSplitStridedCat = 2 * 1024 * 1024;

// Row stride, in elements, of a non-empty input viewed as outer rows of
// t.size(dimension) * innerSize contiguous elements (the layout
// CatArrayBatchedCopy_unaligned_vectorized reads), or nullopt if the elements
// of a row are not contiguous or the dims before `dimension` do not collapse to
// one stride.
static inline std::optional<int64_t> cat_input_row_stride(const Tensor& t, int64_t dimension) {
  int64_t expected = 1;
  for (int64_t d = t.dim() - 1; d >= dimension; --d) {
    if (t.size(d) != 1 && t.stride(d) != expected) {
      return std::nullopt;
    }
    expected *= t.size(d);
  }
  // A single row (all outer sizes 1) keeps the row size as its stride.
  int64_t rowStride = expected;
  bool collapsed = false;
  for (int64_t d = dimension - 1; d >= 0; --d) {
    if (t.size(d) == 1) {
      continue;
    }
    if (!collapsed) {
      rowStride = t.stride(d);
      collapsed = true;
    } else if (t.stride(d) != expected) {
      return std::nullopt;
    }
    expected = t.stride(d) * t.size(d);
  }
  return rowStride;
}
#endif

template <typename scalar_t, int batch_size, int stride_size>
void parallel_cat(const Tensor &out, const MaterializedITensorListRef& inputs, int64_t dimension,
                  int nDims, c10::MemoryFormat memory_format) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t *data = (scalar_t *)(out.mutable_data_ptr());
  CatArrInputTensorMetadata<scalar_t, unsigned int, batch_size, stride_size> catMetaData;
  TensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> outputParam;
  // If all batches are contiguous we can call a specialized implementation
  // which requires the input tensor addresses to be aligned to a
  // 16 Byte boundary.

  constexpr bool isContig = stride_size == 1;
  constexpr int alignment = 16;
#ifdef USE_ROCM
  // CatArrayBatchedCopy_contig is faster on most ROCm GPUs (#118685), but on
  // gfx1250 (wave32) its narrow scalar accesses are limited by the memory
  // instruction rate, so gfx1250 takes the same 16-byte vectorized path as CUDA.
  // Also gates CatArrayBatchedCopy_unaligned_vectorized, so it must agree with
  // that kernel's device gate (__gfx1250__, __gfx1250_strict__ or __SPIRV__);
  // a mismatch asserts.
  const bool vectorizedCatAllowed = is_gfx1250_device(out.device().index());
#else
  bool isAligned = true;
  constexpr bool vectorizedCatAllowed = true;
#endif

  // Next, let's initialize the size, stride arrays for the output Tensor.
  // for contig case, we'll canonicalize output strides, so that
  // we don't have arbitrary strides for dims of size 0
  size_t stride0 = 1;
  if (memory_format == c10::MemoryFormat::Contiguous) {
    for (int i = nDims - 1; i >= 0; --i) {
      outputParam.tensorSize[i] = out.size(i);
      if (isContig) {
        outputParam.tensorStride[i] = stride0;
        stride0 *= out.size(i);
      } else {
        outputParam.tensorStride[i] = out.stride(i);
      }
    }
  } else if (memory_format == c10::MemoryFormat::ChannelsLast || memory_format == c10::MemoryFormat::ChannelsLast3d) {
    // permute the semantics of dims from NCHW to NHWC so that the input
    // tensor is now contiguous
    outputParam.tensorSize[0] = out.size(0);
    outputParam.tensorStride[0] = out.stride(0);
    for (int i = 1; i < nDims - 1; ++i) {
      outputParam.tensorSize[i] = out.size(i + 1);
      outputParam.tensorStride[i] = out.stride(i + 1);
    }
    outputParam.tensorSize[nDims - 1] = out.size(1);
    outputParam.tensorStride[nDims - 1] = out.stride(1);
  } else {
    TORCH_CHECK(false, "unsupported memory format");
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();


  // for channels last computing slice size correctly is much more involved, so we never send it
  // on the fully vectorized path
  // we need output stride in cat dimension to be multiple of alignment,
  // if we ever use it to compute offsets
  // for catting in 0th dimension it doesn't matter
  // CatArrayBatchedCopy_vectorized also needs every input 16-byte aligned with
  // a slice that is a multiple of 16 bytes; both are checked per input below.
  bool isInOutAligned = vectorizedCatAllowed && isContig && at::native::memory::get_alignment(data) >= alignment &&
                        memory_format == c10::MemoryFormat::Contiguous && (dimension == 0 ||
                        outputParam.tensorStride[dimension - 1] * sizeof(scalar_t) % alignment == 0);
  unsigned int max_elements_per_tensor = 0;

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (unsigned i = 0; i < inputs.size() ; i += batch_size) {
    for (batchCounter = 0;
          batchCounter < batch_size &&
            (i+batchCounter) < inputs.size();
          ++batchCounter) {
      int64_t dimSize = 0;
      // There is a legacy case where a 1-D empty tensor can be concat with
      // high-dimensional tensor
      if (inputs[i+batchCounter].get().numel() > 0) {
        dimSize = inputs[i+batchCounter].get().size(dimension);
        if (isInOutAligned) {
          auto t = inputs[i+batchCounter].get();
          // similarly to output stride, we cannot trust stride value to
          // determine slice size if the corresponding dimension is 1
          // we have to multiply all the subsequent sizes
          int64_t slice_size = dimension == 0 ? t.numel() : t.sizes()[dimension - 1] != 1 ?
             t.strides()[dimension - 1] : c10::multiply_integers(t.sizes().begin() + dimension, t.sizes().end());
          slice_size *= sizeof(scalar_t);
          isInOutAligned &= (slice_size % alignment == 0);
        }
      }

      catMetaData.input[batchCounter] = (scalar_t*)(inputs[i+batchCounter].get().const_data_ptr());
      catMetaData.offset[batchCounter] = offset;
      catMetaData.dimSize[batchCounter] = dimSize;
      catMetaData.nElements[batchCounter] = inputs[i+batchCounter].get().numel();

#ifndef USE_ROCM
      // If at least one of the inputs is not aligned, we can't call the
      // CatArrayBatchedCopy_alignedK_contig
      isAligned &= is_aligned_vec4(catMetaData.input[batchCounter]);
#endif
      isInOutAligned &= at::native::memory::get_alignment(catMetaData.input[batchCounter]) >= alignment;

      if (stride_size > 1) {
        auto strides = inputs[i+batchCounter].get().strides();
        auto sizes = inputs[i+batchCounter].get().sizes();
        for(int j = 0; j < nDims; j++){
          catMetaData.tensorStride[batchCounter].tensorSize[j] = sizes[j];
          catMetaData.tensorStride[batchCounter].tensorStride[j] = strides[j];
        }
        catMetaData.isContiguous[batchCounter] = false;
      } else {
        catMetaData.isContiguous[batchCounter] = true;
      }

      // Update offset
      offset += dimSize;

      // We need max elements per tensor to compute grid parameters
      max_elements_per_tensor = std::max(max_elements_per_tensor,
        catMetaData.nElements[batchCounter]);
    }

    // Skip if the tensor is empty. Otherwise, the grid dim is invalid
    if (max_elements_per_tensor == 0)
      continue;

#ifdef USE_ROCM
    // On gfx1250 (vectorizedCatAllowed), contiguous cats that miss the 16-byte
    // vectorized gate above take the alignment-tolerant 16-byte kernel instead of
    // CatArrayBatchedCopy_contig. 16-byte types are always 16-byte aligned.
    if constexpr (isContig && sizeof(scalar_t) < ALIGNED_VEC_LOAD_BYTES_16) {
      if (vectorizedCatAllowed && !isInOutAligned &&
          can_use_unaligned_vectorized_cat(catMetaData, out, dimension, batchCounter, memory_format)) {
        launch_unaligned_vectorized_cat_kernel(
            reinterpret_cast<char*>(data), catMetaData, out, dimension, batchCounter, max_elements_per_tensor,
            stream.stream());
        continue;
      }
    }
    // On gfx1250, the inputs of a strided batch whose rows are contiguous
    // (contiguous inputs, slices along the cat dim, broadcast outer dims) take
    // the same kernel, with each input's row stride in tensorStride[0], if they
    // are all the inputs or hold at least kMinRowBytesToSplitStridedCat bytes.
    // CatArrayBatchedCopy below copies the other inputs and skips these, whose
    // nElements become 0.
    if constexpr (!isContig && sizeof(scalar_t) < ALIGNED_VEC_LOAD_BYTES_16) {
      if (vectorizedCatAllowed) {
        auto rowMetaData = catMetaData;
        std::array<bool, batch_size> byRows{};
        unsigned int rowMaxElements = 0;
        int64_t rowBytes = 0;
        bool hasOtherInputs = false;
        for (int k = 0; k < batchCounter; ++k) {
          const auto rowStride = catMetaData.nElements[k] > 0
              ? cat_input_row_stride(inputs[i + k].get(), dimension)
              : std::nullopt;
          if (rowStride.has_value()) {
            byRows[k] = true;
            rowMetaData.tensorStride[k].tensorStride[0] = static_cast<unsigned int>(*rowStride);
            rowMaxElements = std::max(rowMaxElements, catMetaData.nElements[k]);
            rowBytes += static_cast<int64_t>(catMetaData.nElements[k]) * sizeof(scalar_t);
          } else {
            rowMetaData.dimSize[k] = 0;
            hasOtherInputs |= catMetaData.nElements[k] > 0;
          }
        }
        // Splitting a batch adds a launch, which pays off only for enough row bytes.
        if (rowMaxElements > 0 && (!hasOtherInputs || rowBytes >= kMinRowBytesToSplitStridedCat) &&
            can_use_unaligned_vectorized_cat(rowMetaData, out, dimension, batchCounter, memory_format)) {
          launch_unaligned_vectorized_cat_kernel(
              reinterpret_cast<char*>(data), rowMetaData, out, dimension, batchCounter, rowMaxElements,
              stream.stream());
          max_elements_per_tensor = 0;
          for (int k = 0; k < batchCounter; ++k) {
            if (byRows[k]) {
              catMetaData.nElements[k] = 0;
            }
            max_elements_per_tensor = std::max(max_elements_per_tensor, catMetaData.nElements[k]);
          }
          if (max_elements_per_tensor == 0) {
            continue;
          }
        }
      }
    }
#endif

#ifdef USE_ROCM
    // always base grid size on max_elements_per_tensor
    auto [catGrid, applyBlock] = isInOutAligned
        ? getCatGridContig<scalar_t, alignment>(max_elements_per_tensor, batchCounter)
        : getCatGridRocm<scalar_t>(max_elements_per_tensor, batchCounter);
#else
    dim3 applyBlock, catGrid;
    if constexpr (isContig) {
      if (isInOutAligned) {
        std::tie(catGrid, applyBlock) = getCatGridContig<scalar_t, alignment>(
          max_elements_per_tensor, batchCounter);
      } else if (isAligned && sizeof(scalar_t) > 2) {
        std::tie(catGrid, applyBlock) = getCatGridContig<scalar_t, ALIGNED_VEC_LOAD_BYTES_16>(
            max_elements_per_tensor, batchCounter);
      } else if (isAligned && sizeof(scalar_t) == 2) {
        std::tie(catGrid, applyBlock) = getCatGridContig<scalar_t, ALIGNED_VEC_LOAD_BYTES_8>(
            max_elements_per_tensor, batchCounter);
      } else {
        applyBlock = dim3(32 * 16);
        getCatGrid(batchCounter, catGrid);
      }
    } else {
      applyBlock = dim3(32 * 16);
      getCatGrid(batchCounter, catGrid);
    }
#endif
    int nDimsLocal = nDims;
    int32_t trailingSize = 0;
    TensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> kernelOutputParam;
    if (isInOutAligned) {
      // in this case we can and should flatten the tensors after the cat dim
      // we want to view the tensors as if consisting of `alignment`-sized elements
      // however, we might not be able to cleanly divide just the last dim -
      // it might not be the multiple of alignment.
      // however, we know that the full concatted slice is multiple of alignment,
      // so if we flatten all the dims after and including concat dim,
      // it will be divisible by alignment
      // then we need to divide last out size by elems_per_vec,
      // and divide all strides except last by elems_per_vec (last stride is 1 always)
      // for input, we will fix up the sizes and strides in the kernel directly
      kernelOutputParam = outputParam;
      nDimsLocal = dimension + 1;
      constexpr auto elems_per_vec = alignment / sizeof(scalar_t);
      auto out_size = dimension == 0 ? out.numel() : kernelOutputParam.tensorStride[dimension-1];
      kernelOutputParam.tensorSize[dimension] = out_size / elems_per_vec;
      trailingSize = outputParam.tensorStride[dimension];
      kernelOutputParam.tensorStride[dimension] = 1;
      for (int i = 0; i < dimension; ++i) {
        kernelOutputParam.tensorStride[i] /= elems_per_vec;
      }
    }

    int cat_dim = dimension;
    if (memory_format != c10::MemoryFormat::Contiguous) {
      switch (cat_dim) {
      case 0:
        break;
      case 1:
        cat_dim = nDimsLocal - cat_dim;
        break;
      default:
        cat_dim--;
      }
    }
    // Template Declarations for dim = 1, 2, 3, 4
    // #ifdef can't nest in a macro body, so ROCm (vectorized on gfx1250, else
    // contig or generic) and CUDA (the full aligned/vectorized chain) get
    // separate definitions.
#ifdef USE_ROCM
#define HANDLE_CASE(DIMS) \
    if constexpr (isContig) {\
      if (isInOutAligned) {\
        constexpr auto elems_per_vec = alignment / sizeof(scalar_t); \
        CatArrayBatchedCopy_vectorized<scalar_t, unsigned int, DIMS, batch_size, stride_size, alignment, elems_per_vec><<<\
        catGrid, applyBlock, 0, stream.stream()>>>(\
          reinterpret_cast<char*>(data), catMetaData, kernelOutputParam, cat_dim, trailingSize);\
      } else {\
        CatArrayBatchedCopy_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
            catGrid, applyBlock, 0, stream.stream()>>>(\
                data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
      }\
    } else {\
      CatArrayBatchedCopy<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
    }\
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
#define HANDLE_CASE(DIMS) \
    if constexpr (isContig) {\
      if (isInOutAligned) {\
        constexpr auto elems_per_vec = alignment / sizeof(scalar_t); \
        CatArrayBatchedCopy_vectorized<scalar_t, unsigned int, DIMS, batch_size, stride_size, alignment, elems_per_vec><<<\
        catGrid, applyBlock, 0, stream.stream()>>>(\
          reinterpret_cast<char*>(data), catMetaData, kernelOutputParam, cat_dim, trailingSize);\
      } else if (isAligned && sizeof(scalar_t) > 2 && sizeof(scalar_t) <= 8) {\
        CatArrayBatchedCopy_alignedK_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size, ALIGNED_VEC_LOAD_BYTES_16><<<\
            catGrid, applyBlock, 0, stream.stream()>>>(\
                data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
      } else if (isAligned && sizeof(scalar_t) == 2) { \
        CatArrayBatchedCopy_alignedK_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size, ALIGNED_VEC_LOAD_BYTES_8><<<\
            catGrid, applyBlock, 0, stream.stream()>>>(\
                data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
      } else {\
        CatArrayBatchedCopy_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
            catGrid, applyBlock, 0, stream.stream()>>>(\
                data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
      }\
    } else {\
      CatArrayBatchedCopy<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
    }\
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    switch (nDimsLocal) {
      case 1:
        HANDLE_CASE(1);
        break;
      case 2:
        HANDLE_CASE(2);
        break;
      case 3:
        HANDLE_CASE(3);
        break;
      case 4:
        HANDLE_CASE(4);
        break;
    }
#undef HANDLE_CASE
  }
}
// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <unsigned N> struct alignas(N) OpaqueType { char data[N]; };

} // namespace

TORCH_IMPL_FUNC(cat_out_cuda)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();

  // We parallelize the copy if all 6 conditions pass:
  //
  // 1. There is more than one input tensor
  // 2. The out tensor is 32-bit indexable
  // 3. The number of dimensions is <= 4
  // 4. All input tensors are contiguous (output tensor may be non-contig)
  // 5. All input tensors can use 32-bit indexing

  const bool all32BitIndexable = std::all_of(materialized.begin(), materialized.end(),
    [] (const Tensor& t) {
      return at::cuda::detail::canUse32BitIndexMath(t);
    });

  int nDims = materialized[valid].get().dim();

  // We support the contiguous inputs and non-contiguous input (<=4 dims) in different ways
  // For contiguous input, we don't need to pass stride meta data to cuda kernel through constant
  // memory. Therefore, we could pass more inputs to cuda threads.
  // For non-contiguous, we reduce the number of inputs passed to cuda kernel due to the limitation
  // of constant memory.



  if (materialized.size() > 1 &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::cuda::detail::canUse32BitIndexMath(result) &&
      all_contiguous &&
      all32BitIndexable &&
      all_same_dtype) {
      if (isBitsType(result.scalar_type())) {
        AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_cuda", [&]() {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          parallel_cat<dtype, CAT_ARRAY_BATCH_SIZE, 1>(result, materialized, dim, nDims, memory_format);
        });
      } else {
        AT_DISPATCH_V2(
            result.scalar_type(),
            "cat_cuda",
            AT_WRAP([&]() {
              using dtype = OpaqueType<sizeof(scalar_t)>;
              parallel_cat<dtype, CAT_ARRAY_BATCH_SIZE, 1>(
                  result, materialized, dim, nDims, memory_format);
            }),
            AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
            kComplexHalf,
            kBComplex32,
            kHalf,
            kBool,
            kBFloat16,
            AT_EXPAND(AT_FLOAT8_TYPES),
            AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
            kFloat4_e2m1fn_x2);
      }
  } else if (materialized.size() > 1 &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::cuda::detail::canUse32BitIndexMath(result) &&
      nDims <= CAT_ARRAY_MAX_INPUT_DIMS &&
      all32BitIndexable &&
      all_same_dtype &&
      memory_format == c10::MemoryFormat::Contiguous) {
      if (isBitsType(result.scalar_type())) {
        AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_cuda", [&]() {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          parallel_cat<dtype, CAT_ARRAY_BATCH_SIZE/2, CAT_ARRAY_BATCH_SIZE/2>(result, materialized, dim, nDims, memory_format);
        });
      } else {
        AT_DISPATCH_V2(
            result.scalar_type(),
            "cat_cuda",
            AT_WRAP([&]() {
              using dtype = OpaqueType<sizeof(scalar_t)>;
              parallel_cat<
                  dtype,
                  CAT_ARRAY_BATCH_SIZE / 2,
                  CAT_ARRAY_BATCH_SIZE / 2>(
                  result, materialized, dim, nDims, memory_format);
            }),
            AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
            kComplexHalf,
            kBComplex32,
            kHalf,
            kBool,
            kBFloat16,
            kFloat8_e4m3fn,
            kFloat8_e4m3fnuz,
            kFloat8_e5m2,
            kFloat8_e5m2fnuz,
            AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
            // TODO(#146647): extend this to other shell dtypes
            kFloat4_e2m1fn_x2);
      }
  } else {
    int64_t offset = 0;
    for (const Tensor& t : materialized) {
      if (cat_should_skip_tensor(t)) continue;
      int64_t dimSize = t.size(dim);
      Tensor nt = at::narrow(result, dim, offset, dimSize);
      copy_(nt, t);
      offset += dimSize;
    }
  }
}

} // namespace at::native
