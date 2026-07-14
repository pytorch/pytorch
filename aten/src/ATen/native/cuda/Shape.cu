#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/Resize.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/TensorShape.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <c10/core/MemoryFormat.h>

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
      ((LT*)reg_data)[0] = const_cast<LT*>((LT*)(data + inputOffset))[0];

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
  bool isAligned = true;
  constexpr int alignment = 16;

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
  bool isInOutAligned = isContig && at::native::memory::get_alignment(data) >= alignment &&
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

#ifdef USE_ROCM
      // On ROCm, CatArrayBatchedCopy_contig is faster
      isAligned = false;
      isInOutAligned = false;
#else
      // If at least one of the inputs is not aligned, we can't call the
      // CatArrayBatchedCopy_alignedK_contig
      isAligned &= is_aligned_vec4(catMetaData.input[batchCounter]);
      isInOutAligned &= at::native::memory::get_alignment(catMetaData.input[batchCounter]) >= alignment;
#endif

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
    // always base grid size on max_elements_per_tensor
    auto [catGrid, applyBlock] = getCatGridRocm<scalar_t>(
          max_elements_per_tensor, batchCounter);
#else
    dim3 applyBlock, catGrid;
    if (isInOutAligned) {
      std::tie(catGrid, applyBlock) = getCatGridContig<scalar_t, alignment>(
        max_elements_per_tensor, batchCounter);
    } else if (isContig && isAligned && sizeof(scalar_t) > 2) {
      std::tie(catGrid, applyBlock) = getCatGridContig<scalar_t, ALIGNED_VEC_LOAD_BYTES_16>(
          max_elements_per_tensor, batchCounter);
    } else if (isContig && isAligned && sizeof(scalar_t) == 2) {
      std::tie(catGrid, applyBlock) = getCatGridContig<scalar_t, ALIGNED_VEC_LOAD_BYTES_8>(
          max_elements_per_tensor, batchCounter);
    } else {
      applyBlock = dim3(32 * 16);
      getCatGrid(batchCounter, catGrid);
    }
#endif
    int32_t trailingSize = 0;
    int nDimsLocal = nDims;
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
#define HANDLE_CASE(DIMS) \
    if (isInOutAligned) {\
      constexpr auto elems_per_vec = alignment / sizeof(scalar_t); \
      CatArrayBatchedCopy_vectorized<scalar_t, unsigned int, DIMS, batch_size, stride_size, alignment, elems_per_vec><<<\
      catGrid, applyBlock, 0, stream.stream()>>>(\
        (char*)data, catMetaData, kernelOutputParam, cat_dim, trailingSize);\
    } else if (isContig && isAligned && sizeof(scalar_t) > 2 && sizeof(scalar_t) <= 8) {\
      CatArrayBatchedCopy_alignedK_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size, ALIGNED_VEC_LOAD_BYTES_16><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
    } else if (isContig && isAligned && sizeof(scalar_t) == 2) { \
      CatArrayBatchedCopy_alignedK_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size, ALIGNED_VEC_LOAD_BYTES_8><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
    } else if (isContig) {\
      CatArrayBatchedCopy_contig<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
    } else {\
      CatArrayBatchedCopy<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, catMetaData, outputParam, cat_dim, outputParam.tensorStride[cat_dim]);\
    }\
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
