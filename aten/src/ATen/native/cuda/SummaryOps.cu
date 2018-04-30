#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

namespace at {

namespace cuda { namespace detail {
#define MIN_NUMBER_BINS_FOR_GLOBAL_MEM 5000
#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
       i += gridDim.x * blockDim.x)

/*
  Memory types used for the 3 histogram implementations.
  See `CUDA_tensor_histogram` below.
 */
enum class CUDAHistogramMemoryType { MULTI_BLOCK, SHARED, GLOBAL };

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int ADims,
    int PDims,
    int BDims,
    CUDAHistogramMemoryType MemoryType = CUDAHistogramMemoryType::MULTI_BLOCK,
    typename Op>
__global__ void kernelHistogram1D(
    detail::TensorInfo<scalar1, IndexType> a, /* output */
    detail::TensorInfo<scalar1, IndexType> p, /* partial output */
    detail::TensorInfo<scalar2, IndexType> b, /* input */
    int binsize,
    IndexType totalElements,
    Op getOp) {
  extern __shared__ unsigned char my_smem[];
  scalar1* smem = nullptr;

  if (MemoryType == CUDAHistogramMemoryType::SHARED) {
    ////////////////////////// Shared memory //////////////////////////
    // atomically add to block specific shared memory
    // then atomically add to the global output tensor
    smem = reinterpret_cast<scalar1*>(my_smem);
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      smem[i] = 0;
    }
    __syncthreads();
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b);
      // Use value at `b` as an offset of `smem`
      const IndexType pOffset = b.data[bOffset] / binsize;
      atomicAdd(&smem[pOffset], getOp(linearIndex));
    }
    __syncthreads();
    // NOTE: atomically update output bin count.
    //   Atomic update is imp since __syncthread() will only synchronize threads
    //   in a given block, not across blocks.
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      const IndexType aOffset =
          detail::IndexToOffset<scalar1, IndexType, ADims>::get(i, a);
      atomicAdd(&a.data[aOffset], smem[i]);
    }

  } else if (MemoryType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    ////////////////////////// Multi Block memory //////////////////////////
    // atomically add to block specific global tensor
    // then atomically add to the global output tensor
    // compute histogram for the block
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b);
      const auto bVal = b.data[bOffset];
      // Use value at `b` as an offset of `p`
      const IndexType pIdx = p.strides[0] * blockIdx.x + bVal / binsize;
      const IndexType pOffset =
          detail::IndexToOffset<scalar1, IndexType, PDims>::get(pIdx, p);
      atomicAdd(&p.data[pOffset], getOp(linearIndex));
    }
    __syncthreads();
    // NOTE: atomically update output bin count.
    //   Atomic update is imp since __syncthread() will only synchronize threads
    //   in a given block, not across blocks.
    const IndexType pIdx = p.strides[0] * blockIdx.x;
    const IndexType pOffset =
        detail::IndexToOffset<scalar1, IndexType, PDims>::get(pIdx, p);
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      const IndexType aOffset =
          detail::IndexToOffset<scalar1, IndexType, ADims>::get(i, a);
      atomicAdd(&a.data[aOffset], p.data[pOffset + i]);
    }

  } else {
    ////////////////////////// Global memory //////////////////////////
    // atomically add to the output tensor
    // compute histogram for the block
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b);
      const auto bVal = b.data[bOffset];
      // Use value at `b` as an offset of `a`
      const IndexType aIdx = bVal / binsize;
      const IndexType aOffset =
          detail::IndexToOffset<scalar1, IndexType, ADims>::get(aIdx, a);
      atomicAdd(&a.data[aOffset], getOp(linearIndex));
    }
  }
}

#define HANDLE_CASE(MEMORY_TYPE, WEIGHTS_OP)                               \
  kernelHistogram1D<scalar1, scalar2, IndexType, 1, 2, 1, MEMORY_TYPE>     \
      <<<grid,                                                             \
         block,                                                            \
         (MEMORY_TYPE == CUDAHistogramMemoryType::SHARED) ? sharedMem : 0, \
         at::globalContext().getCurrentCUDAStream()>>>(                    \
          aInfo, pInfo, bInfo, binsize, totalElements, WEIGHTS_OP);        \
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");

#define HANDLE_SWITCH_CASE(mType, getOp)                                      \
  switch (mType) {                                                            \
    case CUDAHistogramMemoryType::SHARED:                                     \
      HANDLE_CASE(CUDAHistogramMemoryType::SHARED, getOp);                    \
      break;                                                                  \
    case CUDAHistogramMemoryType::MULTI_BLOCK:                                \
      HANDLE_CASE(CUDAHistogramMemoryType::MULTI_BLOCK, getOp);               \
      break;                                                                  \
    default:                                                                  \
      std::cerr << "WARNING: Potentially slow. "                              \
                   "CUDA_tensor_histogram with nbins = "                      \
                << nbins << " uses global memory with atomics." << std::endl; \
      HANDLE_CASE(CUDAHistogramMemoryType::GLOBAL, getOp);                    \
  }

/*
  Calculate the frequency of the input values.

  `a` contains the final output or the histogram.
  Input `b` is assumed to be 1-D non-negative int array.
  `c` optionally contains the weight vector.
  See `help torch.bincount` for details on the math.

  3 implementations based of input size and memory usage:
    case: #bins < blockDim.x
        SHARED: Each block atomically adds to it's own **shared** hist copy,
        then atomically updates the global tensor.
    case: blockDim.x <= #bins < MIN_NUMBER_BINS_FOR_GLOBAL_MEM
        MULTI_BLOCK: Each block atomically adds to it's own **global** hist
        copy, then atomically updates the global tensor.
    case: MIN_NUMBER_BINS_FOR_GLOBAL_MEM <= #bins
        GLOBAL: all threads atomically update to a single **global** hist copy.
 */
template <typename scalar1, typename scalar2, bool HasWeights>
bool CUDA_tensor_histogram(
    at::Tensor a, /* output */
    at::Tensor b, /* input */
    at::Tensor c, /* weights(optional) */
    int64_t nbins,
    int binsize,
    TensorArgType aType = TensorArgType::ReadWrite,
    TensorArgType bType = TensorArgType::ReadOnly,
    TensorArgType cType = TensorArgType::ReadOnly) {
  checkBackend("CUDA_tensor_histogram", {a, b}, Backend::CUDA);
  if (HasWeights) {
    checkBackend("CUDA_tensor_histogram", {c}, Backend::CUDA);
  }
  auto totalElements = b.size(0);

  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(totalElements, grid)) {
    return false;
  }
#if CUDA_VERSION < 9000
  grid.x = std::min(
      (unsigned int)at::globalContext()
              .getCurrentDeviceProperties()
              ->multiProcessorCount *
          AT_APPLY_BLOCKS_PER_SM,
      grid.x);
#endif

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::SHARED;
  auto maxSharedMem =
      at::globalContext().getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(scalar1) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
  if (nbins < block.x && sharedMem < maxSharedMem) {
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (nbins < MIN_NUMBER_BINS_FOR_GLOBAL_MEM) {
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  } else {
    memType = CUDAHistogramMemoryType::GLOBAL;
  }
  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = detail::getTensorInfo<scalar1, IndexType>(a);
  auto bInfo = detail::getTensorInfo<scalar2, IndexType>(b);
  detail::TensorInfo<scalar1, IndexType> pInfo = aInfo;
  Tensor partial_output;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    partial_output = a.type().zeros({grid.x, nbins});
    pInfo = detail::getTensorInfo<scalar1, IndexType>(partial_output);
  }

  if (HasWeights) {
    auto cInfo = detail::getTensorInfo<scalar1, IndexType>(c);
    const auto getWeightsOp = [cInfo] __device__(IndexType cIndex) {
      const IndexType cOffset =
          detail::IndexToOffset<scalar1, IndexType, 1>::get(cIndex, cInfo);
      return cInfo.data[cOffset];
    };
    HANDLE_SWITCH_CASE(memType, getWeightsOp)
  } else {
    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    HANDLE_SWITCH_CASE(memType, getDummyOp)
  }
  return true;
}

#undef HANDLE_CASE
#undef HANDLE_SWITCH_CASE
#undef FOR_KERNEL_LOOP
#undef MIN_NUMBER_BINS_FOR_GLOBAL_MEM
}} // namespace cuda::detail

namespace {
///////////////// bincount /////////////////
template <typename input_t, typename weights_t>
Tensor _bincount_cuda_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() != 1 || self.numel() == 0 ||
      (!std::is_same<input_t, uint8_t>::value &&
       *self.min().toBackend(kCPU).data<input_t>() < 0)) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    AT_ERROR("input and weights should have the same length");
  }

  auto maxScalarGpu = Scalar(self.max());
  auto nbins = maxScalarGpu.local().to<int64_t>() + 1L;
  nbins = std::max(nbins, minlength);
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = zeros(weights.type(), {nbins});
    auto ret = cuda::detail::CUDA_tensor_histogram<weights_t, input_t, true>(
        output, self, weights, nbins, 1);
  } else {
    output = zeros(CUDA(kLong), {nbins});
    auto ret = cuda::detail::CUDA_tensor_histogram<int64_t, input_t, false>(
        output, self, weights, nbins, 1);
  }
  return output;
}
} // namespace

namespace native {
Tensor
_bincount_cuda(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.type(), "bincount", [&] {
    const auto scalar = weights.type().scalarType();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cuda_template<scalar_t, float>(self, weights, minlength);
    return _bincount_cuda_template<scalar_t, double>(
        self, weights.toType(CUDA(kDouble)), minlength);
  });
}

} // namespace native
} // namespace at
