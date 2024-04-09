#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bincount_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/histc_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace cuda {
#define RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD 8
#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
       i += gridDim.x * blockDim.x)

/*
  Memory types used for the 3 histogram implementations.
  See `CUDA_tensor_histogram` below.
 */
enum class CUDAHistogramMemoryType { SHARED, GLOBAL };
namespace {
template <typename input_t, typename IndexType>
__device__ static IndexType getBin(
    input_t bVal,
    at::acc_type<input_t, /*is_cuda=*/true> minvalue,
    at::acc_type<input_t, /*is_cuda=*/true> maxvalue,
    int64_t nbins) {
  IndexType bin = (int)(((bVal - minvalue)) * nbins / (maxvalue - minvalue));
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end) the last bin is inclusive at both, i.e. [start, end], in
  // order to include maxvalue if exists therefore when bin == nbins, adjust bin
  // to the last bin
  if (bin == nbins)
    bin -= 1;
  return bin;
}
}

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename output_t,
    typename input_t,
    typename IndexType,
    int ADims,
    int PDims,
    int BDims,
    CUDAHistogramMemoryType MemoryType,
    typename Op>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void kernelHistogram1D(
    detail::TensorInfo<output_t, IndexType> a, /* output */
    detail::TensorInfo<output_t, IndexType> p, /* partial output */
    detail::TensorInfo<const input_t, IndexType> b, /* input */
    int64_t nbins,
    at::acc_type<input_t, /*is_cuda=*/true> minvalue,
    at::acc_type<input_t, /*is_cuda=*/true> maxvalue,
    IndexType totalElements,
    Op getOp) {
  extern __shared__ unsigned char my_smem[];
  output_t* smem = nullptr;

  if (MemoryType == CUDAHistogramMemoryType::SHARED) {
    ////////////////////////// Shared memory //////////////////////////
    // atomically add to block specific shared memory
    // then atomically add to the global output tensor
    smem = reinterpret_cast<output_t*>(my_smem);
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      smem[i] = 0;
    }
    __syncthreads();
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          detail::IndexToOffset<const input_t, IndexType, BDims>::get(linearIndex, b);
      const auto bVal = b.data[bOffset];
      if (bVal >= minvalue && bVal <= maxvalue) {
        // Use value at `b` as an offset of `smem`
        const IndexType bin =
            getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
        gpuAtomicAddNoReturn(&smem[bin], getOp(linearIndex));
      }
    }
    __syncthreads();
    // NOTE: atomically update output bin count.
    //   Atomic update is imp since __syncthread() will only synchronize threads
    //   in a given block, not across blocks.
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      const IndexType aOffset =
          detail::IndexToOffset<output_t, IndexType, ADims>::get(i, a);
      gpuAtomicAddNoReturn(&a.data[aOffset], smem[i]);
    }

  } else {
    ////////////////////////// Global memory //////////////////////////
    // atomically add to the output tensor
    // compute histogram for the block
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          detail::IndexToOffset<const input_t, IndexType, BDims>::get(linearIndex, b);
      const auto bVal = b.data[bOffset];
      if (bVal >= minvalue && bVal <= maxvalue) {
        // Use value at `b` as an offset of `a`
        const IndexType bin =
            getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
        const IndexType aOffset =
            detail::IndexToOffset<output_t, IndexType, ADims>::get(bin, a);
        gpuAtomicAddNoReturn(&a.data[aOffset], getOp(linearIndex));
      }
    }
  }
}

#define HANDLE_CASE(MEMORY_TYPE, WEIGHTS_OP, SHARED_MEM)                 \
  kernelHistogram1D<                                                     \
      output_t,                                                          \
      input_t,                                                           \
      IndexType,                                                         \
      1,                                                                 \
      2,                                                                 \
      -1,                                                                \
      MEMORY_TYPE><<<grid, block, SHARED_MEM, getCurrentCUDAStream()>>>( \
      aInfo,                                                             \
      pInfo,                                                             \
      bInfo,                                                             \
      nbins,                                                             \
      minvalue,                                                          \
      maxvalue,                                                          \
      totalElements,                                                     \
      WEIGHTS_OP);                                                       \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define HANDLE_SWITCH_CASE(mType, getOp)                                   \
  switch (mType) {                                                         \
    case CUDAHistogramMemoryType::SHARED:                                  \
      HANDLE_CASE(CUDAHistogramMemoryType::SHARED, getOp, sharedMem);      \
      break;                                                               \
    default:                                                               \
      HANDLE_CASE(CUDAHistogramMemoryType::GLOBAL, getOp, 0);              \
  }

/*
  Calculate the frequency of the input values.

  `a` contains the final output or the histogram.
  Input `b` is assumed to be 1-D non-negative int array.
  `c` optionally contains the weight vector.
  See `help torch.bincount` for details on the math.

  3 implementations based of input size and memory usage:
    case: enough shared mem
        SHARED: Each block atomically adds to it's own **shared** hist copy,
        then atomically updates the global tensor.
    case: no enough shared mem
        GLOBAL: all threads atomically update to a single **global** hist copy.
 */
template <typename output_t, typename input_t, bool HasWeights>
bool CUDA_tensor_histogram(
    at::Tensor a, /* output */
    at::Tensor b, /* input */
    at::Tensor c, /* weights(optional) */
    int64_t nbins,
    at::acc_type<input_t, /*is_cuda=*/true> minvalue,
    at::acc_type<input_t, /*is_cuda=*/true> maxvalue,
    TensorArgType aType = TensorArgType::ReadWrite,
    TensorArgType bType = TensorArgType::ReadOnly,
    TensorArgType cType = TensorArgType::ReadOnly) {
  checkBackend("CUDA_tensor_histogram", {a, b}, Backend::CUDA);
  if (HasWeights) {
    checkBackend("CUDA_tensor_histogram", {c}, Backend::CUDA);
  }
  auto totalElements = b.numel();

  if (totalElements == 0) {
    return false;
  }

  const dim3 block = getApplyBlock();
  dim3 grid;
  auto curDevice = current_device();
  if (curDevice == -1 || !getApplyGrid(totalElements, grid, curDevice)) {
    return false;
  }

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(output_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
  if (sharedMem < maxSharedMem) {
    // Solve equations:
    // (1) #(smem atomicAdd per SM) = totalElements / min(grid.x, #SM)
    // (2) #(gmem atomicAdd) = grid.x * nbins
    // (3) RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD = #(gmem atomicAdd) / #(smem atomicAdd per SM)
    unsigned optimalGrid = ceil_div<size_t>(RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD * totalElements,
                                            nbins * getCurrentDeviceProperties()->multiProcessorCount);
    if (optimalGrid < (unsigned)getCurrentDeviceProperties()->multiProcessorCount) {
      optimalGrid = 1 + (unsigned)std::sqrt(RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD * totalElements / nbins);
    }
    auto optimalSteps = ceil_div<size_t>(totalElements, optimalGrid * block.x);
    optimalGrid = ceil_div<size_t>(totalElements, optimalSteps * block.x);
    grid.x = std::min(grid.x, optimalGrid);
    memType = CUDAHistogramMemoryType::SHARED;
  }

  using IndexType = int64_t;
  auto aInfo = detail::getTensorInfo<output_t, IndexType>(a);
  auto bInfo = detail::getTensorInfo<const input_t, IndexType>(b);
  detail::TensorInfo<output_t, IndexType> pInfo(nullptr, 0, {}, {});

  if (HasWeights) {
    auto cInfo = detail::getTensorInfo<output_t, IndexType>(c);
    const auto getWeightsOp = [cInfo] __device__(IndexType cIndex) {
      const IndexType cOffset =
          detail::IndexToOffset<output_t, IndexType, 1>::get(cIndex, cInfo);
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
#undef RATIO_OF_GMEM_ATOMIC_ADD_TO_SMEM_ATOMIC_ADD
} // namespace cuda

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
  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros(
        {minlength},
        kLong,
        c10::nullopt /* layout */,
        kCUDA,
        c10::nullopt /* pin_memory */);
  }
  if (self.dim() != 1 ||
      (!std::is_same<input_t, uint8_t>::value &&
       *self.min().cpu().const_data_ptr<input_t>() < 0)) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
    AT_ERROR("weights should be 1-d and have the same length as input");
  }

  const int64_t nbins =
      std::max(self.max().item<input_t>() + (int64_t)1, minlength);

  // we are using acc_type for the bounds, in particular int64_t for integers
  // in order to avoid overflows (e.g. using 256 bins for dtype uint8)
  using bounds_t = at::acc_type<input_t, /*is_cuda=*/true>;
  const bounds_t minvalue = 0;
  const bounds_t maxvalue = nbins;
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = at::zeros(
        {nbins},
        optTypeMetaToScalarType(weights.options().dtype_opt()),
        weights.options().layout_opt(),
        weights.options().device_opt(),
        weights.options().pinned_memory_opt());
    cuda::CUDA_tensor_histogram<weights_t, input_t, true>(
        output, self, weights, nbins, minvalue, maxvalue);
  } else {
    output = at::zeros(
        {nbins},
        kLong,
        c10::nullopt /* layout */,
        DeviceType::CUDA,
        c10::nullopt /* pin_memory */);
    cuda::CUDA_tensor_histogram<int64_t, input_t, false>(
        output, self, weights, nbins, minvalue, maxvalue);
  }
  return output;
}

///////////////// histc /////////////////
template <typename input_t>
Tensor _histc_cuda_template(
    const Tensor& self,
    int64_t nbins,
    at::acc_type<input_t, /*is_cuda=*/true> min,
    at::acc_type<input_t, /*is_cuda=*/true> max) {
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output = at::zeros(
      {nbins},
      self.scalar_type(),
      c10::nullopt /* layout */,
      DeviceType::CUDA,
      c10::nullopt /* pin_memory */);
  input_t minvalue = min;
  input_t maxvalue = max;
  if (min == max && self.numel() > 0) {
    minvalue = *self.min().cpu().const_data_ptr<input_t>();
    maxvalue = *self.max().cpu().const_data_ptr<input_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

#if !defined(USE_ROCM)
  TORCH_CHECK(
      !(at::_isinf(minvalue) || at::_isinf(maxvalue) ||
        at::_isnan(minvalue) || at::_isnan(maxvalue)),
      "range of [",
      minvalue,
      ", ",
      maxvalue,
      "] is not finite");
#else
  TORCH_CHECK(
      !(std::isinf(minvalue) || std::isinf(maxvalue) || std::isnan(minvalue) ||
        std::isnan(maxvalue)),
      "range of [",
      minvalue,
      ", ",
      maxvalue,
      "] is not finite");
#endif
  TORCH_CHECK(minvalue < maxvalue, "max must be larger than min");

  cuda::CUDA_tensor_histogram<input_t, input_t, false>(
      output, self, Tensor(), nbins, minvalue, maxvalue);
  return output;
}
} // namespace

namespace native {
Tensor _bincount_cuda(
    const Tensor& self, const c10::optional<Tensor>& weights_opt,
    int64_t minlength) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  if (weights_opt.has_value()) {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic if weights are given, because of floating point
    // atomicAdd usage
    globalContext().alertNotDeterministic("_bincount_cuda");
  }
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cuda", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cuda_template<scalar_t, float>(self, weights, minlength);
    return _bincount_cuda_template<scalar_t, double>(
        self, weights.to(kDouble), minlength);
  });
}

Tensor _histc_cuda(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("_histc_cuda");
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    using bounds_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
    return _histc_cuda_template<scalar_t>(
        self, nbins, min.to<bounds_t>(), max.to<bounds_t>());
  });
}

Tensor& _histc_out_cuda(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max, Tensor& result) {
  auto ret = _histc_cuda(self, bins, min, max);
  resize_output(result, ret.sizes());
  result.copy_(ret);
  return result;
}
} // namespace native
} // namespace at
