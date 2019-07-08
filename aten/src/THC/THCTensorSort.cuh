#ifndef THC_TENSORSORT_CUH
#define THC_TENSORSORT_CUH

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCSortUtils.cuh>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorTypeUtils.cuh>

#include <THC/THCThrustAllocator.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if CUDA_VERSION >= 7000 || defined(__HIP_PLATFORM_HCC__)
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename T, bool handleNaN = false>
struct ThrustGTOp {
  __device__ bool operator()(const T& lhs, const T& rhs) const {
    return (handleNaN && THCNumerics<T>::isnan(lhs) && !THCNumerics<T>::isnan(rhs)) || THCNumerics<T>::gt(lhs, rhs);
  }
};

template <typename T, bool handleNaN = false>
struct ThrustLTOp {
  __device__ bool operator()(const T& lhs, const T& rhs) const {
    return (handleNaN && THCNumerics<T>::isnan(rhs) && !THCNumerics<T>::isnan(lhs)) || THCNumerics<T>::lt(lhs, rhs);
  }
};

template <typename T, typename IndT, bool handleNaN = true>
struct ThrustSliceGTOp {
ThrustSliceGTOp(int64_t size) : sliceSize(size) {}
  __device__ bool operator()(const thrust::tuple<int64_t, T>& lhs, const thrust::tuple<int64_t, T>& rhs) const {
    IndT segA = (IndT)thrust::get<0>(lhs) / sliceSize;
    IndT segB = (IndT)thrust::get<0>(rhs) / sliceSize;
    if (segA != segB)
        return segA < segB;
    else
        return (handleNaN && THCNumerics<T>::isnan(thrust::get<1>(lhs)) && !THCNumerics<T>::isnan(thrust::get<1>(rhs))) || THCNumerics<T>::gt(thrust::get<1>(lhs), thrust::get<1>(rhs));
  }
  const IndT sliceSize;
};

template <typename T, typename IndT, bool handleNaN = true>
struct ThrustSliceLTOp {
ThrustSliceLTOp(int64_t size) : sliceSize(size) {}
  __device__ bool operator()(const thrust::tuple<int64_t, T>& lhs, const thrust::tuple<int64_t, T>& rhs) const {
    IndT segA = (IndT)thrust::get<0>(lhs) / sliceSize;
    IndT segB = (IndT)thrust::get<0>(rhs) / sliceSize;
    if (segA != segB)
        return segA < segB;
    else
        return (handleNaN && THCNumerics<T>::isnan(thrust::get<1>(rhs)) && !THCNumerics<T>::isnan(thrust::get<1>(lhs))) || THCNumerics<T>::lt(thrust::get<1>(lhs), thrust::get<1>(rhs));
  }
  const IndT sliceSize;
};




// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename IndexType, int Dim>
__global__ void
fillSliceWithIndex(TensorInfo<int64_t, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride) {
  IndexType slice = getLinearBlockId<IndexType>();

  if (slice >= totalSlices) {
    return;
  }

  const uint64_t offset =
    IndexToOffset<int64_t, IndexType, Dim>::get(slice, out);
  int64_t* base = &out.data[offset];

  for (int64_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = i;
  }
}

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(int64_t size) : sliceSize(size) {}

  __device__ inline void operator()(int64_t& v) const {
    v = v % sliceSize;
  }

  const int64_t sliceSize;
};

void THCudaLongTensor_fillSliceWithIndex(THCState* state,
                                         THCudaLongTensor* t,
                                         int dim);
#endif // THC_TENSORSORT_CUH
