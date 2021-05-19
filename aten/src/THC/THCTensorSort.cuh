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

#endif // THC_TENSORSORT_CUH
