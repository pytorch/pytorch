#include <THC/THC.h>
#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <TH/THHalf.h>
#include <THC/THCApply.cuh>
#include <THC/THCReduce.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCAtomics.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm> // for std::min
#include <c10/macros/Macros.h>
#include <ATen/WrapDimUtils.h>

template <int Dims, typename T, typename IndexType>
__device__ __forceinline__ IndexType indexToOffset(
    const TensorInfo<T, IndexType>& info,
    int64_t index,
    IndexType size)
{
  IndexType linearIndex = static_cast<IndexType>(index);
  CUDA_KERNEL_ASSERT(linearIndex < size && linearIndex >= -size);
  if (linearIndex < 0) {
    linearIndex += size;
  }
  return IndexToOffset<T, IndexType, Dims>::get(linearIndex, info);
}

struct WrapIndexOp {
  WrapIndexOp(int64_t size) : size(size) {}

  __device__ __forceinline__ void operator()(int64_t* out, int64_t* in) {
    auto idx = *in;
    CUDA_KERNEL_ASSERT(idx < size && idx >= -size);
    *out = idx < 0 ? idx + size : idx;
  }

  int64_t size;
};

template <typename T, typename IndexType, int Dims>
struct TensorPutOp {
  TensorPutOp(TensorInfo<T, IndexType> info, IndexType numel, int64_t*, int64_t*)
    : info(info), numel(numel) {}

  __device__ __forceinline__ void operator()(T* value, int64_t* index) {
    auto offset = indexToOffset<Dims>(info, *index, numel);
    info.data[offset] = *value;
  }

  const TensorInfo<T, IndexType> info;
  IndexType numel;
};

template <typename T, typename IndexType, int Dims>
struct TensorPutAccumulateOp {
  TensorPutAccumulateOp(TensorInfo<T, IndexType> info, IndexType numel, int64_t* start, int64_t* end)
    : info(info), numel(numel), start(start), end(end) {}

  __device__ __forceinline__ void operator()(T* value, int64_t* index) {
    if (index == start || *index != *(index - 1)) {
      int64_t linear_index = *index;
      auto offset = indexToOffset<Dims>(info, linear_index, numel);
      do {
        info.data[offset] = THCNumerics<T>::add(info.data[offset], *value);
        index++;
        value++;
      } while (index != end && *index == linear_index);
    }
  }

  const TensorInfo<T, IndexType> info;
  IndexType numel;
  int64_t* start;
  int64_t* end;
};


template<typename IndexType, typename T, template<class, class, int> class Op, typename TensorType>
void dispatchTakePutImpl(THCState *state, TensorType *a, TensorType *b, THCudaLongTensor *index) {
  // These are only valid if index is contiguous
  auto start = THCudaLongTensor_data(state, index);
  auto end = start + THCudaLongTensor_numel(state, index);

  auto aInfo = getTensorInfo<T, TensorType, IndexType>(state, a);
  aInfo.collapseDims();
  auto numel = THCTensor_nElement(state, a);
  if (aInfo.isContiguous()) {
    auto op = Op<T, IndexType, -2>(aInfo, numel, start, end);
    THC_pointwiseApply2<T, int64_t>(state, b, index, op);
  } else {
    auto op = Op<T, IndexType, -1>(aInfo, numel, start, end);
    THC_pointwiseApply2<T, int64_t>(state, b, index, op);
  }
}

template<typename T, template<class, class, int> class Op, typename TensorType>
void dispatchTakePut(THCState *state, TensorType *a, TensorType *b, THCudaLongTensor *index) {
  if (THCTensor_canUse32BitIndexMath(state, a, INT_MAX)) {
    dispatchTakePutImpl<int32_t, T, Op>(state, a, b, index);
  } else {
    dispatchTakePutImpl<int64_t, T, Op>(state, a, b, index);
  }
}

#include <THC/generic/THCTensorIndex.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorIndex.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensorIndex.cu>
#include <THC/THCGenerateBFloat16Type.h>
