#ifndef THC_TENSOR_KTHVALUE_CUH
#define THC_TENSOR_KTHVALUE_CUH

#include "THCTensorTopK.cuh"

template <typename T, typename IndexType, int Dim>
__global__ void gatherKthValue(TensorInfo<T, IndexType> input,
                               IndexType inputSliceSize,
                               IndexType k,

                               IndexType numInputSlices,
                               IndexType inputWithinSliceStride,

                               TensorInfo<T, IndexType> kthValue,
                               TensorInfo<int64_t, IndexType> indices) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
  __shared__ int smem[32]; // one per each warp, up to warp limit

  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    IndexToOffset<T, IndexType, Dim>::get(slice, input);
  IndexType kthValueSliceStartIndex =
    IndexToOffset<T, IndexType, Dim>::get(slice, kthValue);
  IndexType indicesSliceStartIndex =
    IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  T* inputSliceStart = &input.data[sliceStartIndex];
  T* kthValueSliceStart = &kthValue.data[kthValueSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T kValue = ScalarConvert<int, T>::to(0);
  radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, false>(
    inputSliceStart, k,
    inputSliceSize, inputWithinSliceStride,
    smem, &kValue);

  // Find the index of the k-th highest element
  IndexType kValueIndex = 0;
  bool foundKValue = false;

  for (IndexType i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : ScalarConvert<int, T>::to(0);
    bool isKValue = inRange && (THCNumerics<T>::eq(v, kValue));

    if (isKValue) {
      kValueIndex = i;
      foundKValue = true;
      break;
    }
  }

  if (foundKValue) {
    kthValueSliceStart[0] = kValue;
    indicesSliceStart[0] = ScalarConvert<IndexType, int64_t>::to(kValueIndex);
  }
}

#endif // THC_TENSOR_KTHVALUE_CUH
