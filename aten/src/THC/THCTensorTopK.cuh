#ifndef THC_TENSOR_TOPK_CUH
#define THC_TENSOR_TOPK_CUH

#include <c10/macros/Macros.h>
#include <ATen/native/cuda/SortingRadixSelect.cuh>

using namespace at::native;

template <typename T, typename IndexType, int Dim, bool Order>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK(TensorInfo<T, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`

                           IndexType numInputSlices,
                           IndexType inputWithinSliceStride,

                           TensorInfo<T, IndexType> topK,
                           IndexType numTopKSlices,
                           IndexType topKWithinSliceStride,

                           TensorInfo<int64_t, IndexType> indices,
                           IndexType indicesWithinSliceStride) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
#ifdef __HIP_PLATFORM_HCC__
  __shared__ int smem[64];
#else
  __shared__ int smem[32]; // one per each warp, up to warp limit
#endif

  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    IndexToOffset<T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T topKValue = ScalarConvert<int, T>::to(0);
  radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>(
    inputSliceStart, outputSliceSize,
    inputSliceSize, inputWithinSliceStride,
    smem, &topKValue);

  // Every value that is strictly less/greater than `pattern`
  // (depending on sort dir) in sorted int format is in the top-K.
  // The top-K value itself might not be unique.
  //
  // Since there are a variable number of elements that we see that
  // are within the top-k, we don't know at what index to write out
  // the resulting values.
  // In order to get this, we perform an exclusive prefix sum of
  // `hasTopK`. This will return the resulting index into which we
  // need to write the result, if a thread has a result.

  // All threads need to participate in the loop and the prefix sum,
  // but not necessarily in the load; hence loop bounds being rounded
  // up to a multiple of the block dim.
  IndexType numIterations = THCRoundUp(inputSliceSize, (IndexType) blockDim.x);
  IndexType writeIndexStart = 0;

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : ScalarConvert<int, T>::to(0);
    bool hasTopK;
    if (Order) {
      hasTopK = inRange && (THCNumerics<T>::gt(v, topKValue));
    } else {
      hasTopK = inRange && (THCNumerics<T>::lt(v, topKValue));
    }

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    writeIndexStart += carry;
  }

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize -
  // writeIndexStart. There might be more than that number available,
  // in which case we have to choose the first seen set. We do this
  // via a prefix sum to calculate indices for writing results.
  CUDA_KERNEL_ASSERT(outputSliceSize >= writeIndexStart);
  IndexType topKRemaining = (outputSliceSize - writeIndexStart);

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v =
      inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : ScalarConvert<int, T>::to(0);
    bool hasTopK = inRange && (THCNumerics<T>::eq(v, topKValue));

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    if (carry >= topKRemaining) {
      break;
    }

    topKRemaining -= carry;
    writeIndexStart += carry;
  }
}

#undef RADIX_BITS
#undef RADIX_SIZE
#undef RADIX_MASK

#endif // THC_TENSOR_TOPK_CUH
