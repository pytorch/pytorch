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
}

#undef RADIX_BITS
#undef RADIX_SIZE
#undef RADIX_MASK

#endif // THC_TENSOR_TOPK_CUH
