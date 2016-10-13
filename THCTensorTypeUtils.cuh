#ifndef THC_TENSOR_TYPE_UTILS_INC
#define THC_TENSOR_TYPE_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensor.h"
#include "THCTensorInfo.cuh"

/// A utility for accessing THCuda*Tensor types in a generic manner

/// Equivalent to C++11's type_traits std::is_same; used for comparing
/// equality of types. Don't assume the existence of C++11
template <typename T, typename U>
struct SameType {
  static const bool same = false;
};

template <typename T>
struct SameType<T, T> {
  static const bool same = true;
};

template <typename T, typename U>
bool isSameType() {
  return SameType<T, U>::same;
}

template <typename TensorType>
struct TensorUtils {
};

#define TENSOR_UTILS(TENSOR_TYPE, DATA_TYPE, ACC_DATA_TYPE)             \
  template <>                                                           \
  struct TensorUtils<TENSOR_TYPE> {                                     \
    typedef DATA_TYPE DataType;                                         \
    typedef ACC_DATA_TYPE AccDataType;                                  \
                                                                        \
    static TENSOR_TYPE* newTensor(THCState* state);                     \
    static TENSOR_TYPE* newContiguous(THCState* state, TENSOR_TYPE* t); \
    static THLongStorage* newSizeOf(THCState* state, TENSOR_TYPE* t);   \
    static void retain(THCState* state, TENSOR_TYPE* t);                \
    static void free(THCState* state, TENSOR_TYPE* t);                  \
    static void freeCopyTo(THCState* state, TENSOR_TYPE* src,           \
                           TENSOR_TYPE* dst);                           \
    static void resize(THCState* state, TENSOR_TYPE* out,               \
                       THLongStorage* sizes,                            \
                       THLongStorage* strides);                         \
    static void resizeAs(THCState* state, TENSOR_TYPE* dst,             \
                         TENSOR_TYPE* src);                             \
    static DATA_TYPE* getData(THCState* state, TENSOR_TYPE* t);         \
    static ptrdiff_t getNumElements(THCState* state, TENSOR_TYPE* t);        \
    static long getSize(THCState* state, TENSOR_TYPE* t, int dim);      \
    static long getStride(THCState* state, TENSOR_TYPE* t, int dim);    \
    static int getDims(THCState* state, TENSOR_TYPE* t);                \
    static bool isContiguous(THCState* state, TENSOR_TYPE* t);          \
    static int getDevice(THCState* state, TENSOR_TYPE* t);              \
    static void copyIgnoringOverlaps(THCState* state,                   \
                                     TENSOR_TYPE* dst, TENSOR_TYPE* src); \
    /* Determines if the given tensor has overlapping data points (i.e., */ \
    /* is there more than one index into the tensor that references */  \
    /* the same piece of data)? */                                      \
    static bool overlappingIndices(THCState* state, TENSOR_TYPE* t);    \
    /* Can we use 32 bit math for indexing? */                          \
    static bool canUse32BitIndexMath(THCState* state, TENSOR_TYPE* t);  \
  }

TENSOR_UTILS(THCudaByteTensor, unsigned char, long);
TENSOR_UTILS(THCudaCharTensor, char, long);
TENSOR_UTILS(THCudaShortTensor, short, long);
TENSOR_UTILS(THCudaIntTensor, int, long);
TENSOR_UTILS(THCudaLongTensor, long, long);
TENSOR_UTILS(THCudaTensor, float, float);
TENSOR_UTILS(THCudaDoubleTensor, double, double);

#ifdef CUDA_HALF_TENSOR
TENSOR_UTILS(THCudaHalfTensor, half, float);
#endif

#undef TENSOR_UTILS

template <typename TensorType, typename IndexType>
TensorInfo<typename TensorUtils<TensorType>::DataType, IndexType>
getTensorInfo(THCState* state, TensorType* t) {
  IndexType sz[MAX_CUTORCH_DIMS];
  IndexType st[MAX_CUTORCH_DIMS];

  int dims = TensorUtils<TensorType>::getDims(state, t);
  for (int i = 0; i < dims; ++i) {
    sz[i] = TensorUtils<TensorType>::getSize(state, t, i);
    st[i] = TensorUtils<TensorType>::getStride(state, t, i);
  }

  return TensorInfo<typename TensorUtils<TensorType>::DataType, IndexType>(
    TensorUtils<TensorType>::getData(state, t), dims, sz, st);
}

template <typename T>
struct ScalarNegate {
  static __host__ __device__ T to(const T v) { return -v; }
};

template <typename T>
struct ScalarInv {
  static __host__ __device__ T to(const T v) { return ((T) 1) / v; }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct ScalarNegate<half> {
  static __host__ __device__ half to(const half v) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hneg(v);
#else
    return __float2half(-__half2float(v));
#endif
#else
    half out = v;
    out.x ^= 0x8000; // toggle sign bit
    return out;
#endif
  }
};

template <>
struct ScalarInv<half> {
  static __host__ __device__ half to(const half v) {
#ifdef __CUDA_ARCH__
    return __float2half(1.0f / __half2float(v));
#else
    float fv = THC_half2float(v);
    fv = 1.0f / fv;
    return THC_float2half(fv);
#endif
  }
};

inline bool operator==(half a, half b) {
  return a.x == b.x;
}

inline bool operator!=(half a, half b) {
  return a.x != b.x;
}

#endif // CUDA_HALF_TENSOR

#endif // THC_TENSOR_TYPE_UTILS_INC
