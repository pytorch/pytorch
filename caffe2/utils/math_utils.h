#ifndef CAFFE2_UTILS_MATH_UTILS_H_
#define CAFFE2_UTILS_MATH_UTILS_H_

#include "caffe2/core/common.h"

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define MATH_UTILS_DECL inline __host__ __device__
#else
#define MATH_UTILS_DECL inline
#endif

namespace caffe2 {
namespace math {
namespace utils {

MATH_UTILS_DECL bool Not(const bool x) {
  return !x;
}

template <typename T>
MATH_UTILS_DECL T Sign(const T x) {
  return x > 0 ? T(1) : (x < 0 ? T(-1) : T(0));
}

template <typename T>
MATH_UTILS_DECL T Negate(const T x) {
  return -x;
}

template <typename T>
MATH_UTILS_DECL T Inv(const T x) {
  return T(1) / x;
}

template <typename T>
MATH_UTILS_DECL T Square(const T x) {
  return x * x;
}

template <typename T>
MATH_UTILS_DECL T Cube(const T x) {
  return x * x * x;
}

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always
// positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than
// 0x800...
// The casting allows to use one condition instead of two.
MATH_UTILS_DECL bool IsAGeZeroAndALtB(const int a, const int b) {
  return static_cast<unsigned int>(a) < static_cast<unsigned>(b);
}

// Increase the index digits by one based on dims.
CAFFE2_API void IncreaseIndexInDims(const int n, const int* dims, int* index);

// Get index value from dims and index digits.
CAFFE2_API int GetIndexFromDims(const int n, const int* dims, const int* index);

// Checks if the input permutation is an identity permutation;
CAFFE2_API bool IsIdentityPermutation(const int n, const int* perm);

CAFFE2_API bool IsRowwiseReduce(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    int* rows,
    int* cols);

CAFFE2_API bool IsColwiseReduce(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    int* rows,
    int* cols);

CAFFE2_API bool IsBothEndsReduce(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    int* pre,
    int* mid,
    int* nxt);

// Computest the broadcast binary operation dims.
CAFFE2_API void ComputeBroadcastBinaryOpDims(
    const int A_ndim,
    const int* A_dims,
    const int B_ndim,
    const int* B_dims,
    int* A_broadcast_dims,
    int* B_broadcast_dims,
    int* C_broadcast_dims);

CAFFE2_API bool IsRowwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols,
    bool* broadcast_1st);

CAFFE2_API bool IsColwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols,
    bool* broadcast_1st);

CAFFE2_API bool IsBothEndsBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pre,
    int* mid,
    int* nxt,
    bool* broadcast_1st);

CAFFE2_API void ComputeTransposeAxesForReduceOp(
    const int num_dims,
    const int num_reduce_axes,
    const int* reduce_axes,
    int* transpose_axes);

CAFFE2_API void ComputeTransposedStrides(
    const int ndim,
    const int* dims,
    const int* axes,
    int* strides);

} // namespace utils
} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_UTILS_H_
