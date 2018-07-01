#ifndef CAFFE2_UTILS_MATH_UTILS_H_
#define CAFFE2_UTILS_MATH_UTILS_H_

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

// Increase the index digits by one based on dims.
void IncreaseIndexInDims(const int n, const int* dims, int* index);

// Get index value from dims and index digits.
int GetIndexFromDims(const int n, const int* dims, const int* index);

// Checks if the input permutation is an identity permutation;
bool IsIdentityPermutation(const int n, const int* perm);

// Computest the broadcast binary operation dims.
void ComputeBroadcastBinaryOpDims(
    const int A_ndim,
    const int* A_dims,
    const int B_ndim,
    const int* B_dims,
    int* A_broadcast_dims,
    int* B_broadcast_dims,
    int* C_broadcast_dims);

bool IsRowwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pivot,
    bool* broadcast_1st);

bool IsColwiseBroadcastBinaryOp(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    int* pivot,
    bool* broadcast_1st);

void ComputeTransposeAxesForReduceOp(
    const int num_dims,
    const int num_reduce_axes,
    const int* reduce_axes,
    int* transpose_axes);

void ComputeTransposedStrides(
    const int ndim,
    const int* dims,
    const int* axes,
    int* strides);

} // namespace utils
} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_UTILS_H_
