#ifndef CAFFE2_UTILS_MATH_UTILS_H_
#define CAFFE2_UTILS_MATH_UTILS_H_

namespace caffe2 {
namespace math {
namespace utils {

// Increase the index digits by one based on dims.
void IncreaseIndexInDims(const int n, const int* dims, int* index);

// Get index value from dims and index digits.
int GetIndexFromDims(const int n, const int* dims, const int* index);

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
