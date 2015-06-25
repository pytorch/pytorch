#ifndef CAFFE2_UTILS_MATH_H_
#define CAFFE2_UTILS_MATH_H_
// This is a simple translation from the old Caffe math interfaces. We aim to
// still keep it simple, so all platforms would be able to support it fairly
// easily.

extern "C" {
#include "caffe2/utils/cblas.h"
}

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {

namespace math {

template <typename T, class DeviceContext>
void Exp(const int N, const T* x, T* y, DeviceContext* context);
template <typename T, class DeviceContext>
void Log(const int N, const T* x, T* y, DeviceContext* context);
template <typename T, class DeviceContext>
void Sqr(const int N, const T* x, T* y, DeviceContext* context);

template <typename T, class DeviceContext>
void Powx(const int N, const T* a, const T b, T* y, DeviceContext* context);


template <typename T, class DeviceContext>
void Add(const int N, const T* a, const T* b, T* y, DeviceContext* context);
template <typename T, class DeviceContext>
void Sub(const int N, const T* a, const T* b, T* y, DeviceContext* context);
template <typename T, class DeviceContext>
void Mul(const int N, const T* a, const T* b, T* y, DeviceContext* context);
template <typename T, class DeviceContext>
void Div(const int N, const T* a, const T* b, T* y, DeviceContext* context);


// Compute the row-wise max of a N*D matrix X, and write it to a N
// dimensional vector y.
template <typename T, class DeviceContext>
void RowwiseMax(const int N, const int D, const T* x, T* y,
                DeviceContext* context);

// Compute the column-wise max of a N*D matrix X, and write it to a D
// dimensional vector y.
template <typename T, class DeviceContext>
void ColwiseMax(const int N, const int D, const T* x, T* y,
                DeviceContext* context);

// AddToRow and AddToCol adds the corresponding row/col vector x to the matrix y
// of shape MxN.
template <typename T, class DeviceContext>
void AddToRow(const int M, const int N, const T* x, T* y,
              DeviceContext* context);
template <typename T, class DeviceContext>
void AddToCol(const int M, const int N, const T* x, T* y,
              DeviceContext* context);

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename T, class DeviceContext>
void Gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const T* alpha, const T* A,
    const T* B, const T* beta, T* C, DeviceContext* context);

// Gemv always takes in a M*N matrix A, and depending on whether we set TransA
// to Trans, the output is:
// CblasNoTrans: x is an N dim vector and y is an M dim vector.
// CblasTrans:   x is an M dim vector and y is an N dim vector.
template <typename T, class DeviceContext>
void Gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const T* alpha, const T* A, const T* x, const T* beta,
    T* y, DeviceContext* context);

template <typename T, class DeviceContext>
void Set(const int N, const T alpha, T* X, DeviceContext* context);

template <typename T, class DeviceContext>
void RandUniform(const int n, const T a, const T b, T* r,
                 DeviceContext* context);

template <typename T, class DeviceContext>
void RandGaussian(const int n, const T mean, const T std, T* r,
                  DeviceContext* context);

// Dot matrix of vector a and b, and writes the result to a single value y.
template <typename T, class DeviceContext>
void Dot(const int N, const T* a, const T* b, T* y, DeviceContext* context);

// Sum of vector x, and writes the result to a single value y.
template <typename T, class DeviceContext>
void Sum(const int N, const T* x, T* y, DeviceContext* context);

// Select does index selection of the rows a N*D matrix x, and gives the N
// dimensional vector y that contains the selected data.
template <typename T, class DeviceContext>
void Select(const int N, const int D, const T* x, const int* idx, T* y,
            DeviceContext* context);

template <typename T, class DeviceContext>
void Scale(const int N, const T* alpha, const T* x, T* y,
           DeviceContext* context);

template <typename T, class DeviceContext>
void Axpy(const int N, const T* alpha, const T* x, T* y,
          DeviceContext* context);

template <typename T, class DeviceContext>
void Axpby(const int N, const T* alpha, const T* x, const T* b, T* y,
           DeviceContext* context);

template <typename T, class DeviceContext, int order>
void Im2col(const T* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h, const int stride_w, T* data_col,
    DeviceContext* context);

template <typename T, class DeviceContext, int order>
void Col2im(const T* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h, const int stride_w, T* data_im,
    DeviceContext* context);

template <typename T, class DeviceContext>
void CopyMatrix(const int M, const int N, const T* A, const int lda,
                T* B, const int ldb, DeviceContext* context);

}  // namespace math
}  // namespace caffe2


#endif  // CAFFE2_UTILS_MATH_H_
