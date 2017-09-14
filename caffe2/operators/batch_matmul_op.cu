#include "caffe2/operators/batch_matmul_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

#if __CUDACC_VER_MAJOR__ >= 8
// CUDA 8 introduced a cublasSgemmStridedBatched function that allows us
// to carry out batched sgemm more efficiently. This is the specialized
// version that implements this.
template <>
bool BatchMatMulOp<float, CUDAContext, DefaultEngine>::RunOnDevice() {
  const auto& A = Input(0);
  const auto& B = Input(1);
  auto* Y = Output(0);

  CAFFE_ENFORCE_EQ(A.ndim(), 3);
  CAFFE_ENFORCE_EQ(B.ndim(), 3);
  CAFFE_ENFORCE_EQ(A.dim32(0), B.dim32(0));

  int a_dim0, a_dim1, b_dim0, b_dim1;

  if (trans_a_) {
    a_dim0 = A.dim32(2);
    a_dim1 = A.dim32(1);
  } else {
    a_dim0 = A.dim32(1);
    a_dim1 = A.dim32(2);
  }

  if (trans_b_) {
    b_dim0 = B.dim32(2);
    b_dim1 = B.dim32(1);
  } else {
    b_dim0 = B.dim32(1);
    b_dim1 = B.dim32(2);
  }

  // Error checking
  CAFFE_ENFORCE(
      a_dim1 == b_dim0,
      "Dimension mismatch: ",
      trans_a_ ? "trans(A): " : "A: ",
      a_dim0,
      " ",
      a_dim1,
      trans_b_ ? ", trans(B): " : ", B: ",
      b_dim0,
      " ",
      b_dim1);

  Y->Resize(A.dim(0), a_dim0, b_dim1);

  if (!A.dim(0)) {
    Y->mutable_data<float>(); // create output tensor
    return true;
  }

  float alpha = 1;
  float beta = 0;

  CUBLAS_ENFORCE(cublasSgemmStridedBatched(
      context_.cublas_handle(),
      trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      b_dim1,
      a_dim0,
      a_dim1,
      &alpha,
      B.data<float>(),
      trans_b_ ? a_dim1 : b_dim1, // ldb
      B.size() / B.dim(0), // b stride
      A.data<float>(),
      trans_a_ ? a_dim0 : a_dim1, // lda
      A.size() / A.dim(0), // a stride
      &beta,
      Y->mutable_data<float>(),
      b_dim1,
      a_dim0 * b_dim1, // y stride
      A.dim32(0) // batch count
      ));
  return true;
}
#endif // __CUDACC_VER_MAJOR__ >= 8

REGISTER_CUDA_OPERATOR(BatchMatMul, BatchMatMulOp<float, CUDAContext>);
} // namespace caffe2
