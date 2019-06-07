#ifndef CAFFE2_OPERATORS_BATCH_MATMUL_OP_H_
#define CAFFE2_OPERATORS_BATCH_MATMUL_OP_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context, class Engine = DefaultEngine>
class BatchMatMulOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit BatchMatMulOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, "trans_a", trans_a_, false),
        OP_SINGLE_ARG(bool, "trans_b", trans_b_, false),
        OP_SINGLE_ARG(bool, "broadcast", broadcast_, false) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& A = Input(0);
    const auto& B = Input(1);
    const int A_ndim = A.dim();
    const int B_ndim = B.dim();
    const std::vector<std::int64_t> A_dims = A.sizes().vec();
    const std::vector<std::int64_t> B_dims = B.sizes().vec();
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();

    if (A_ndim == 1 && B_ndim == 1) {
      CAFFE_ENFORCE_EQ(A.numel(), B.numel());
      auto* Y = Output(0, {1}, at::dtype<T>());
      T* Y_data = Y->template mutable_data<T>();
      math::Dot<T, Context>(A.numel(), A_data, B_data, Y_data, &context_);
      return true;
    }
    if (A_ndim == 1) {
      const int N = A.numel();
      if (trans_b_) {
        CAFFE_ENFORCE_EQ(B_dims[B_ndim - 1], N);
      } else {
        CAFFE_ENFORCE_EQ(B_dims[B_ndim - 2], N);
      }
      std::vector<std::int64_t> Y_dims(B_ndim - 1);
      if (trans_b_) {
        std::copy_n(B_dims.cbegin(), B_ndim - 1, Y_dims.begin());
      } else {
        std::copy_n(B_dims.cbegin(), B_ndim - 2, Y_dims.begin());
        Y_dims.back() = B_dims.back();
      }
      auto* Y = Output(0, Y_dims, at::dtype<T>());
      T* Y_data = Y->template mutable_data<T>();
      if (trans_b_) {
        const int M = B.numel() / N;
        math::Gemv<T, Context, Engine>(
            CblasNoTrans, M, N, 1.0f, B_data, A_data, 0.0f, Y_data, &context_);
      } else {
        const int M = B_dims[B_ndim - 1];
        const int batch_size = B.numel() / (M * N);
        if (batch_size == 1) {
          math::Gemv<T, Context, Engine>(
              CblasTrans, N, M, 1.0f, B_data, A_data, 0.0f, Y_data, &context_);
        } else {
          math::GemmStridedBatched<T, Context, Engine>(
              CblasTrans,
              CblasNoTrans,
              batch_size,
              M,
              1,
              N,
              1.0f,
              B_data,
              M * N,
              A_data,
              0,
              0.0f,
              Y_data,
              M,
              &context_);
        }
      }
      return true;
    }
    if (B_ndim == 1) {
      const int N = B.numel();
      if (trans_a_) {
        CAFFE_ENFORCE_EQ(A_dims[A_ndim - 2], N);
      } else {
        CAFFE_ENFORCE_EQ(A_dims[A_ndim - 1], N);
      }
      const std::vector<std::int64_t> Y_dims(
          A_dims.cbegin(), A_dims.cbegin() + A_ndim - 1);
      auto* Y = Output(0, Y_dims, at::dtype<T>());
      T* Y_data = Y->template mutable_data<T>();
      if (trans_a_) {
        const int M = A_dims[A_ndim - 1];
        const int batch_size = A.numel() / (M * N);
        if (batch_size == 1) {
          math::Gemv<T, Context, Engine>(
              CblasTrans, N, M, 1.0f, A_data, B_data, 0.0f, Y_data, &context_);
        } else {
          math::GemmStridedBatched<T, Context, Engine>(
              CblasTrans,
              CblasNoTrans,
              batch_size,
              M,
              1,
              N,
              1.0f,
              A_data,
              M * N,
              B_data,
              0,
              0.0f,
              Y_data,
              M,
              &context_);
        }
      } else {
        const int M = A.numel() / N;
        math::Gemv<T, Context, Engine>(
            CblasNoTrans, M, N, 1.0f, A_data, B_data, 0.0f, Y_data, &context_);
      }
      return true;
    }

    const int M = trans_a_ ? A_dims[A_ndim - 1] : A_dims[A_ndim - 2];
    const int K = trans_a_ ? A_dims[A_ndim - 2] : A_dims[A_ndim - 1];
    if (trans_b_) {
      CAFFE_ENFORCE_EQ(B_dims[B_ndim - 1], K);
    } else {
      CAFFE_ENFORCE_EQ(B_dims[B_ndim - 2], K);
    }
    const int N = trans_b_ ? B_dims[B_ndim - 2] : B_dims[B_ndim - 1];
    const int ndim = std::max(A_ndim, B_ndim);
    std::vector<std::int64_t> A_broadcast_dims(ndim);
    std::vector<std::int64_t> B_broadcast_dims(ndim);
    std::vector<std::int64_t> Y_broadcast_dims(ndim);
    math::utils::ComputeBroadcastBinaryOpDims(
        A_ndim - 2,
        A_dims.data(),
        B_ndim - 2,
        B_dims.data(),
        A_broadcast_dims.data(),
        B_broadcast_dims.data(),
        Y_broadcast_dims.data());
    Y_broadcast_dims[ndim - 2] = M;
    Y_broadcast_dims[ndim - 1] = N;
    auto* Y = Output(0, Y_broadcast_dims, at::dtype<T>());
    T* Y_data = Y->template mutable_data<T>();

    const int batch_dim = ndim - 2;
    const bool is_broadcast_dims = !std::equal(
        A_broadcast_dims.cbegin(),
        A_broadcast_dims.cbegin() + batch_dim,
        B_broadcast_dims.cbegin());
    if (is_broadcast_dims) {
      CAFFE_ENFORCE(broadcast_);
    }

    const std::int64_t A_batch_size = std::accumulate(
        A_broadcast_dims.cbegin(),
        A_broadcast_dims.cbegin() + batch_dim,
        1LL,
        std::multiplies<std::int64_t>());
    const std::int64_t B_batch_size = std::accumulate(
        B_broadcast_dims.cbegin(),
        B_broadcast_dims.cbegin() + batch_dim,
        1LL,
        std::multiplies<std::int64_t>());
    const std::int64_t Y_batch_size = std::accumulate(
        Y_broadcast_dims.cbegin(),
        Y_broadcast_dims.cbegin() + batch_dim,
        1LL,
        std::multiplies<std::int64_t>());
    if (Y_batch_size == 0) {
      return true;
    }
    if (A_batch_size == 1 && B_batch_size == 1) {
      math::Gemm<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          M,
          N,
          K,
          1.0f,
          A_data,
          B_data,
          0.0f,
          Y_data,
          &context_);
    } else if (A_batch_size == 1) {
      math::GemmStridedBatched<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          Y_batch_size,
          M,
          N,
          K,
          1.0f,
          A_data,
          0,
          B_data,
          K * N,
          0.0f,
          Y_data,
          M * N,
          &context_);
    } else if (B_batch_size == 1) {
      math::GemmStridedBatched<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          Y_batch_size,
          M,
          N,
          K,
          1.0f,
          A_data,
          M * K,
          B_data,
          0,
          0.0f,
          Y_data,
          M * N,
          &context_);
    } else if (!is_broadcast_dims) {
      math::GemmStridedBatched<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          Y_batch_size,
          M,
          N,
          K,
          1.0f,
          A_data,
          M * K,
          B_data,
          K * N,
          0.0f,
          Y_data,
          M * N,
          &context_);
    } else {
      std::vector<const T*> A_ptr(Y_batch_size);
      std::vector<const T*> B_ptr(Y_batch_size);
      std::vector<T*> Y_ptr(Y_batch_size);
      std::vector<std::int64_t> index(batch_dim);
      for (std::int64_t i = 0; i < Y_batch_size; ++i) {
        const std::int64_t A_index = math::utils::GetIndexFromDims(
            batch_dim, A_broadcast_dims.data(), index.data());
        const std::int64_t B_index = math::utils::GetIndexFromDims(
            batch_dim, B_broadcast_dims.data(), index.data());
        A_ptr[i] = A_data + A_index * M * K;
        B_ptr[i] = B_data + B_index * K * N;
        Y_ptr[i] = Y_data + i * M * N;
        math::utils::IncreaseIndexInDims(
            batch_dim, Y_broadcast_dims.data(), index.data());
      }
      math::GemmBatched<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          Y_batch_size,
          M,
          N,
          K,
          1.0f,
          A_ptr.data(),
          B_ptr.data(),
          0.0f,
          Y_ptr.data(),
          &context_);
    }
    return true;
  }

 private:
  const bool trans_a_;
  const bool trans_b_;
  const bool broadcast_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_MATMUL_OP_H_
