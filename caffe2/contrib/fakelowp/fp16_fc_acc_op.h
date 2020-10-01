/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <fbgemm/FbgemmConvert.h>
#include <fbgemm/FbgemmFP16.h>
#include <immintrin.h>

#include "caffe2/contrib/fakelowp/fp16_gemm_utils.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

using namespace std;

// C2 wrapper for fp16 gemm with fp16 accumulation
template <
    class Context,
    class Engine = DefaultEngine,
    bool USE_ACC_FP16 = false, // Whether use fp16 accumulation
    bool USE_TMP_ACCUMULATOR = false,
    bool ADD_BIAS_FIRST = false>
class Fp16FCAccOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  Fp16FCAccOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)) {}
  ~Fp16FCAccOp() noexcept override {
    if (X_fp16_ != nullptr) {
      delete[] X_fp16_;
    }
    if (W_fp16_ != nullptr) {
      delete[] W_fp16_;
    }
    if (b_fp16_ != nullptr) {
      delete[] b_fp16_;
    }
    if (bias_multiplier_fp16_ != nullptr) {
      delete[] bias_multiplier_fp16_;
    }
  }

  // template on X, B, W and Y.
  template <typename T_X, typename T_B, typename T_W, typename T_Y>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& W_blob = OperatorBase::InputBlob(1);
    const auto& b = Input(2);
    auto* Y = Output(0);
    CAFFE_ENFORCE(b.ndim() == 1, b.ndim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int M = X.size_to_dim(canonical_axis);
    const int N = b.size();
    const int K = X.size_from_dim(canonical_axis);

    Y_shape_cache_ = X.sizes().vec();
    // This is an invariant of canonical_axis, so we can DCHECK.
    DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
    Y_shape_cache_.resize(canonical_axis + 1);
    Y_shape_cache_[canonical_axis] = N;
    Y->Resize(Y_shape_cache_);

    if (X.size() == 0) {
      // skip the rest of the computation if X is empty
      Y->template mutable_data<T_Y>();
      return true;
    }

    // Convert X and W to fp16
    int X_size = M * K;
    int W_size = N * K;
    if (X_fp16_ == nullptr) {
      X_fp16_ = new float[X_size];
      X_size_cached_ = X_size;
    } else if (X_size > X_size_cached_) {
      delete[] X_fp16_;
      X_fp16_ = new float[X_size];
      X_size_cached_ = X_size;
    }
    fbgemm::RoundToFloat16(
        X.template data<T_X>(),
        X_fp16_,
        X_size,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    if (W_fp16_ == nullptr) {
      W_fp16_ = new float[W_size];
      const T_W* W_data = nullptr;
      if (W_blob.template IsType<
              caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16>>()) {
        auto* W_fbgemm =
            OperatorBase::Input<
                caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16>>(1)
                .get();

        if (!W_fbgemm->packed()) {
          float* W_fp16_trans = new float[W_size];
          fbgemm::Float16ToFloat_avx2(W_fbgemm->pmat(), W_fp16_trans, W_size);
          for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
              W_fp16_[j * N + i] = W_fp16_trans[i * K + j];
            }
          }
          delete[] W_fp16_trans;
        } else {
          vector<fbgemm::float16> unpacked_mat;
          unpacked_mat.resize(W_size);
          W_fbgemm->unpack(
              unpacked_mat.data(), fbgemm::matrix_op_t::NoTranspose);
          fbgemm::Float16ToFloat_avx2(unpacked_mat.data(), W_fp16_, W_size);
        }

      } else {
        const auto& W = Input(1);
        W_data = W.template data<T_W>();
        // Transpose W
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < K; j++) {
            W_fp16_[j * N + i] = W_data[i * K + j];
          }
        }
      }

      fbgemm::RoundToFloat16(
          W_fp16_, W_fp16_, W_size, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    }

    auto Y_data = Y->template mutable_data<T_Y>();
    int Y_size = M * N;

    // Initialize Y
    memset(Y_data, 0.0, sizeof(float) * Y_size);

    // Add bias term, accumulation is in fp16.
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T_B, Context>(
          M,
          convert::To<float, T_B>(1),
          bias_multiplier_.template mutable_data<T_B>(),
          &context_);
    }
    if (bias_multiplier_fp16_ == nullptr) {
      bias_multiplier_fp16_ = new float[M];
      M_cached_ = M;
    } else if (M > M_cached_) {
      delete[] bias_multiplier_fp16_;
      bias_multiplier_fp16_ = new float[M];
      M_cached_ = M;
    }
    fbgemm::RoundToFloat16(
        bias_multiplier_.template data<T_B>(),
        bias_multiplier_fp16_,
        M,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    if (b_fp16_ == nullptr) {
      b_fp16_ = new float[N];
    }
    fbgemm::RoundToFloat16(
        b.template data<T_B>(),
        b_fp16_,
        N,
        FLAGS_caffe2_fbgemm_fake_fp16_clamp);

    if (ADD_BIAS_FIRST) {
      custom_fp16_gemm(
          M,
          1,
          N,
          bias_multiplier_fp16_,
          b_fp16_,
          0.f,
          Y->template mutable_data<T_Y>(),
          USE_ACC_FP16,
          USE_TMP_ACCUMULATOR);
#ifdef LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG
      float* Y_ref = new float[M * N]();
      TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
      math::Gemm<T_B, Context, Engine>(
          CblasNoTrans,
          CblasNoTrans,
          M,
          N,
          1,
          1,
          bias_multiplier_.template data<T_B>(),
          b.template data<T_B>(),
          0.f,
          Y_ref,
          &context_,
          math_type);

      relative_error =
          compute_relative_error(Y->template mutable_data<T_Y>(), Y_ref, M * N);
      total_error_with_bias += relative_error;
      VLOG(LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG)
          << "Relative error for Y = bias_multiplier_ * b' = " << relative_error
          << ", average error with bias after " << runs
          << " runs = " << total_error_with_bias / runs << endl;
#endif

      custom_fp16_gemm(
          M,
          K,
          N,
          X_fp16_,
          W_fp16_,
          1.f,
          Y->template mutable_data<T_Y>(),
          USE_ACC_FP16,
          USE_TMP_ACCUMULATOR);

#ifdef LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG
      if (!W_blob.IsType<caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16>>()) {
        const auto& W = Input(1);
        math::Gemm<float, Context, Engine>(
            CblasNoTrans,
            CblasTrans,
            M,
            N,
            K,
            1,
            X.template data<T_X>(),
            W.template data<T_W>(),
            1.f,
            Y_ref,
            &context_,
            math_type);

        runs++;
        float relative_error = compute_relative_error(
            Y->template mutable_data<T_Y>(), Y_ref, M * N);
        total_error += relative_error;
        VLOG(LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG)
            << "Relative error for Y = bias_multiplier_ * b' + X * W' = "
            << relative_error << ", average error after " << runs
            << " runs = " << total_error / runs << endl;

        if (Y_ref != nullptr) {
          delete[] Y_ref;
        }
      }
#endif

    } else {
      custom_fp16_gemm(
          M,
          K,
          N,
          X_fp16_,
          W_fp16_,
          0.f,
          Y->template mutable_data<T_Y>(),
          USE_ACC_FP16,
          USE_TMP_ACCUMULATOR);
#ifdef LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG
      if (!W_blob.IsType<caffe2::unique_ptr<fbgemm::PackedGemmMatrixFP16>>()) {
        const auto& W = Input(1);
        float* Y_ref = new float[M * N]();
        TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
        math::Gemm<float, Context, Engine>(
            CblasNoTrans,
            CblasTrans,
            M,
            N,
            K,
            1,
            X.template data<T_X>(),
            W.template data<T_W>(),
            0.f,
            Y_ref,
            &context_,
            math_type);

        runs++;
        float relative_error = compute_relative_error(
            Y->template mutable_data<T_Y>(), Y_ref, M * N);
        total_error += relative_error;
        VLOG(LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG)
            << "Relative error for Y = X * W' = " << relative_error
            << ", average error after " << runs
            << " runs = " << total_error / runs << endl;
      }
#endif

      custom_fp16_gemm(
          M,
          1,
          N,
          bias_multiplier_fp16_,
          b_fp16_,
          1.f,
          Y->template mutable_data<T_Y>(),
          USE_ACC_FP16,
          USE_TMP_ACCUMULATOR);

#ifdef LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG
      math::Gemm<T_B, Context, Engine>(
          CblasNoTrans,
          CblasNoTrans,
          M,
          N,
          1,
          1,
          bias_multiplier_.template data<T_B>(),
          b.template data<T_B>(),
          1,
          Y_ref,
          &context_,
          math_type);

      relative_error =
          compute_relative_error(Y->template mutable_data<T_Y>(), Y_ref, M * N);
      total_error_with_bias += relative_error;
      VLOG(LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG)
          << "Relative error for Y = X * W' + bias_multiplier_ * b' = "
          << relative_error << ", average error with bias after " << runs
          << " runs = " << total_error_with_bias / runs << endl;
      if (Y_ref != nullptr) {
        delete[] Y_ref;
      }
#endif
    }

    return true;
  }

#ifdef LOG_LEVEL_FOR_FBFCPACKEDACC16_ACCURACY_LOG
  float compute_L2_norm(float* A, int size) {
    float square_sum = 0.0;
    for (int i = 0; i < size; i++) {
      square_sum += A[i] * A[i];
    }
    return std::sqrt(square_sum);
  }

  float compute_relative_error(float* A, float* A_ref, int size) {
    float error = 0.0;
    for (int i = 0; i < size; i++) {
      error += (A[i] - A_ref[i]) * (A[i] - A_ref[i]);
    }
    error = std::sqrt(error);
    float L2_norm = compute_L2_norm(A, size);
    return error / L2_norm;
  }
#endif

  bool RunOnDevice() override {
    return DoRunWithType<
        float, // X
        float, // B
        float, // W
        float>(); // Y
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  size_t X_size_cached_{0};
  size_t M_cached_{0};
  static int runs;
  static float total_error;
  static float total_error_with_bias;
  float* X_fp16_ = nullptr;
  float* W_fp16_ = nullptr;
  float* b_fp16_ = nullptr;
  float* bias_multiplier_fp16_ = nullptr;
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_;
  Tensor bias_multiplier_{Context::GetDeviceType()};
};

} // namespace caffe2
