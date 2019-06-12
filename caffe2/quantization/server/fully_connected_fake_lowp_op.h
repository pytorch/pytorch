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

#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_FP16_OP_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_FP16_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"
#include <immintrin.h>


const int nlines_log = 10000;

namespace caffe2 {

// convert to float16 reducing mantissa, preserving exponent
void fp32_to_bfp16(const float* source, size_t size, float* dest)
{
  // Results on a 1 sign, 8 exponent, 7 mantissa
  constexpr int mask = 0xFFFF0000;
  __m256 wmask = _mm256_broadcast_ss((float*)(&mask));

  for (auto i = 0; i < (size / 8) * 8; i += 8) {
    __m256 data = _mm256_loadu_ps(&source[i]);
    _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
  }
  for (auto i = (size / 8) * 8; i < size; i++) {
    float tmp[8] __attribute__((aligned(8)));
    __m256 data = _mm256_and_ps(wmask, _mm256_set1_ps(source[i]));
    _mm256_storeu_ps(&tmp[0], data);
    dest[i] = tmp[0];
  }
}

// convert to float24 reducing mantissa, preserving exponent
void fp32_to_bfp24(const float* source, size_t size, float* dest)
{
  // Results on a 1 sign, 8 exponent, 7 mantissa
  constexpr int mask = 0xFFFFFF00;
  __m256 wmask = _mm256_broadcast_ss((float*)(&mask));

  for (auto i = 0; i < (size / 8) * 8; i += 8) {
    __m256 data = _mm256_loadu_ps(&source[i]);
    _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
  }
  for (auto i = (size / 8) * 8; i < size; i++) {
    float tmp[8] __attribute__((aligned(8)));
    __m256 data = _mm256_and_ps(wmask, _mm256_set1_ps(source[i]));
    _mm256_storeu_ps(&tmp[0], data);
    dest[i] = tmp[0];
  }
}

// convert to float14 reducing mantissa, preserving exponent
void fp32_to_bfp14(const float* source, size_t size, float* dest)
{
  // Results on a 1 sign, 8 exponent, 7 mantissa
  constexpr int mask = 0xFFFC0000;
  __m256 wmask = _mm256_broadcast_ss((float*)(&mask));

  for (auto i = 0; i < (size / 8) * 8; i += 8) {
    __m256 data = _mm256_loadu_ps(&source[i]);
    _mm256_storeu_ps(&dest[i], _mm256_and_ps(wmask, data));
  }
  for (auto i = (size / 8) * 8; i < size; i++) {
    float tmp[8] __attribute__((aligned(8)));
    __m256 data = _mm256_and_ps(wmask, _mm256_set1_ps(source[i]));
    _mm256_storeu_ps(&tmp[0], data);
    dest[i] = tmp[0];
  }
}

void fp32_to_bfp16_scalar(const float* source, size_t size, float* dest)
{
  constexpr int mask = 0xFFFF0000;
  for(auto i =0; i < size; i++){
    *(int*)(dest+i) = *(int *)(source+i) & mask;
  }
}

// convert to IEEE float16
void fp32_to_fp16(const float* source, size_t size, float* dest)
{
  for (auto i = 0; i < (size / 8) * 8; i += 8) {
    __m128i vin_fp16 = _mm256_cvtps_ph(_mm256_loadu_ps(&source[i]), 0);
    _mm256_storeu_ps(&dest[i], _mm256_cvtph_ps(vin_fp16));
  }
  for (auto i = (size / 8) * 8; i < size; i++) {
    float tmp[8] __attribute__((aligned(8)));
    __m128i vin_fp16 = _mm256_cvtps_ph(_mm256_set1_ps(source[i]), 0);
    _mm256_storeu_ps(&tmp[0], _mm256_cvtph_ps(vin_fp16));
    dest[i] = tmp[0];
  }
}

// fp32 -> int32 -> += 1<< 15 -> fp32 -> truncation
void fp32_to_bfp16_round(const float* source, size_t size, float* dest)
{
  constexpr int offset = 0x00008000; // 1 << 15
  constexpr int mask = 0xFFFF0000;

  __m256i woffset = _mm256_set1_epi32(offset);
  __m256i wmask = _mm256_set1_epi32(mask);

  for (auto i = 0; i < (size / 8) * 8; i += 8) {
    __m256i v32int = _mm256_add_epi32(_mm256_loadu_si256((__m256i const*)&source[i]), woffset);
    _mm256_storeu_si256((__m256i *)&dest[i], _mm256_and_si256(wmask, v32int));
  }
  for (auto i = (size / 8) * 8; i < size; i++) {
    float tmp[8] __attribute__((aligned(8)));
    __m256i v32int = _mm256_add_epi32(_mm256_set1_epi32(*(int *)&source[i]), woffset);
    _mm256_storeu_si256((__m256i *)&tmp[0], _mm256_and_si256(wmask, v32int));
    dest[i] = tmp[0];
  }
}

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <
    void (*Q)(const float*, size_t, float*),
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true>
class FullyConnectedFakeLowpFPOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedFakeLowpFPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            OperatorBase::GetSingleArgument<bool>("float16_compute", false)) {}
  ~FullyConnectedFakeLowpFPOp() {}

  template <
      typename T_X,
      typename T_W,
      typename T_B,
      typename T_Y,
      typename MATH>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& b = Input(2);
    auto* Y = Output(0);
    CAFFE_ENFORCE(b.dim() == 1, b.dim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const auto M = X.size_to_dim(canonical_axis);
    const auto K = X.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
    const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                  : W.size_from_dim(canonical_axis_w);

    auto dimErrorString = [&]() {
      return c10::str(
          "Dimension mismatch: ",
          "X: ",
          X.sizes(),
          ", W: ",
          W.sizes(),
          ", b: ",
          b.sizes(),
          ", axis: ",
          axis_,
          ", M: ",
          M,
          ", N: ",
          N,
          ", K: ",
          K);
    };

    // Error checking
    CAFFE_ENFORCE(M == X.size() / K, dimErrorString());
    CAFFE_ENFORCE(K == W.size() / N, dimErrorString());
    CAFFE_ENFORCE(N == b.dim32(0), dimErrorString());
    CAFFE_ENFORCE(N == b.size(), dimErrorString());

    static int log_occurences = 0;
    if (log_occurences % nlines_log == 0) {
      ++log_occurences;
      LOG(INFO) << "FAKE_FP16 fc running";
    }

    Y_shape_cache_ = X.sizes().vec();
    // This is an invariant of canonical_axis, so we can DCHECK.
    DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
    Y_shape_cache_.resize(canonical_axis + 1);
    Y_shape_cache_[canonical_axis] = N;
    Y->Resize(Y_shape_cache_);
    CAFFE_ENFORCE(M * N == Y->size(), dimErrorString());

    if (X.size() == 0) {
      // skip the rest of the computation if X is empty
      Y->template mutable_data<T_Y>();
      return true;
    }

    // default to FLOAT as math.h does.
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
    if (fp16_type<MATH>()) {
      math_type = TensorProto_DataType_FLOAT16;
    }

    // Y = W * X + b
    // Quantize W, X, b
    auto type = Context::GetDeviceType();
    Tensor Xh(type);
    Xh.ResizeLike(X);
    Q(X.template data<T_X>(),
      Xh.size(),
      Xh.template mutable_data<T_X>()
    );

    Tensor Wh(type);
    Wh.ResizeLike(W);
    Q(W.template data<T_W>(),
      Wh.size(),
      Wh.template mutable_data<T_W>()
    );

    Tensor bh(type);
    bh.ResizeLike(b);
    Q(b.template data<T_B>(),
      bh.size(),
      bh.template mutable_data<T_B>()
    );

    // W * x
    math::Gemm<T_X, Context, Engine>(
        CblasNoTrans,
        TransposeWeight ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        1,
        Xh.template data<T_X>(),
        Wh.template data<T_W>(),
        0,
        Y->template mutable_data<T_Y>(),
        &context_,
        math_type);
    // Add bias term
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T_B, Context>(
          M,
          convert::To<float, T_B>(1),
          bias_multiplier_.template mutable_data<T_B>(),
          &context_);
    }
    math::Gemm<T_B, Context, Engine>(
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        1,
        1,
        bias_multiplier_.template data<T_B>(),
        bh.template data<T_B>(),
        1,
        Y->template mutable_data<T_Y>(),
        &context_,
        math_type);

    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_;
  Tensor bias_multiplier_{Context::GetDeviceType()};

  bool float16_compute_;
};

template <
    void (*Q)(const float*, size_t, float*),
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true>
class FullyConnectedGradientFakeLowpFPOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedGradientFakeLowpFPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            OperatorBase::GetSingleArgument<bool>("float16_compute", false)) {}
  ~FullyConnectedGradientFakeLowpFPOp() {}

  template <
      typename T_X,
      typename T_W,
      typename T_DY,
      typename T_B,
      typename T_DX,
      typename T_DW,
      typename T_DB,
      typename MATH>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& dY = Input(2);
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int M = X.size_to_dim(canonical_axis);
    const int K = X.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
    const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                  : W.size_from_dim(canonical_axis_w);
    CAFFE_ENFORCE(M * K == X.size());
    CAFFE_ENFORCE(K * N == W.size());

    auto* dW = Output(0);
    auto* db = Output(1);
    dW->ResizeLike(W);
    db->Resize(N);

    if (X.size() == 0) {
      // generate a zero blob for db and dW when X is empty
      math::Set<T_DB, Context>(
          db->size(),
          convert::To<float, T_DB>(0),
          db->template mutable_data<T_DB>(),
          &context_);
      math::Set<T_DW, Context>(
          dW->size(),
          convert::To<float, T_DW>(0),
          dW->template mutable_data<T_DW>(),
          &context_);

      if (OutputSize() == 3) {
        auto* dX = Output(2);
        dX->ResizeLike(X);
        dX->template mutable_data<T_DX>();
      }

      return true;
    }

    // default to FLOAT as math.h does.
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
    if (fp16_type<MATH>()) {
      math_type = TensorProto_DataType_FLOAT16;
    }

    auto type = Context::GetDeviceType();
    // Quantize: W, X, dY
    Tensor Xh(type);
    Xh.ResizeLike(X);
    Q(X.template data<T_X>(),
      Xh.size(),
      Xh.template mutable_data<T_X>()
    );

    Tensor Wh(type);
    Wh.ResizeLike(W);
    Q(W.template data<T_W>(),
      Wh.size(),
      Wh.template mutable_data<T_W>()
    );

    Tensor dYh(type);
    dYh.ResizeLike(dY);
    Q(dY.template data<T_DY>(),
      dYh.size(),
      dYh.template mutable_data<T_DY>()
    );

    static int log_occurences = 0;
    if (log_occurences % nlines_log == 0) {
      ++log_occurences;
      LOG(INFO) << "FAKE_FP16 fc grad running";
    }

    // Compute dW
    math::Gemm<T_DY, Context, Engine>(
        CblasTrans,
        CblasNoTrans,
        TransposeWeight ? N : K,
        TransposeWeight ? K : N,
        M,
        1,
        TransposeWeight ? dYh.template data<T_DY>() : Xh.template data<T_X>(),
        TransposeWeight ? Xh.template data<T_X>() : dYh.template data<T_DY>(),
        0,
        dW->template mutable_data<T_DW>(),
        &context_,
        math_type);
    if (bias_multiplier_.size() != M) {
      // If the helper bias multiplier is not M, reshape and fill it
      // with one.
      bias_multiplier_.Resize(M);
      math::Set<T_B, Context>(
          M,
          convert::To<float, T_B>(1),
          bias_multiplier_.template mutable_data<T_B>(),
          &context_);
    }
    // Compute dB
    math::Gemv<T_DY, Context>(
        CblasTrans,
        M,
        N,
        1,
        dYh.template data<T_DY>(),
        bias_multiplier_.template data<T_B>(),
        0,
        db->template mutable_data<T_DB>(),
        &context_);

    // Compute dX
    if (OutputSize() == 3) {
      auto* dX = Output(2);
      dX->ResizeLike(X);
      math::Gemm<T_DX, Context, Engine>(
          CblasNoTrans,
          TransposeWeight ? CblasNoTrans : CblasTrans,
          M,
          K,
          N,
          1,
          dYh.template data<T_DY>(),
          Wh.template data<T_W>(),
          0,
          dX->template mutable_data<T_DX>(),
          &context_,
          math_type);
    }

    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  Tensor bias_multiplier_{Context::GetDeviceType()};
  bool float16_compute_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FULLY_CONNECTED_FP16_OP_H_
