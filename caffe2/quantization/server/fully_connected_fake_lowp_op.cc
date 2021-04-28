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

#include <functional>

#include "fully_connected_fake_lowp_op.h"

namespace caffe2 {

constexpr int nlines_log = 10000;

template <
    void (*Q)(const float*, size_t, float*),
    class Context,
    class Engine,
    bool TransposeWeight>
template <typename T_X, typename T_W, typename T_B, typename T_Y, typename MATH>
bool FullyConnectedFakeLowpFPOp<Q, Context, Engine, TransposeWeight>::
    DoRunWithType() {
  const auto& X = Input(0);
  const auto& W = Input(1);
  const auto& b = Input(2);

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
  auto* Y = Output(0, Y_shape_cache_, at::dtype<T_Y>());
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
  Q(X.template data<T_X>(), Xh.size(), Xh.template mutable_data<T_X>());

  Tensor Wh(type);
  Wh.ResizeLike(W);
  Q(W.template data<T_W>(), Wh.size(), Wh.template mutable_data<T_W>());

  Tensor bh(type);
  bh.ResizeLike(b);
  Q(b.template data<T_B>(), bh.size(), bh.template mutable_data<T_B>());

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
    ReinitializeTensor(
        &bias_multiplier_,
        {M},
        at::dtype<T_B>().device(Context::GetDeviceType()));
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

template <
    void (*Q)(const float*, size_t, float*),
    class Context,
    class Engine,
    bool TransposeWeight>
template <
    typename T_X,
    typename T_W,
    typename T_DY,
    typename T_B,
    typename T_DX,
    typename T_DW,
    typename T_DB,
    typename MATH>
bool FullyConnectedGradientFakeLowpFPOp<Q, Context, Engine, TransposeWeight>::
    DoRunWithType() {
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

  auto* dW = Output(0, W.sizes(), at::dtype<T_DW>());
  auto* db = Output(1, {N}, at::dtype<T_DB>());

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
      Output(2, X.sizes(), at::dtype<T_DX>());
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
  Q(X.template data<T_X>(), Xh.size(), Xh.template mutable_data<T_X>());

  Tensor Wh(type);
  Wh.ResizeLike(W);
  Q(W.template data<T_W>(), Wh.size(), Wh.template mutable_data<T_W>());

  Tensor dYh(type);
  dYh.ResizeLike(dY);
  Q(dY.template data<T_DY>(), dYh.size(), dYh.template mutable_data<T_DY>());

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
    ReinitializeTensor(
        &bias_multiplier_,
        {M},
        at::dtype<T_B>().device(Context::GetDeviceType()));
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
    auto* dX = Output(2, X.sizes(), at::dtype<T_DX>());
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

// IEEE FP16
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    FAKE_FP16,
    FullyConnectedFakeLowpFPOp<fp32_to_fp16, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FCGradient,
    FAKE_FP16,
    FullyConnectedGradientFakeLowpFPOp<fp32_to_fp16, CPUContext>);

// BFLOAT 16
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    FAKE_BFP_16,
    FullyConnectedFakeLowpFPOp<fp32_to_bfp16, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FCGradient,
    FAKE_BFP_16,
    FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp16, CPUContext>);

// BFLOAT 24 (chop the least significant 8 bits)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    FAKE_BFP_24,
    FullyConnectedFakeLowpFPOp<fp32_to_bfp24, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FCGradient,
    FAKE_BFP_24,
    FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp24, CPUContext>);

// BFLOAT 14 (chop 2 extra bits from BFLOAT 16)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    FAKE_BFP_14,
    FullyConnectedFakeLowpFPOp<fp32_to_bfp14, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FCGradient,
    FAKE_BFP_14,
    FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp14, CPUContext>);

// BFLOAT16 with rounding
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FC,
    FAKE_BFP_16_ROUND,
    FullyConnectedFakeLowpFPOp<fp32_to_bfp16_round, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FCGradient,
    FAKE_BFP_16_ROUND,
    FullyConnectedGradientFakeLowpFPOp<fp32_to_bfp16_round, CPUContext>);

} // namespace caffe2
