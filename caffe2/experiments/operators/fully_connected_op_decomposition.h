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

#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_DECOMPOSITION_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_DECOMPOSITION_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
/*
 * Although a FC_decomp is just like 2 small FC,
 * it is better to have it as one op for future analysis.
 * And if we have 2 FC with bias, it is not right.
 * TODO(wyiming): decompose the layer into 2 matrices
 * W(N * K) = U(N * middle) * trans(V(K * middle))
 * */
// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedOpDecomp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedOpDecomp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedOpDecomp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& U = Input(1);
    const auto& V = Input(2);
    const auto& b = Input(3);

    //auto* buffer_ptr = Output(1);
    // Size M * middle;
    //auto& multi_buffer_ = *buffer_ptr;
    CAFFE_ENFORCE_GE(X.dim(), 1);
    CAFFE_ENFORCE_GE(U.dim(), 2);
    CAFFE_ENFORCE_GE(V.dim(), 2);
    if (X.dim() > 2 || U.dim() > 2 || V.dim() > 2) {
      VLOG(1) << "Using legacy support for arbitrary input and weight "
                       "dimensions.";
    }
    CAFFE_ENFORCE_EQ(b.dim(), 1);
    // batch size
    int M = X.dim() > 1 ? X.dim32(0) : 1;
    // Feature dimension
    int K = X.numel() / M;
    // number of outputs.
    int N = U.dim32(0);
    int middle = U.dim32(0);
    CAFFE_ENFORCE_EQ(K, V.dim32(0));
    CAFFE_ENFORCE_EQ(N, b.dim32(0));
    std::vector<int64_t> dims;
    if (X.dim() > 1) {
      dims = {M, N};
      multi_buffer_.Resize(M, middle);
    } else {
      dims = {N};
      multi_buffer_.Resize(middle);
    }
    auto* Y = Output(0, dims, at::dtype<T>());
    // The col buffer is stored in CHW order as well - kernel_dim, and the
    // height and width.
    //  multi_buffer_.Resize(M, middle);
    T* multi_buffer_data = multi_buffer_.template mutable_data<T>();
    //  X * V * tans(U)
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, middle, K, 1, X.template data<T>(),
        V.template data<T>(), 0, multi_buffer_data,
        &context_);
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasTrans, M, N, middle, 1, multi_buffer_data,
        U.template data<T>(), 0, Y->template mutable_data<T>(),
        &context_);
    // Add bias term
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T, Context>(
          M, static_cast<T>(1), bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, N, 1, 1,
        bias_multiplier_.template data<T>(), b.template data<T>(), 1,
        Y->template mutable_data<T>(), &context_);
    return true;
  }

 protected:
  Tensor bias_multiplier_{Context::GetDeviceType()};
  Tensor multi_buffer_{Context::GetDeviceType()};
};

template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedDecompGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedDecompGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedDecompGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& U = Input(1);
    const auto& V = Input(2);
    const auto& dY = Input(3);
    DCHECK_GE(X.dim(), 1);
    DCHECK_GE(U.dim(), 2);
    DCHECK_GE(V.dim(), 2);
    DCHECK_LE(dY.dim(), 2);
    // batch size
    int M = X.dim() > 1 ? X.dim32(0) : 1;
    // Feature dimension
    int K = X.numel() / M;
    // number of outputs.
    int N = U.dim32(0);
    int middle = U.dim32(1);
    DCHECK_EQ(K, V.dim32(0));
    if (dY.dim() > 1) {
      DCHECK_EQ(M, dY.dim32(0));
      DCHECK_EQ(N, dY.dim32(1));
    } else {
      DCHECK_EQ(X.dim(), 1);
      DCHECK_EQ(N, dY.numel());
    }

    auto* dU = Output(0, U.sizes(), at::dtype<T>());
    auto* dV = Output(1, V.sizes(), at::dtype<T>());
    auto* db = Output(2, {N}, at::dtype<T>());

    // Compute dU
    // first compute X * V
    du_buffer_.Resize(N, middle);
    T* du_buffer_data = du_buffer_.template mutable_data<T>();
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, middle, K, 1,
        X.template data<T>(), V.template data<T>(),
        0, du_buffer_data,
        &context_);
    math::Gemm<T, Context, Engine>(
        CblasTrans, CblasNoTrans, N, middle, M, 1,
        dY.template data<T>(), du_buffer_data,
        0, dU->template mutable_data<T>(),
        &context_);
    // Compute dV
    // first compute dY * U
    dv_buffer_.Resize(M, middle);
    T* dv_buffer_data = dv_buffer_.template mutable_data<T>();
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, M, middle, N, 1,
        dY.template data<T>(), U.template data<T>(),
        0, dv_buffer_data,
        &context_);
    math::Gemm<T, Context, Engine>(
        CblasTrans, CblasNoTrans, K, middle, M, 1,
        dY.template data<T>(), du_buffer_data,
        0, dV->template mutable_data<T>(),
        &context_);
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T, Context>(
          M, static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    // Compute dB
    math::Gemv<T, Context>(
        CblasTrans, M, N, 1, dY.template data<T>(),
        bias_multiplier_.template data<T>(), 0,
        db->template mutable_data<T>(),
        &context_);
    // Compute dX if necessary.
    if (OutputSize() == 4) {
      auto* dX = Output(3, X.sizes(), at::dtype<T>());
      dx_buffer_.Resize(M, middle);
      T* dx_buffer_data = dx_buffer_.template mutable_data<T>();
      math::Gemm<T, Context, Engine>(
          CblasNoTrans, CblasNoTrans, M, middle, N, 1,
          dY.template data<T>(), U.template data<T>(),
          0, dx_buffer_data,
          &context_);
      math::Gemm<T, Context, Engine>(
          CblasNoTrans, CblasTrans, M, K, middle, 1,
          dx_buffer_data, V.template data<T>(),
          0, dX->template mutable_data<T>(),
          &context_);
    }

    return true;
  }

 protected:
  Tensor bias_multiplier_{Context::GetDeviceType()};
  Tensor du_buffer_{Context::GetDeviceType()};
  Tensor dv_buffer_{Context::GetDeviceType()};
  Tensor dx_buffer_{Context::GetDeviceType()};
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
