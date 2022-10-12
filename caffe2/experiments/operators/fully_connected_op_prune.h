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

#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_PRUNE_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_PRUNE_H_

#include <c10/util/Logging.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <int N>
using Shape = std::array<int, N>;

template <int N>
const std::vector<int64_t>& shape(Shape<N> vs) {
  static thread_local std::vector<int64_t> cache;
  cache.resize(vs.size());
  for (const auto i : c10::irange(vs.size())) {
    cache[i] = vs[i];
  }
  return cache;
}

inline const std::vector<int64_t>& shape(int i) {
  return shape<1>(Shape<1>({i}));
}

inline const std::vector<int64_t>& shape(int i, int j) {
  return shape<2>(Shape<2>({i, j}));
}

template <typename T, class Context>
void MaskMatrix(const T* mask, T* mat, int M, int N);

template <typename T, class Context>
void MaskMatrix_Inc(T* mask_seq, T* mat, int M, int N, int seq_len, T target);

template <typename T, class Context>
void AggrDW(T* ag_dw, const T* dw, int N, int K, Context* context);

template <typename T>
int MatrixCompare_LT(const T* mat, float thres, T* mask_seq, int M, int N);

// TODO(wyiming): write an incremental Mask
// Incremental Mask: only give the new mask positions;
// Assuming that weights masked will not be mask again;
// The incremental mask can also be used to update mask matrix;
// But this will include template for bool and float;
template <>
void MaskMatrix<float, CPUContext>(
    const float* mask,
    float* mat,
    int M,
    int N) {
  int offset = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      mat[offset] = mask[offset] ? mat[offset] : 0;
      offset++;
    }
  }
}

template <>
void MaskMatrix_Inc<float, CPUContext>(
    float* mask_seq,
    float* mat,
    int /*M*/,
    int /*N*/,
    int seq_len,
    float target) {
  for (const auto i : c10::irange(seq_len)) {
    // assume that the mask_seq is smaller than size
    // Although it seems that random access gets bad performance,
    // we make sure that seq is in order;
    mat[static_cast<int>(mask_seq[i])] = target;
  }
}

template <>
void AggrDW<float, CPUContext>(
    float* ag_dw,
    const float* dw,
    int N,
    int K,
    CPUContext* context) {
  math::Add<float, CPUContext>(N * K, dw, ag_dw, ag_dw, context);
}

template <>
int MatrixCompare_LT<float>(
    const float* mat,
    float thres,
    float* mask_seq,
    int M,
    int N) {
  int seq_len = 0;
  int offset = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (mat[offset] != 0 && (mat[offset] < thres && mat[offset] > -thres)) {
        mask_seq[seq_len++] = static_cast<float>(offset);
      }
      offset++;
    }
  }
  return seq_len;
}

} // namespace

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <typename T, class Context, class Engine = DefaultEngine>
class FullyConnectedOpPrune final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedOpPrune(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedOpPrune() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& Mask = Input(2);
    const auto& b = Input(3);

    CAFFE_ENFORCE_GE(X.dim(), 1);
    CAFFE_ENFORCE_GE(W.dim(), 2);
    if (X.dim() > 2 || W.dim() > 2) {
      VLOG(1) << "Using legacy support for arbitrary input and weight "
                 "dimensions.";
    }
    CAFFE_ENFORCE_EQ(b.dim(), 1);
    // batch size
    int M = X.dim() > 1 ? X.dim32(0) : 1;
    // Feature dimension
    int K = X.numel() / M;
    // number of outputs.
    int N = W.dim32(0);
    CAFFE_ENFORCE_EQ(K, W.numel() / W.dim32(0));
    CAFFE_ENFORCE_EQ(N, b.dim32(0));
    std::vector<int64_t> dims;
    if (X.dim() > 1) {
      dims = {M, N};
    } else {
      dims = {N};
    }
    auto* Y = Output(0, dims, at::dtype<T>());
    // W * x
    math::Gemm<T, Context, Engine>(
        CblasNoTrans,
        CblasTrans,
        M,
        N,
        K,
        1,
        X.template data<T>(),
        W.template data<T>(),
        0,
        Y->template mutable_data<T>(),
        &context_);
    // Add bias term
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M,
      // reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T, Context>(
          M,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    math::Gemm<T, Context, Engine>(
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        1,
        1,
        bias_multiplier_.template data<T>(),
        b.template data<T>(),
        1,
        Y->template mutable_data<T>(),
        &context_);
    if (OutputSize() == 2) {
      auto* Comp_rate = Output(1, vector<int64_t>(), at::dtype<T>());
      T* comp_data = Comp_rate->template mutable_data<T>();
      math::Sum<T, Context>(
          Mask.numel(), Mask.template data<T>(), comp_data, &context_);
      math::Scale<float, T, Context>(
          1,
          static_cast<T>(1.) / Mask.numel(),
          comp_data,
          comp_data,
          &context_);
    }
    return true;
  }

 protected:
  Tensor bias_multiplier_{Context::GetDeviceType()};
};

template <typename T, class Context, class Engine = DefaultEngine>
class FullyConnectedPruneGradientOp : public Operator<Context> {
 public:
  int iter_offset;

 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedPruneGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    iter_offset = 0;
  }
  ~FullyConnectedPruneGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    // const auto& W = Input(1);
    auto* W_ptr = Output(2);
    auto& W = *W_ptr;
    // const auto& Mask = Input(2);
    auto* Mask_ptr = Output(3);
    auto& Mask = *Mask_ptr;
    const auto& dY = Input(3);
    // const auto& Ag_dW = Input(4);
    auto* Ag_dW_ptr = Output(4);
    auto& Ag_dW = *Ag_dW_ptr;
    // it is also the Input(5)

    // how about get threshold
    auto& thres = Input(6);
    // TODO(wyiming): check comp_lb is a float
    auto& comp_lb = Input(7);
    TORCH_DCHECK_GE(X.dim(), 1);
    TORCH_DCHECK_GE(W.dim(), 2);
    TORCH_DCHECK_LE(dY.dim(), 2);
    // batch size
    int M = X.dim() > 1 ? X.dim32(0) : 1;
    // Feature dimension
    int K = X.numel() / M;
    // number of outputs.
    int N = W.dim32(0);
    // TODO(wyiming): add this window_size to workspace?
    int window_size = 100;
    // TODO(wyiming): this threshold should be
    // based on distribution of the layer weight
    float thr = 0.01;
    TORCH_DCHECK_EQ(Mask.dim32(0), W.dim32(0));
    TORCH_DCHECK_EQ(Mask.dim32(1), W.dim32(1));
    TORCH_DCHECK_EQ(Ag_dW.dim32(0), W.dim32(0));
    TORCH_DCHECK_EQ(Ag_dW.dim32(1), W.dim32(1));
    TORCH_DCHECK_EQ(K, W.numel() / W.dim32(0));
    if (dY.dim() > 1) {
      TORCH_DCHECK_EQ(M, dY.dim32(0));
      TORCH_DCHECK_EQ(N, dY.dim32(1));
    } else {
      TORCH_DCHECK_EQ(X.dim(), 1);
      TORCH_DCHECK_EQ(N, dY.numel());
    }

    auto* dW = Output(0, W.sizes(), at::dtype<T>());
    auto* db = Output(1, {N}, at::dtype<T>());

    // Compute dW
    math::Gemm<T, Context, Engine>(
        CblasTrans,
        CblasNoTrans,
        N,
        K,
        M,
        1,
        dY.template data<T>(),
        X.template data<T>(),
        0,
        dW->template mutable_data<T>(),
        &context_);

    comp_r_buf_.Resize(vector<int64_t>());
    T* comp_data = comp_r_buf_.template mutable_data<T>();
    math::Sum<T, Context>(
        Mask.numel(), Mask.template data<T>(), comp_data, &context_);
    math::Scale<float, T, Context>(
        1, static_cast<T>(1.) / Mask.numel(), comp_data, comp_data, &context_);
    // update W size window
    // Notice here we need to maintain state in OP.
    // This is new in Caffe2.
    // And this is something we might need to discuss in the future.
    // at most mask half of the matrix at time
    // 1. mask dw with previous mask
    MaskMatrix<T, Context>(
        Mask.template mutable_data<T>(), dW->template mutable_data<T>(), N, K);
    if (*comp_data > *(comp_lb.template data<T>())) {
      iter_offset++;
      if (iter_offset % window_size == 0) {
        // TODO(wyiming):do the prune here;
        sum_buffer_.ResizeLike(W);
        math::Add<T, Context>(
            W.numel(),
            W.template mutable_data<T>(),
            Ag_dW.template mutable_data<T>(),
            sum_buffer_.template mutable_data<T>(),
            &context_);
        auto* mask_seq_auto = Output(5, W.sizes(), at::dtype<T>());
        T* mask_seq = mask_seq_auto->template mutable_data<T>();
        math::Set<T, Context>(
            N * K,
            static_cast<T>(0),
            mask_seq_auto->template mutable_data<T>(),
            &context_);
        // 2. find dw below thres but not eq 0
        int seq_len = MatrixCompare_LT<T>(
            Ag_dW_ptr->template mutable_data<T>(),
            *thres.template data<T>(),
            mask_seq,
            N,
            K);
        // 3. use the mask_seq to update W and dw
        MaskMatrix_Inc<T, Context>(
            mask_seq, dW->template mutable_data<T>(), N, K, seq_len, 0);
        MaskMatrix_Inc<T, Context>(
            mask_seq, W.template mutable_data<T>(), N, K, seq_len, 0);
        MaskMatrix_Inc<T, Context>(
            mask_seq, Mask.template mutable_data<T>(), N, K, seq_len, 0);
        math::Set<T, Context>(
            N * K,
            static_cast<T>(0),
            Ag_dW.template mutable_data<T>(),
            &context_);
      } else {
        // add dW to Aggregate dW.
        AggrDW<T, Context>(
            Ag_dW.template mutable_data<T>(),
            dW->template mutable_data<T>(),
            N,
            K,
            &context_);
      }
    }
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M,
      // reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T, Context>(
          M,
          static_cast<T>(1),
          bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    // Compute dB
    math::Gemv<T, Context>(
        CblasTrans,
        M,
        N,
        1,
        dY.template data<T>(),
        bias_multiplier_.template data<T>(),
        0,
        db->template mutable_data<T>(),
        &context_);
    // Compute dX if necessary.
    if (OutputSize() == 7) {
      auto* dX = Output(6, X.sizes(), at::dtype<T>());
      math::Gemm<T, Context, Engine>(
          CblasNoTrans,
          CblasNoTrans,
          M,
          K,
          N,
          1,
          dY.template data<T>(),
          W.template data<T>(),
          0,
          dX->template mutable_data<T>(),
          &context_);
    }

    return true;
  }

 protected:
  Tensor bias_multiplier_{Context::GetDeviceType()};
  Tensor sum_buffer_{Context::GetDeviceType()};
  Tensor comp_r_buf_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
