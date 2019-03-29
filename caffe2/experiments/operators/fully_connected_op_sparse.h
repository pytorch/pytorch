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

#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_SPARSE_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_SPARSE_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif  // CAFFE2_USE_MKL

namespace caffe2 {

namespace {

template<int N>
using Shape = std::array<int, N>;

template<int N>
const std::vector<int64_t>& shape(Shape<N> vs) {
  static thread_local std::vector<int64_t> cache;
  cache.resize(vs.size());
  for (auto i = 0; i < vs.size(); ++i) {
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
void Sparse_mm(const T* acsr, const int* ia, const int* ja,
              int m, int k, int n, const T* b, T* c, Context* context);

template<typename T, class Context>
void trans_mat(const T* o, T* t, int m, int n, Context* context);

template <>
void trans_mat<float, CPUContext>(
    const float* o,
    float* t,
    int m,
    int n,
    CPUContext* /*context*/) {
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      t[j*m+i]=o[i*n+j];
    }
  }
}

// C = A(sparse) * B
// No transpose;
template <>
void Sparse_mm<float, CPUContext>(
    const float* acsr,
    const int* ia,
    const int* ja,
    int m,
    int k,
    int n,
    const float* b,
    float* c,
    CPUContext* /*context*/) {
  float alpha = 1.0, beta = 0.;
  mkl_scsrmm("N", &m, &n, &k, &alpha, "GLNC",
             acsr, ja, ia, ia+1, b, &n, &beta, c, &n);
}

}

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <typename T, class Context, class Engine=DefaultEngine>
class FullyConnectedOp_SPARSE final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedOp_SPARSE(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~FullyConnectedOp_SPARSE() {}

  bool RunOnDevice() override {
    const auto& Xt = Input(0); // transposed X
    const auto& Wcsr = Input(1);
    const auto& iw = Input(2);
    const auto& jw = Input(3);
    // Notice that we do not need to transpose b
    const auto& b = Input(4);
    // transposed Y
    // here we assume X is k-by-m
    CAFFE_ENFORCE_EQ(Xt.dim(), 2);
    CAFFE_ENFORCE_EQ(b.dim(), 1);
    // batch size
    int K = Xt.dim() > 1 ? Xt.dim32(0) : 1;
    // Feature dimension
    int M = Xt.numel() / K;
    // number of outputs.
    int N = iw.dim32(0)-1;
    CAFFE_ENFORCE_EQ(N, b.dim32(0));
    auto* Yt = Output(0, shape(N, M), at::dtype<T>());

    // Y' = W * X';
    Sparse_mm<T, Context>(
      Wcsr.template data<T>(), iw.template data<int>(),
      jw.template data<int>(), N, K, M, Xt.template data<T>(),
      Yt->template mutable_data<T>(), &context_);
    // Add bias term
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(shape(M));
      math::Set<T, Context>(
          M, static_cast<T>(1), bias_multiplier_.template mutable_data<T>(),
          &context_);
    }
    math::Gemm<T, Context, Engine>(
        CblasNoTrans, CblasNoTrans, N, M, 1, 1,
        b.template data<T>(), bias_multiplier_.template data<T>(), 1,
        Yt->template mutable_data<T>(), &context_);
    return true;
  }

 protected:
  Tensor bias_multiplier_{Context::GetDeviceType()};
};


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
