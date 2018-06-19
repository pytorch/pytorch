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

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void adadelta_update(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* p,
    float* nw,
    float* nh,
    float* np,
    float epsilon,
    float decay,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + (1.0f - decay) * gi * gi;
    float pi = gi * (std::sqrt(p[i]) + epsilon) / (std::sqrt(hi) + epsilon);
    np[i] = decay * p[i] + (1.0f - decay) * pi * pi;
    nw[i] = w[i] - pi;
  }
}

template <typename T, class Context>
class AdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<T>("epsilon", 1e-5f)),
        decay_(OperatorBase::GetSingleArgument<T>("decay", 1.0f)) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENT_GRAD).size());
    CAFFE_ENFORCE(Input(PARAM).size() == Input(MOMENT_DELTA).size());
    CAFFE_ENFORCE(Input(GRAD).size() == Input(PARAM).size());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_GRAD)->ResizeLike(Input(MOMENT_GRAD));
    Output(OUTPUT_MOMENT_DELTA)->ResizeLike(Input(MOMENT_DELTA));
    adadelta_update<Context>(
        Input(GRAD).size(),
        Input(PARAM).template data<T>(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_GRAD).template data<T>(),
        Input(MOMENT_DELTA).template data<T>(),
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_GRAD)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_DELTA)->template mutable_data<T>(),
        epsilon_,
        decay_,
        &context_);
    return true;
  }

 protected:
  T epsilon_;
  T decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, GRAD);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

template <typename T, class Context>
class SparseAdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)),
        decay_(OperatorBase::GetSingleArgument<T>("decay", 1.0f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_GRAD).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_DELTA).size());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* momentIn = Input(MOMENT_GRAD).template data<T>();
    const auto* moment2In = Input(MOMENT_DELTA).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* momentOut = Output(OUTPUT_MOMENT_GRAD)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_DELTA)->template mutable_data<T>();

    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).size() / n;
    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = momentOut[idx] =
            decay_ * momentIn[idx] + (T(1.) - decay_) * gi * gi;
        float pi = (std::sqrt(moment2In[idx]) + epsilon_) * gi /
            (std::sqrt(hi) + epsilon_);
        moment2Out[idx] = decay_ * moment2In[idx] + (T(1.) - decay_) * pi * pi;
        paramOut[idx] = paramIn[idx] - pi;
      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;

#ifndef NDEBUG
        CAFFE_ENFORCE_GE(
            Input(PARAM).size(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input i:",
            i,
            " and block size:",
            block_size);
        CAFFE_ENFORCE_GE(
            Input(GRAD).size(),
            block_size + offsetI,
            this->debug_def().input(GRAD),
            ", out of bound idx, idx:",
            idx,
            " for input i:",
            i);
#endif
        adadelta_update(
            block_size,
            paramIn + offsetIdx,
            gradIn + offsetI,
            momentIn + offsetIdx,
            moment2In + offsetIdx,
            paramOut + offsetIdx,
            momentOut + offsetIdx,
            moment2Out + offsetIdx,
            epsilon_,
            decay_,
            &context_);
      }
    }
    return true;
  }

 protected:
  T epsilon_;
  T decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, INDICES, GRAD);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

template <typename T, class Context>
class RowWiseSparseAdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RowWiseSparseAdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)),
        decay_(OperatorBase::GetSingleArgument<T>("decay", 1.0f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).dims()[0], Input(MOMENT_GRAD).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).dims()[0], Input(MOMENT_DELTA).size());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* momentIn = Input(MOMENT_GRAD).template data<T>();
    const auto* moment2In = Input(MOMENT_DELTA).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* momentOut = Output(OUTPUT_MOMENT_GRAD)->template mutable_data<T>();
    auto* moment2Out = Output(OUTPUT_MOMENT_DELTA)->template mutable_data<T>();

    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).size() / n;

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = momentOut[idx] =
            decay_ * momentIn[idx] + (T(1.) - decay_) * gi * gi;
        float pi = (std::sqrt(moment2In[idx]) + epsilon_) * gi /
            (std::sqrt(hi) + epsilon_);
        moment2Out[idx] = decay_ * moment2In[idx] + (T(1.) - decay_) * pi * pi;
        paramOut[idx] = paramIn[idx] - pi;
      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;

#ifndef NDEBUG
        CAFFE_ENFORCE_GE(
            Input(PARAM).size(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input i:",
            i,
            " and block size:",
            block_size);
        CAFFE_ENFORCE_GE(
            Input(GRAD).size(),
            block_size + offsetI,
            this->debug_def().input(GRAD),
            ", out of bound idx, idx:",
            idx,
            " for input i:",
            i);
#endif

        const float* w = paramIn + offsetIdx;
        const float* g = gradIn + offsetI;
        const float* h = momentIn + idx;
        const float* p = moment2In + idx;
        float* nw = paramOut + offsetIdx;
        float* nh = momentOut + idx;
        float* np = moment2Out + idx;
        float hs = 0.;
        for (auto j = 0; j < block_size; ++j) {
          float gj = g[j];
          hs += gj * gj;
        }
        float hi = nh[0] = decay_ * h[0] + (T(1.) - decay_) * hs / block_size;
        float step = (std::sqrt(p[0]) + epsilon_) / (std::sqrt(hi) + epsilon_);
        float ps = 0;
        for (auto j = 0; j < block_size; ++j) {
          float delta_p = g[j] * step;
          ps += delta_p * delta_p;
          nw[j] = w[j] - delta_p;
        }
        np[0] = decay_ * p[0] + (T(1.) - decay_) * ps / block_size;
      }
    }
    return true;
  }

 protected:
  T epsilon_;
  T decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, INDICES, GRAD);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};
} // namespace caffe2
