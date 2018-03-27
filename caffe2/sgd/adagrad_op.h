#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void adagrad_update(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    const float* lr,
    Context* /*context*/) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + gi * gi;
    nw[i] = w[i] + lr[0] * gi / (std::sqrt(hi) + epsilon);
  }
}

template <typename T, class Context>
class AdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<T>("epsilon", 1e-5f)),
        decay_(OperatorBase::GetSingleArgument<T>("decay", 1.0f)) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).size() == Input(MOMENT_1).size());
    CAFFE_ENFORCE(Input(GRAD).size() == Input(PARAM).size());
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
    adagrad_update<Context>(
        Input(GRAD).size(),
        Input(PARAM).template data<T>(),
        Input(GRAD).template data<T>(),
        Input(MOMENT_1).template data<T>(),
        Output(OUTPUT_PARAM)->template mutable_data<T>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<T>(),
        epsilon_,
        decay_,
        Input(LR).template data<T>(),
        &context_);
    return true;
  }

 protected:
  T epsilon_;
  T decay_;
  INPUT_TAGS(PARAM, MOMENT_1, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T, class Context>
class SparseAdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_1).size());
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).size() / n;
    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = momentOut[idx] = momentIn[idx] + gi * gi;
        paramOut[idx] = paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
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
        adagrad_update(
            block_size,
            paramIn + offsetIdx,
            gradIn + offsetI,
            momentIn + offsetIdx,
            paramOut + offsetIdx,
            momentOut + offsetIdx,
            epsilon_,
            1.0f,
            lr,
            &context_);
      }
    }
    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T, class Context>
class RowWiseSparseAdagradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RowWiseSparseAdagradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).dims()[0], Input(MOMENT_1).size());
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* momentIn = Input(MOMENT_1).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).size() / n;

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float hi = momentOut[idx] = momentIn[idx] + gi * gi;
        paramOut[idx] = paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
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
        float* nw = paramOut + offsetIdx;
        float* nh = momentOut + idx;
        float hs = 0.;
        for (auto j = 0; j < block_size; ++j) {
          float gj = g[j];
          hs += gj * gj;
        }
        float hi = nh[0] = h[0] + hs / block_size;
        float step = lr[0] / (std::sqrt(hi) + epsilon_);
        for (auto j = 0; j < block_size; ++j) {
          nw[j] = w[j] + g[j] * step;
        }
      }
    }
    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};
}
