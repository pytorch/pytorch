#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void storm_update(
    const int N,
    const float* paramIn,
    const float* momentIn,
    const float* gradSqSumIn,
    const float* gradIn,
    const float* lr,
    float* paramOut,
    float* momentOut,
    float* gradSqSumOut,
    const float momentum,
    const float beta,
    Context* /*context*/) {
  float gradSqSumTmp = 0.0;
  for (auto i = 0; i < N; ++i) {
    const float gi = gradIn[i];
    gradSqSumTmp += gi * gi;
  }
  gradSqSumOut[0] = gradSqSumIn[0] + gradSqSumTmp;

  const float nlr = lr[0] * std::pow(beta + gradSqSumOut[0], -1.0 / 3.0);
  const float alpha = momentum * nlr * nlr;
  for (auto i = 0; i < N; ++i) {
    const float gi = gradIn[i];
    const float mi = momentIn[i];
    float new_mi = momentOut[i] = gi + (1.0 - alpha) * (mi - gi);
    paramOut[i] = paramIn[i] + nlr * new_mi;
  }
}

template <class Context>
class StormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  StormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "momentum", momentum_, 10.0),
        OP_SINGLE_ARG(float, "beta", beta_, 0.1) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(PARAM).numel());
    CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(MOMENT).numel());
    CAFFE_ENFORCE_EQ(Input(GRADSQSUM).numel(), 1);
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);

    // Resize [potentially] out-of-place blobs
    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT)->ResizeLike(Input(MOMENT));
    Output(OUTPUT_GRAGSQSUM)->ResizeLike(Input(GRADSQSUM));

    storm_update<Context>(
        Input(GRAD).numel(),
        Input(PARAM).template data<float>(),
        Input(MOMENT).template data<float>(),
        Input(GRADSQSUM).template data<float>(),
        Input(GRAD).template data<float>(),
        Input(LR).template data<float>(),
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT)->template mutable_data<float>(),
        Output(OUTPUT_GRAGSQSUM)->template mutable_data<float>(),
        momentum_,
        beta_,
        &context_);
    return true;
  }

 protected:
  const float momentum_;
  const float beta_;
  INPUT_TAGS(PARAM, MOMENT, GRADSQSUM, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT, OUTPUT_GRAGSQSUM);
};

template <class Context>
class SparseStormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseStormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "momentum", momentum_, 10.0),
        OP_SINGLE_ARG(float, "beta", beta_, 0.1) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT).numel());
    CAFFE_ENFORCE_EQ(Input(GRADSQSUM).numel(), 1);
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* paramIn = Input(PARAM).template data<float>();
    const auto* momentIn = Input(MOMENT).template data<float>();
    const auto* gradSqSumIn = Input(GRADSQSUM).template data<float>();
    const auto* gradIn = Input(GRAD).template data<float>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* lr = Input(LR).template data<float>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
    auto* momentOut = Output(OUTPUT_MOMENT)->template mutable_data<float>();
    auto* gradSqSumOut =
        Output(OUTPUT_GRAGSQSUM)->template mutable_data<float>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }

    float gradSqSumTmp = 0.0;
    for (auto i = 0; i < Input(GRAD).numel(); ++i) {
      const float gi = gradIn[i];
      gradSqSumTmp += gi * gi;
    }
    gradSqSumOut[0] = gradSqSumIn[0] + gradSqSumTmp;

    const float nlr = lr[0] * std::pow(beta_ + gradSqSumOut[0], -1.0 / 3.0);
    const float alpha = momentum_ * nlr * nlr;
    const auto block_size = Input(GRAD).numel() / n;

    for (auto i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        const float gi = gradIn[i];
        const float mi = momentIn[idx];
        float new_mi = momentOut[idx] = gi + (1.0 - alpha) * (mi - gi);
        paramOut[idx] = paramIn[idx] + nlr * new_mi;
      } else {
        auto offsetI = i * block_size;
        auto offsetIdx = idx * block_size;

#ifndef NDEBUG
        CAFFE_ENFORCE_GE(
            Input(PARAM).numel(),
            block_size + offsetIdx,
            this->debug_def().input(PARAM),
            ", out of bound,  idx:",
            idx,
            " for input i:",
            i,
            " and block size:",
            block_size);
        CAFFE_ENFORCE_GE(
            Input(GRAD).numel(),
            block_size + offsetI,
            this->debug_def().input(GRAD),
            ", out of bound idx, idx:",
            idx,
            " for input i:",
            i);
#endif

        for (auto j = 0; j < block_size; ++j) {
          const float gi = gradIn[offsetI + j];
          const float mi = momentIn[offsetIdx + j];
          float new_mi = momentOut[offsetIdx + j] =
              gi + (1.0 - alpha) * (mi - gi);
          paramOut[offsetIdx + j] = paramIn[offsetIdx + j] + nlr * new_mi;
        }
      }
    }

    return true;
  }

 protected:
  const float momentum_;
  const float beta_;
  INPUT_TAGS(PARAM, MOMENT, GRADSQSUM, GRAD, INDICES, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT, OUTPUT_GRAGSQSUM);
};
} // namespace caffe2
