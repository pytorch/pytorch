#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
void wngrad_update(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    const float* lr,
    Context* /*context*/) {
  for (const auto i : c10::irange(N)) {
    float gi = g[i];
    nw[i] = w[i] + lr[0] * gi / (h[0] + epsilon);
  }
  float nhTmp = 0.0;
  for (const auto i : c10::irange(N)) {
    float gi = g[i];
    nhTmp += gi * gi;
  }
  nhTmp /= (h[0] + epsilon);
  nh[0] = h[0] + nhTmp;
}

template <typename Context>
void wngrad_update_output_effective_lr(
    int N,
    const float* paramIn,
    const float* gradIn,
    const float* seqBIn,
    float* paramOut,
    float* seqBOut,
    float* effectiveLROut,
    float epsilon,
    const float* lr,
    Context* /*context*/) {
  effectiveLROut[0] = lr[0] / (seqBIn[0] + epsilon);
  float seqBTmp = 0.0;
  for (const auto i : c10::irange(N)) {
    float gi = gradIn[i];
    seqBTmp += gi * gi;
  }
  seqBTmp /= (seqBIn[0] + epsilon);
  seqBOut[0] = seqBIn[0] + seqBTmp;
  for (const auto i : c10::irange(N)) {
    float grad = gradIn[i];
    paramOut[i] = paramIn[i] + effectiveLROut[0] * grad;
  }
}

template <typename Context>
void wngrad_update_output_effective_lr_and_update(
    int N,
    const float* paramIn,
    const float* gradIn,
    const float* seqBIn,
    float* paramOut,
    float* seqBOut,
    float* effectiveLROut,
    float* updateOut,
    float epsilon,
    const float* lr,
    Context* /*context*/) {
  effectiveLROut[0] = lr[0] / (seqBIn[0] + epsilon);
  float seqBTmp = 0.0;
  for (const auto i : c10::irange(N)) {
    float gi = gradIn[i];
    seqBTmp += gi * gi;
  }
  seqBTmp /= (seqBIn[0] + epsilon);
  seqBOut[0] = seqBIn[0] + seqBTmp;

  for (const auto i : c10::irange(N)) {
    float grad = gradIn[i];
    float update = updateOut[i] = effectiveLROut[0] * grad;
    paramOut[i] = paramIn[i] + update;
  }
}

template <typename T, class Context>
class WngradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  WngradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(
        Input(GRAD).numel(),
        Input(PARAM).numel(),
        "PARAM size: ",
        Input(PARAM).numel(),
        ", GRAD size: ",
        Input(GRAD).numel(),
        ", SEQ_B size: ",
        Input(SEQ_B).numel(),
        ", LR size: ",
        Input(LR).numel());

    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_SEQ_B)->ResizeLike(Input(SEQ_B));
    if (OutputSize() == 2) {
      wngrad_update<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<T>(),
          Input(GRAD).template data<T>(),
          Input(SEQ_B).template data<T>(),
          Output(OUTPUT_PARAM)->template mutable_data<T>(),
          Output(OUTPUT_SEQ_B)->template mutable_data<T>(),
          epsilon_,
          Input(LR).template data<T>(),
          &context_);
    } else if (OutputSize() == 3) {
      Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(SEQ_B));
      wngrad_update_output_effective_lr<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<T>(),
          Input(GRAD).template data<T>(),
          Input(SEQ_B).template data<T>(),
          Output(OUTPUT_PARAM)->template mutable_data<T>(),
          Output(OUTPUT_SEQ_B)->template mutable_data<T>(),
          Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<T>(),
          epsilon_,
          Input(LR).template data<T>(),
          &context_);
    } else {
      Output(OUTPUT_EFFECTIVE_LR)->ResizeLike(Input(SEQ_B));
      Output(OUTPUT_UPDATE)->ResizeLike(Input(GRAD));
      wngrad_update_output_effective_lr_and_update<Context>(
          Input(GRAD).numel(),
          Input(PARAM).template data<T>(),
          Input(GRAD).template data<T>(),
          Input(SEQ_B).template data<T>(),
          Output(OUTPUT_PARAM)->template mutable_data<T>(),
          Output(OUTPUT_SEQ_B)->template mutable_data<T>(),
          Output(OUTPUT_EFFECTIVE_LR)->template mutable_data<T>(),
          Output(OUTPUT_UPDATE)->template mutable_data<T>(),
          epsilon_,
          Input(LR).template data<T>(),
          &context_);
    }

    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, SEQ_B, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_SEQ_B, OUTPUT_EFFECTIVE_LR, OUTPUT_UPDATE);
};

template <typename T, class Context>
class SparseWngradOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseWngradOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(SEQ_B).numel(), 1);
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<T>();
    const auto* seqBIn = Input(SEQ_B).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<T>();
    auto* seqBOut = Output(OUTPUT_SEQ_B)->template mutable_data<T>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).numel() / n;

    for (const auto i : c10::irange(n)) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        paramOut[idx] = paramIn[idx] + lr[0] * gi / (seqBIn[0] + epsilon_);
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
        for (const auto j : c10::irange(block_size)) {
          float gi = gradIn[offsetI + j];
          paramOut[offsetIdx + j] =
              paramIn[offsetIdx + j] + lr[0] * gi / (seqBIn[0] + epsilon_);
        }
      }
    }
    float seqBTmp = 0.0;
    for (const auto i : c10::irange(Input(GRAD).numel())) {
      float gi = gradIn[i];
      seqBTmp += gi * gi;
    }
    seqBTmp /= seqBIn[0];
    seqBOut[0] = seqBTmp + seqBIn[0];
    return true;
  }

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, SEQ_B, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_SEQ_B);
};

} // namespace caffe2
