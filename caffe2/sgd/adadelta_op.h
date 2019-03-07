#include "caffe2/core/operator.h"

namespace caffe2 {

namespace {

template <typename Context>
void AdadeltaUpdate(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* d,
    const float epsilon,
    const float decay,
    const float* lr,
    float* nw,
    float* nh,
    float* nd,
    Context* /*context*/) {
  for (int i = 0; i < N; ++i) {
    float gi = g[i];
    float di = d[i];
    float hi = nh[i] = decay * h[i] + (1.0f - decay) * gi * gi;
    float ng = (std::sqrt(di + epsilon) / std::sqrt(hi + epsilon)) * gi;
    nw[i] = w[i] + lr[0] * ng;
    nd[i] = decay * di + (1.0f - decay) * ng * ng;
  }
}

} // namespace

template <class Context>
class AdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(float, "decay", decay_, 0.95f) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_GRAD).numel());
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(MOMENT_DELTA).numel());
    CAFFE_ENFORCE(Input(GRAD).numel() == Input(PARAM).numel());
    CAFFE_ENFORCE_GE(epsilon_, 0.0f);
    CAFFE_ENFORCE_GT(decay_, 0.0f);
    CAFFE_ENFORCE_LT(decay_, 1.0f);

    Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
    Output(OUTPUT_MOMENT_GRAD)->ResizeLike(Input(MOMENT_GRAD));
    Output(OUTPUT_MOMENT_DELTA)->ResizeLike(Input(MOMENT_DELTA));
    AdadeltaUpdate<Context>(
        Input(GRAD).numel(),
        Input(PARAM).template data<float>(),
        Input(GRAD).template data<float>(),
        Input(MOMENT_GRAD).template data<float>(),
        Input(MOMENT_DELTA).template data<float>(),
        epsilon_,
        decay_,
        Input(LR).template data<float>(),
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT_GRAD)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT_DELTA)->template mutable_data<float>(),
        &context_);
    return true;
  }

 protected:
  const float epsilon_;
  const float decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

template <class Context>
class SparseAdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseAdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(float, "decay", decay_, 0.95f) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_GRAD).numel());
    CAFFE_ENFORCE_EQ(Input(PARAM).numel(), Input(MOMENT_DELTA).numel());
    CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).dim()));

    // Enforce domain constraints for attributes
    CAFFE_ENFORCE_GE(epsilon_, 0.0f);
    CAFFE_ENFORCE_GT(decay_, 0.0f);
    CAFFE_ENFORCE_LT(decay_, 1.0f);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    const auto* lr = Input(LR).template data<float>();
    const auto* indices = Input(INDICES).template data<SIndex>();
    const auto* gradIn = Input(GRAD).template data<float>();
    const auto* paramIn = Input(PARAM).template data<float>();
    const auto* momentIn = Input(MOMENT_GRAD).template data<float>();
    const auto* momentDeltaIn = Input(MOMENT_DELTA).template data<float>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
    auto* momentOut =
        Output(OUTPUT_MOMENT_GRAD)->template mutable_data<float>();
    auto* momentDeltaOut =
        Output(OUTPUT_MOMENT_DELTA)->template mutable_data<float>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }

    auto block_size = Input(GRAD).numel() / n;
    for (int i = 0; i < n; ++i) {
      auto idx = indices[i];
      if (block_size == 1) {
        float gi = gradIn[i];
        float di = momentDeltaIn[idx];
        float hi = momentOut[idx] =
            decay_ * momentIn[idx] + (1.0f - decay_) * gi * gi;
        float ng = (std::sqrt(di + epsilon_) / std::sqrt(hi + epsilon_)) * gi;
        paramOut[idx] = paramIn[idx] + lr[0] * ng;
        momentDeltaOut[idx] = decay_ * di + (1.0f - decay_) * ng * ng;
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
        AdadeltaUpdate(
            block_size,
            paramIn + offsetIdx,
            gradIn + offsetI,
            momentIn + offsetIdx,
            momentDeltaIn + offsetIdx,
            epsilon_,
            decay_,
            lr,
            paramOut + offsetIdx,
            momentOut + offsetIdx,
            momentDeltaOut + offsetIdx,
            &context_);
      }
    }
    return true;
  }

 protected:
  const float epsilon_;
  const float decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

} // namespace caffe2
